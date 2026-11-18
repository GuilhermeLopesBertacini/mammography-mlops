"""Testes unitários do PreprocessDatasetUseCase (split, filtros e idempotência)."""

import csv
from pathlib import Path

import numpy as np

from src.application.preprocess_dataset import PreprocessDatasetUseCase
from src.application.tiling_engine import TilingEngine
from src.domain.bounding_box import BoundingBox
from src.domain.entities import FindingAnnotation, MammogramImage, PatientStudy
from src.domain.enums import (
    BiRadsScore,
    BreastDensity,
    DatasetSplit,
    Laterality,
    PhotometricInterpretation,
    ViewPosition,
)
from src.ports.dataset_reader import DatasetReaderPort
from src.ports.image_preprocessor import ImagePreprocessorPort, PreprocessResult


class _FakeReader(DatasetReaderPort):
    def __init__(self, studies: list[PatientStudy]) -> None:
        self._studies = studies

    def load_studies(self, data_dir: Path) -> list[PatientStudy]:
        return self._studies


class _FakePreprocessor(ImagePreprocessorPort):
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def preprocess_dicom(
        self,
        dicom_path: Path,
        photometric: PhotometricInterpretation,
    ) -> PreprocessResult:
        self.calls.append(dicom_path)
        return self.preprocess_array(np.zeros((1024, 1024), dtype=np.float32), photometric)

    def preprocess_array(
        self,
        raw_array: np.ndarray,
        photometric: PhotometricInterpretation,
    ) -> PreprocessResult:
        return PreprocessResult(
            image_rgb=np.zeros((1024, 1024, 3), dtype=np.uint8),
            crop_xmin=0,
            crop_ymin=0,
            crop_xmax=1024,
            crop_ymax=1024,
        )


def _make_study(
    study_id: str,
    split: DatasetSplit,
    birads: BiRadsScore,
    with_bbox: bool,
) -> PatientStudy:
    study = PatientStudy(study_id=study_id, split=split)
    for view in (ViewPosition.CC, ViewPosition.MLO):
        img = MammogramImage(
            image_id=f"{study_id}_{view.value}",
            study_id=study_id,
            series_id="s1",
            laterality=Laterality.LEFT,
            view_position=view,
            width=1024,
            height=1024,
            breast_birads=birads,
            breast_density=BreastDensity.DENSITY_B,
            split=split,
            photometric_interpretation=PhotometricInterpretation.MONOCHROME2,
        )
        if with_bbox:
            img.add_finding(
                FindingAnnotation(
                    category="Mass",
                    birads=birads,
                    bbox=BoundingBox(xmin=100.0, ymin=100.0, xmax=300.0, ymax=300.0),
                )
            )
        study.add_image(img)
    return study


def test_use_case_enforces_study_split_excludes_birads345_negatives_and_is_idempotent(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cache_dir = tmp_path / "cache"

    studies = [
        _make_study("pos_1", DatasetSplit.TRAINING, BiRadsScore.BI_RADS_4, with_bbox=True),
        _make_study("pos_2", DatasetSplit.TRAINING, BiRadsScore.BI_RADS_5, with_bbox=True),
        _make_study("pos_test", DatasetSplit.TEST, BiRadsScore.BI_RADS_4, with_bbox=True),
        _make_study("neg_clean_1", DatasetSplit.TRAINING, BiRadsScore.BI_RADS_1, with_bbox=False),
        _make_study("neg_clean_2", DatasetSplit.TRAINING, BiRadsScore.BI_RADS_2, with_bbox=False),
        # Deve ser excluído dos negativos por ser BI-RADS 4 sem bbox
        _make_study("neg_noisy_b4", DatasetSplit.TRAINING, BiRadsScore.BI_RADS_4, with_bbox=False),
    ]

    preprocessor = _FakePreprocessor()
    use_case = PreprocessDatasetUseCase(
        reader=_FakeReader(studies),
        preprocessor=preprocessor,
        tiling_engine=TilingEngine(patch_size=1024, overlap_ratio=0.2),
    )

    manifest_path = use_case.execute(
        raw_images_dir=raw_dir,
        annotations_dir=tmp_path,
        cache_dir=cache_dir,
        val_fraction=0.5,
        seed=42,
        max_negative_studies=1,
        max_workers=1,
    )

    assert manifest_path.is_file()
    with manifest_path.open("r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    study_splits: dict[str, set[str]] = {}
    for row in rows:
        study_splits.setdefault(row["study_id"], set()).add(row["split"])
        assert (cache_dir / row["patch_path"]).is_file()

    # Invariante 1: nenhum study_id pode aparecer em mais de um split
    assert all(len(s) == 1 for s in study_splits.values())
    # Invariante 2: BI-RADS 4 sem bbox foi excluído
    assert "neg_noisy_b4" not in study_splits
    # Invariante 3: test oficial preservado
    assert study_splits["pos_test"] == {"test"}
    # Invariante 4: max_negative_studies=1 limitou os negativos (3 pos + 1 neg = 4 estudos * 2 imgs = 8 calls)
    assert len(preprocessor.calls) == 8

    # Segunda execução: 100% idempotente (0 novas chamadas a preprocess_dicom)
    preprocessor.calls.clear()
    use_case.execute(
        raw_images_dir=raw_dir,
        annotations_dir=tmp_path,
        cache_dir=cache_dir,
        val_fraction=0.5,
        seed=42,
        max_negative_studies=1,
        max_workers=1,
    )
    assert len(preprocessor.calls) == 0

    # Terceira execução: simula _meta.json corrompido (0 bytes) e valida auto-recuperação
    first_meta = next((cache_dir / "patches").rglob("_meta.json"))
    first_meta.write_text("", encoding="utf-8")
    preprocessor.calls.clear()
    use_case.execute(
        raw_images_dir=raw_dir,
        annotations_dir=tmp_path,
        cache_dir=cache_dir,
        val_fraction=0.5,
        seed=42,
        max_negative_studies=1,
        max_workers=1,
    )
    assert len(preprocessor.calls) == 1
    assert first_meta.stat().st_size > 0
