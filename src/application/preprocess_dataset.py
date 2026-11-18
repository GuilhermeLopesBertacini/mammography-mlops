"""Caso de uso para pré-processamento offline, tiling e geração de cache/manifesto."""

from __future__ import annotations

import csv
import json
import os
import random
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.application.tiling_engine import TilingEngine
from src.domain.bounding_box import BoundingBox
from src.domain.entities import MammogramImage, PatientStudy
from src.domain.enums import DatasetSplit, PhotometricInterpretation
from src.ports.dataset_reader import DatasetReaderPort
from src.ports.image_preprocessor import ImagePreprocessorPort


@dataclass(frozen=True)
class PatchRecord:
    """Metadados de um patch salvo em disco para indexação no manifest.csv."""

    patch_path: str
    study_id: str
    image_id: str
    split: str
    has_lesion: bool
    bboxes_json: str


@dataclass(frozen=True)
class _ImageTask:
    """Payload leve e serializável (pickleable) enviado aos workers."""

    dicom_path: Path
    study_id: str
    image_id: str
    split: str
    photometric: PhotometricInterpretation
    bboxes: tuple[tuple[float, float, float, float], ...]
    cache_dir: Path
    preprocessor: ImagePreprocessorPort
    tiling_engine: TilingEngine


def _write_png_atomic(rgb_array: npt.NDArray[np.uint8], target_path: Path) -> None:
    """Grava PNG em arquivo temporário e substitui atomicamente no filesystem."""
    tmp_path = target_path.with_suffix(".png.tmp")
    Image.fromarray(rgb_array).save(tmp_path, format="PNG")
    os.replace(tmp_path, target_path)


def _write_json_atomic(payload: object, target_path: Path) -> None:
    """Grava JSON em arquivo temporário e substitui atomicamente."""
    tmp_path = target_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp_path, target_path)


def _process_single_image(task: _ImageTask) -> list[PatchRecord]:
    """Executa pré-processamento + tiling para 1 imagem com suporte a retomada idempotente."""
    image_cache_dir = task.cache_dir / "patches" / task.study_id / task.image_id
    meta_path = image_cache_dir / "_meta.json"

    # Idempotência: se _meta.json existe, todos os PNGs dessa imagem já foram gravados atomicamente
    if meta_path.is_file():
        raw_records = json.loads(meta_path.read_text(encoding="utf-8"))
        return [PatchRecord(**item) for item in raw_records]

    image_cache_dir.mkdir(parents=True, exist_ok=True)
    result = task.preprocessor.preprocess_dicom(task.dicom_path, task.photometric)

    # Projeta BBoxes originais para o espaço do crop antes do tiling
    cropped_bboxes: list[BoundingBox] = []
    for xmin, ymin, xmax, ymax in task.bboxes:
        box = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        clipped = box.clip_to_patch(
            result.crop_xmin,
            result.crop_ymin,
            result.crop_xmax,
            result.crop_ymax,
        )
        if clipped is not None:
            cropped_bboxes.append(clipped)

    img_h, img_w = result.image_rgb.shape[:2]
    patches = task.tiling_engine.generate_patches(
        image_width=img_w,
        image_height=img_h,
        target_bboxes=cropped_bboxes,
    )

    records: list[PatchRecord] = []
    for patch in patches:
        x1, y1 = int(patch.xmin), int(patch.ymin)
        x2, y2 = int(patch.xmax), int(patch.ymax)
        patch_rgb = result.image_rgb[y1:y2, x1:x2]

        rel_path = Path("patches") / task.study_id / task.image_id / f"patch_{patch.patch_id:03d}.png"
        _write_png_atomic(patch_rgb, task.cache_dir / rel_path)

        # Fase 1 binária: label = 1 (lesion)
        boxes_payload = [
            [b.xmin, b.ymin, b.xmax, b.ymax, 1] for b in patch.local_bboxes
        ]
        records.append(
            PatchRecord(
                patch_path=str(rel_path),
                study_id=task.study_id,
                image_id=task.image_id,
                split=task.split,
                has_lesion=len(boxes_payload) > 0,
                bboxes_json=json.dumps(boxes_payload),
            )
        )

    # Marcador gravado por último: garante consistência em caso de interrupção
    _write_json_atomic([asdict(r) for r in records], meta_path)
    return records


class PreprocessDatasetUseCase:
    """Orquestra split anti-leakage por study_id, filtragem e cache offline em paralelo."""

    def __init__(
        self,
        reader: DatasetReaderPort,
        preprocessor: ImagePreprocessorPort,
        tiling_engine: TilingEngine,
    ) -> None:
        self.reader = reader
        self.preprocessor = preprocessor
        self.tiling_engine = tiling_engine

    def execute(
        self,
        raw_images_dir: Path,
        annotations_dir: Path,
        cache_dir: Path,
        val_fraction: float = 0.15,
        seed: int = 42,
        exclude_birads_from_negatives: Sequence[str] = ("BI-RADS 3", "BI-RADS 4", "BI-RADS 5"),
        max_negative_studies: int | None = 500,
        max_workers: int | None = None,
    ) -> Path:
        """Executa o pipeline offline e retorna o caminho do manifest.csv gerado."""
        if not raw_images_dir.is_dir():
            raise FileNotFoundError(f"raw_images_dir não encontrado: {raw_images_dir}")

        cache_dir.mkdir(parents=True, exist_ok=True)
        studies = self.reader.load_studies(annotations_dir)
        split_assignments = self._assign_splits(
            studies=studies,
            val_fraction=val_fraction,
            seed=seed,
            exclude_birads=set(exclude_birads_from_negatives),
            max_negative_studies=max_negative_studies,
        )

        tasks = self._build_tasks(
            studies=studies,
            split_assignments=split_assignments,
            raw_images_dir=raw_images_dir,
            cache_dir=cache_dir,
        )

        workers = max_workers if max_workers is not None else min(os.cpu_count() or 4, 8)
        all_records: list[PatchRecord] = []

        if workers <= 1:
            for task in tasks:
                all_records.extend(_process_single_image(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for image_records in executor.map(_process_single_image, tasks):
                    all_records.extend(image_records)

        manifest_path = cache_dir / "manifest.csv"
        self._write_manifest_atomic(all_records, manifest_path)
        return manifest_path

    def _assign_splits(
        self,
        studies: list[PatientStudy],
        val_fraction: float,
        seed: int,
        exclude_birads: set[str],
        max_negative_studies: int | None,
    ) -> dict[str, str]:
        """Divide estudos preservando test oficial e agrupando estritamente por study_id."""
        rng = random.Random(seed)
        pos_train: list[PatientStudy] = []
        neg_train: list[PatientStudy] = []
        pos_test: list[PatientStudy] = []
        neg_test: list[PatientStudy] = []

        for study in studies:
            has_lesion = any(bool(img.bboxes) for img in study.images)
            if has_lesion:
                (pos_test if study.split == DatasetSplit.TEST else pos_train).append(study)
            else:
                has_excluded = any(img.breast_birads.value in exclude_birads for img in study.images)
                if has_excluded:
                    continue
                (neg_test if study.split == DatasetSplit.TEST else neg_train).append(study)

        # Amostragem determinística de estudos negativos caso max_negative_studies esteja definido
        if max_negative_studies is not None:
            total_neg = len(neg_train) + len(neg_test)
            if total_neg > max_negative_studies:
                test_share = round(max_negative_studies * (len(neg_test) / max(1, total_neg)))
                train_share = max_negative_studies - test_share
                rng.shuffle(neg_train)
                rng.shuffle(neg_test)
                neg_train = neg_train[:train_share]
                neg_test = neg_test[:test_share]

        assignments: dict[str, str] = {}
        for study in pos_test + neg_test:
            assignments[study.study_id] = "test"

        # Estratifica train/val separadamente para positivos e negativos por study_id
        for group in (pos_train, neg_train):
            shuffled = list(group)
            rng.shuffle(shuffled)
            n_val = round(len(shuffled) * val_fraction)
            for idx, study in enumerate(shuffled):
                assignments[study.study_id] = "val" if idx < n_val else "train"

        return assignments

    def _build_tasks(
        self,
        studies: list[PatientStudy],
        split_assignments: dict[str, str],
        raw_images_dir: Path,
        cache_dir: Path,
    ) -> list[_ImageTask]:
        tasks: list[_ImageTask] = []
        for study in studies:
            split = split_assignments.get(study.study_id)
            if split is None:
                continue

            for img in study.images:
                dicom_path = self._resolve_dicom_path(raw_images_dir, study.study_id, img)
                bboxes = tuple(
                    (b.xmin, b.ymin, b.xmax, b.ymax)
                    for b in img.bboxes
                )
                tasks.append(
                    _ImageTask(
                        dicom_path=dicom_path,
                        study_id=study.study_id,
                        image_id=img.image_id,
                        split=split,
                        photometric=img.photometric_interpretation,
                        bboxes=bboxes,
                        cache_dir=cache_dir,
                        preprocessor=self.preprocessor,
                        tiling_engine=self.tiling_engine,
                    )
                )
        return tasks

    def _resolve_dicom_path(
        self,
        raw_images_dir: Path,
        study_id: str,
        img: MammogramImage,
    ) -> Path:
        """Resolve extensão .dicom ou .dcm dentro da pasta do estudo."""
        for ext in (".dicom", ".dcm"):
            candidate = raw_images_dir / study_id / f"{img.image_id}{ext}"
            if candidate.is_file():
                return candidate
        return raw_images_dir / study_id / f"{img.image_id}.dicom"

    def _write_manifest_atomic(self, records: list[PatchRecord], manifest_path: Path) -> None:
        tmp_path = manifest_path.with_suffix(".csv.tmp")
        fieldnames = ["patch_path", "study_id", "image_id", "split", "has_lesion", "bboxes_json"]
        with tmp_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))
        os.replace(tmp_path, manifest_path)
