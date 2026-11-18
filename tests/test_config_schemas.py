"""Testes unitários para validação Fail-Fast dos schemas Pydantic e YAMLs Hydra."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.infrastructure.config import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    PreprocessingConfig,
    TrackingConfig,
    TrainerConfig,
)


def _valid_data_kwargs(tmp_path: Path) -> dict:
    csv_path = tmp_path / "finding_annotations.csv"
    csv_path.write_text("study_id,image_id\n", encoding="utf-8")
    return {
        "raw_images_dir": tmp_path / "images",
        "annotations_csv": csv_path,
        "cache_dir": tmp_path / "cache",
        "val_fraction": 0.15,
        "positive_negative_ratio": 1.0,
        "num_workers": 4,
    }


def test_repo_yaml_configs_match_pydantic_schemas() -> None:
    """Garante que os YAMLs reais em configs/ passam na validação sem erros."""
    root = Path("configs")
    config_root = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    data_cfg = yaml.safe_load((root / "data/vindr.yaml").read_text(encoding="utf-8"))
    prep_cfg = yaml.safe_load((root / "preprocessing/default.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((root / "model/cascade_convnext.yaml").read_text(encoding="utf-8"))
    trainer_cfg = yaml.safe_load((root / "trainer/default.yaml").read_text(encoding="utf-8"))
    tracking_cfg = yaml.safe_load((root / "tracking/mlflow.yaml").read_text(encoding="utf-8"))

    tracker_fallback = Path("data/finding_annotations.csv")
    if not Path(data_cfg["annotations_csv"]).is_file() and tracker_fallback.is_file():
        data_cfg["annotations_csv"] = str(tracker_fallback)

    pipeline = PipelineConfig(
        seed=config_root["seed"],
        data=DataConfig(**data_cfg),
        preprocessing=PreprocessingConfig(**prep_cfg),
        model=ModelConfig(**model_cfg),
        trainer=TrainerConfig(**trainer_cfg),
        tracking=TrackingConfig(**tracking_cfg),
    )

    assert pipeline.trainer.effective_batch_size == 32
    assert pipeline.model.num_classes == 2


def test_data_config_fails_when_annotations_csv_missing(tmp_path: Path) -> None:
    kwargs = _valid_data_kwargs(tmp_path)
    kwargs["annotations_csv"] = tmp_path / "nao_existe.csv"

    with pytest.raises(ValidationError, match="annotations_csv"):
        DataConfig(**kwargs)


def test_preprocessing_rejects_patch_size_not_multiple_of_32() -> None:
    with pytest.raises(ValidationError, match="múltiplo de 32"):
        PreprocessingConfig(
            patch_size=1000,
            overlap_ratio=0.2,
            min_bbox_coverage=0.3,
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=8,
        )


def test_model_rejects_non_increasing_cascade_iou_thresholds() -> None:
    with pytest.raises(ValidationError, match="estritamente crescentes"):
        ModelConfig(
            num_classes=2,
            backbone_name="convnext_base.fb_in22k_ft_in1k_384",
            fpn_out_channels=256,
            cascade_iou_thresholds=(0.6, 0.5, 0.7),
            norm_type="group_norm",
            num_groups=32,
            learning_rate=1e-4,
            weight_decay=1e-4,
            warmup_epochs=1,
        )


def test_strict_config_forbids_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainerConfig(
            batch_size=2,
            accumulate_grad_batches=16,
            max_epochs=20,
            gradient_clip_val=1.0,
            log_every_n_steps=10,
            typo_key=123,  # type: ignore[call-arg]
        )
