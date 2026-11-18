"""Contratos Pydantic para validação Fail-Fast das configurações Hydra."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictBaseConfig(BaseModel):
    """Base imutável que rejeita chaves desconhecidas vindas de typos em YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_StrictBaseConfig):
    """Configuração de caminhos de entrada, cache e balanceamento de amostragem."""

    raw_images_dir: Path
    annotations_csv: Path
    cache_dir: Path
    val_fraction: float = Field(gt=0.0, lt=0.5)
    positive_negative_ratio: float = Field(ge=0.0, le=5.0)
    max_negative_studies: int | None = Field(default=500, ge=0)
    exclude_birads_from_negatives: tuple[str, ...] = ("BI-RADS 3", "BI-RADS 4", "BI-RADS 5")
    num_workers: int = Field(ge=0, le=32)
    pin_memory: bool = True

    @field_validator("annotations_csv")
    @classmethod
    def _validate_annotations_csv_exists(cls, value: Path) -> Path:
        if not value.is_file():
            raise ValueError(f"annotations_csv não encontrado em: {value}")
        return value

    def require_raw_images_dir(self) -> Path:
        """Valida existência do diretório de DICOMs brutos (usado no pré-processamento)."""
        if not self.raw_images_dir.is_dir():
            raise FileNotFoundError(f"raw_images_dir não encontrado: {self.raw_images_dir}")
        return self.raw_images_dir


class PreprocessingConfig(_StrictBaseConfig):
    """Parâmetros de recorte, janelamento CLAHE e fatiamento (TilingEngine)."""

    patch_size: int = Field(gt=0)
    overlap_ratio: float = Field(ge=0.0, lt=1.0)
    min_bbox_coverage: float = Field(gt=0.0, le=1.0)
    crop_black_background: bool = True
    clahe_clip_limit: float = Field(gt=0.0, le=10.0)
    clahe_tile_grid_size: int = Field(ge=2, le=64)

    @field_validator("patch_size")
    @classmethod
    def _validate_patch_size_multiple_of_32(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError(
                f"patch_size deve ser múltiplo de 32 (stride máximo da FPN/ConvNeXt), recebido: {value}"
            )
        return value


class ModelConfig(_StrictBaseConfig):
    """Arquitetura do detector (Backbone timm + FPN + Cascade R-CNN)."""

    num_classes: int = Field(ge=2)
    backbone_name: str = Field(min_length=1)
    pretrained: bool = True
    fpn_out_channels: int = Field(ge=64)
    cascade_iou_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7)
    norm_type: Literal["group_norm", "layer_norm", "none"] = "group_norm"
    num_groups: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0, lt=1.0)
    weight_decay: float = Field(ge=0.0, lt=1.0)
    warmup_epochs: int = Field(ge=0)

    @field_validator("cascade_iou_thresholds")
    @classmethod
    def _validate_increasing_iou_thresholds(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) < 2:
            raise ValueError("Cascade R-CNN requer ao menos 2 estágios de IoU.")
        if any(not (0.0 < t < 1.0) for t in value):
            raise ValueError(f"Limiares de IoU devem estar em (0, 1): {value}")
        if list(value) != sorted(value) or len(set(value)) != len(value):
            raise ValueError(f"Limiares de IoU devem ser estritamente crescentes: {value}")
        return value

    @model_validator(mode="after")
    def _validate_group_norm_divisibility(self) -> "ModelConfig":
        if self.norm_type == "group_norm" and self.fpn_out_channels % self.num_groups != 0:
            raise ValueError(
                f"fpn_out_channels ({self.fpn_out_channels}) deve ser divisível por num_groups ({self.num_groups})."
            )
        return self


class TrainerConfig(_StrictBaseConfig):
    """Parâmetros de hardware, precisão e acúmulo de gradientes."""

    accelerator: Literal["gpu", "cpu", "auto"] = "gpu"
    devices: int = Field(ge=1)
    precision: Literal["bf16-mixed", "16-mixed", "32-true"] = "bf16-mixed"
    batch_size: int = Field(ge=1)
    accumulate_grad_batches: int = Field(ge=1)
    max_epochs: int = Field(ge=1)
    gradient_clip_val: float = Field(ge=0.0)
    log_every_n_steps: int = Field(ge=1)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulate_grad_batches


class TrackingConfig(_StrictBaseConfig):
    """Metadados de rastreamento no MLflow (credenciais vêm do ambiente/.env)."""

    experiment_name: str = Field(min_length=1)
    run_name: str | None = None
    log_model: bool = False
    tags: dict[str, str] = Field(default_factory=dict)


class PipelineConfig(_StrictBaseConfig):
    """Raiz validada do pipeline completo."""

    seed: int = Field(ge=0)
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    trainer: TrainerConfig
    tracking: TrackingConfig
