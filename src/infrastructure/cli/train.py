"""CLI Hydra (Composition Root) para treinamento com PyTorch Lightning e MLflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from src.infrastructure.config import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    PreprocessingConfig,
    TrackingConfig,
    TrainerConfig,
)
from src.infrastructure.datasets.vindr_datamodule import VinDrLightningDataModule
from src.infrastructure.evaluation import DetectionReportCallback
from src.infrastructure.models import CascadeRCNNLightningModule, TimmFPNBackbone

logger = logging.getLogger(__name__)


def _load_dotenv(env_path: Path) -> None:
    """Carrega variáveis do .env para o ambiente sem sobrescrever variáveis já exportadas."""
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _validate_config(cfg: DictConfig) -> PipelineConfig:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Configuração Hydra raiz deve ser um mapeamento.")

    return PipelineConfig(
        seed=raw["seed"],
        data=DataConfig(**raw["data"]),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        model=ModelConfig(**raw["model"]),
        trainer=TrainerConfig(**raw["trainer"]),
        tracking=TrackingConfig(**raw["tracking"]),
    )


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entrypoint principal de treinamento."""
    _load_dotenv(Path(".env"))
    pipeline = _validate_config(cfg)

    L.seed_everything(pipeline.seed, workers=True)

    datamodule = VinDrLightningDataModule(
        cache_dir=pipeline.data.cache_dir,
        batch_size=pipeline.trainer.batch_size,
        num_workers=pipeline.data.num_workers,
        pin_memory=pipeline.data.pin_memory,
        positive_negative_ratio=pipeline.data.positive_negative_ratio,
        seed=pipeline.seed,
    )

    backbone = TimmFPNBackbone(
        backbone_name=pipeline.model.backbone_name,
        pretrained=pipeline.model.pretrained,
        out_channels=pipeline.model.fpn_out_channels,
        norm_type=pipeline.model.norm_type,
        num_groups=pipeline.model.num_groups,
        drop_path_rate=pipeline.model.drop_path_rate,
    )

    model = CascadeRCNNLightningModule(
        backbone=backbone,
        num_classes=pipeline.model.num_classes,
        cascade_iou_thresholds=pipeline.model.cascade_iou_thresholds,
        stage_loss_weights=pipeline.model.stage_loss_weights,
        learning_rate=pipeline.model.learning_rate,
        backbone_lr_scale=pipeline.model.backbone_lr_scale,
        weight_decay=pipeline.model.weight_decay,
        warmup_epochs=pipeline.model.warmup_epochs,
        image_size=pipeline.preprocessing.patch_size,
    )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or None
    mlf_logger = MLFlowLogger(
        experiment_name=pipeline.tracking.experiment_name,
        run_name=pipeline.tracking.run_name,
        tracking_uri=tracking_uri,
        log_model=pipeline.tracking.log_model,
        tags=pipeline.tracking.tags,
    )
    try:
        flat_params = _flatten_dict(pipeline.model_dump(mode="json"))
        flat_params["trainer.effective_batch_size"] = pipeline.trainer.effective_batch_size
        mlf_logger.log_hyperparams(flat_params)
    except Exception as exc:  # noqa: BLE001 - falha no tracking não deve abortar treino
        logger.warning("Falha ao registrar hiperparâmetros completos no MLflow: %s", exc)

    callbacks = [
        ModelCheckpoint(
            monitor="val/map_50",
            mode="max",
            save_top_k=2,
            save_last=True,
            auto_insert_metric_name=False,
            filename="cascade-ep{epoch:02d}",
        ),
        EarlyStopping(monitor="val/map_50", mode="max", patience=6),
        LearningRateMonitor(logging_interval="step"),
        DetectionReportCallback(),
    ]

    trainer = L.Trainer(
        accelerator=pipeline.trainer.accelerator,
        devices=pipeline.trainer.devices,
        precision=pipeline.trainer.precision,
        max_epochs=pipeline.trainer.max_epochs,
        accumulate_grad_batches=pipeline.trainer.accumulate_grad_batches,
        gradient_clip_val=pipeline.trainer.gradient_clip_val,
        log_every_n_steps=pipeline.trainer.log_every_n_steps,
        logger=mlf_logger,
        callbacks=callbacks,
    )

    logger.info(
        "Iniciando treinamento | backbone=%s | backbone_lr=%.1e | head_lr=%.1e | effective_batch=%d | precision=%s",
        pipeline.model.backbone_name,
        pipeline.model.learning_rate * pipeline.model.backbone_lr_scale,
        pipeline.model.learning_rate,
        pipeline.trainer.effective_batch_size,
        pipeline.trainer.precision,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=pipeline.trainer.ckpt_path)


if __name__ == "__main__":
    main()
