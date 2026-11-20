"""Testes unitários e smoke-test fim-a-fim do CascadeRCNNLightningModule."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")
pytest.importorskip("timm")
pytest.importorskip("torchmetrics")

from src.infrastructure.datasets.vindr_datamodule import VinDrLightningDataModule
from src.infrastructure.evaluation import DetectionReportCallback
from src.infrastructure.models import CascadeRCNNLightningModule, TimmFPNBackbone
from tests.test_vindr_datamodule import _create_synthetic_cache


def _build_tiny_module() -> CascadeRCNNLightningModule:
    backbone = TimmFPNBackbone(
        backbone_name="convnext_tiny",
        pretrained=False,
        out_channels=64,
        norm_type="group_norm",
        num_groups=8,
    )
    return CascadeRCNNLightningModule(
        backbone=backbone,
        num_classes=2,
        cascade_iou_thresholds=(0.5, 0.6, 0.7),
        learning_rate=1e-4,
        image_size=128,
    )


def test_cascade_rcnn_contains_no_batchnorm() -> None:
    model = _build_tiny_module()
    bn_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)
    assert not any(isinstance(m, bn_types) for m in model.modules())


def test_cascade_rcnn_train_and_eval_forward_contracts() -> None:
    model = _build_tiny_module()
    images = [torch.randn(3, 128, 128), torch.randn(3, 128, 128)]
    targets = [
        {"boxes": torch.tensor([[10.0, 20.0, 60.0, 80.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)},
    ]

    # Modo treino: retorna losses da RPN e dos 3 estágios Cascade
    model.train()
    detections, losses = model.detector(images, targets)
    assert detections == []
    assert "loss_objectness" in losses
    assert "stage_0_loss_classifier" in losses
    assert "stage_1_loss_classifier" in losses
    assert "stage_2_loss_classifier" in losses

    total_loss = sum(losses.values())
    assert isinstance(total_loss, torch.Tensor)
    total_loss.backward()

    # Modo eval: retorna predições com boxes, scores e labels
    model.eval()
    with torch.no_grad():
        eval_detections, eval_losses = model.detector(images)
    assert eval_losses == {}
    assert len(eval_detections) == 2
    for det in eval_detections:
        assert set(det.keys()) == {"boxes", "scores", "labels"}
        assert det["boxes"].ndim == 2 and det["boxes"].shape[1] == 4


def test_differential_lr_param_groups() -> None:
    model = _build_tiny_module()
    groups = model._build_param_groups()
    lrs = {round(g["lr"], 8) for g in groups}
    wds = {round(g["weight_decay"], 8) for g in groups}
    # Backbone tem LR reduzido (0.2x = 2e-5) e heads mantêm 1e-4; norms/biases têm wd=0.0
    assert lrs == {2e-5, 1e-4}
    assert 0.0 in wds and 0.05 in wds


def test_lightning_trainer_fast_dev_run_and_report_callback_logs_artifacts(tmp_path: Path) -> None:
    _create_synthetic_cache(tmp_path)
    dm = VinDrLightningDataModule(
        cache_dir=tmp_path,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        positive_negative_ratio=1.0,
        seed=42,
    )
    model = _build_tiny_module()

    logged: list[tuple[str, str]] = []

    class _FakeExperiment:
        def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
            logged.append((Path(local_path).name, artifact_path or ""))

    class _FakeLogger(L.pytorch.loggers.Logger):
        @property
        def name(self) -> str:
            return "fake"

        @property
        def version(self) -> str:
            return "0"

        @property
        def experiment(self) -> _FakeExperiment:
            return _FakeExperiment()

        @property
        def run_id(self) -> str:
            return "run-123"

        def log_hyperparams(self, params) -> None:
            pass

        def log_metrics(self, metrics, step=None) -> None:
            pass

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=_FakeLogger(),
        callbacks=[DetectionReportCallback()],
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)

    artifact_names = {name for name, _ in logged}
    assert any(n.startswith("confusion_matrix_") for n in artifact_names)
    assert any(n.startswith("val_report_") for n in artifact_names)
    assert any(n.startswith("val_predictions_") for n in artifact_names)
