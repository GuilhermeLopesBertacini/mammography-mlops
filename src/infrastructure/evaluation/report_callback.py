"""Callback Lightning para logging de curvas, matriz de confusão e relatório no MLflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import lightning as L
import matplotlib
import numpy as np

from src.infrastructure.evaluation.detection_metrics import (
    ConfusionMatrix,
    clinical_operating_point,
    operating_point,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detection_findings(
    result: Mapping[str, Any],
    confusion: ConfusionMatrix | None = None,
    min_clinical_recall: float = 0.90,
) -> list[str]:
    """Gera conclusões diretas a partir do resultado de DetectionMetrics.compute()."""
    if not result or not result.get("num_instances"):
        return ["Nenhum objeto ground-truth neste split: métricas não podem ser calculadas."]

    best = operating_point(result, float(result["best_conf"]))
    clinical = clinical_operating_point(result, min_recall=min_clinical_recall)

    findings = [
        (
            f"Melhor F1={best['f1']:.3f} em conf={best['conf']:.2f} "
            f"(Precision={best['precision']:.1%}, Recall={best['recall']:.1%}, "
            f"mAP@50={result['mAP50']:.3f}, mAP@50-95={result['mAP50-95']:.3f})."
        ),
        (
            f"Ponto clínico (Recall >= {min_clinical_recall:.0%}): "
            f"conf={clinical['conf']:.2f} -> Recall={clinical['recall']:.1%}, "
            f"Precision={clinical['precision']:.1%}, F1={clinical['f1']:.3f}."
        ),
    ]

    if confusion is not None:
        bg_fp = confusion.background_fp
        missed = confusion.missed
        findings.append(
            f"Matriz (conf>={confusion.conf_thresh:.2f}, IoU>={confusion.iou_thresh:.2f}): "
            f"background_fp={bg_fp}, missed={missed}."
        )
        if bg_fp > max(10, 2 * missed):
            findings.append(
                "Alerta: excesso de falsos positivos no fundo — avalie elevar score_thresh ou aumentar razão de patches negativos."
            )
    return findings


def render_detection_artifacts(
    output_dir: Path,
    result: Mapping[str, Any],
    confusion: ConfusionMatrix,
    class_names: list[str],
    epoch: int,
) -> list[Path]:
    """Renderiza PR curve, confidence curves, matriz de confusão e relatório .md em disco."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    curves = result.get("curves")

    if curves is not None:
        # 1. PR Curve
        fig, ax = plt.subplots(figsize=(6, 5))
        mean_prec = np.mean(curves["prec_values"], axis=0)
        x_rec = np.linspace(0, 1, len(mean_prec))
        ax.plot(x_rec, mean_prec, lw=2, label=f"mAP@50={result['mAP50']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve (Epoch {epoch})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
        pr_path = output_dir / f"pr_curve_epoch_{epoch:03d}.png"
        fig.tight_layout()
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)
        artifacts.append(pr_path)

        # 2. Confidence Curves (P / R / F1 vs Confidence)
        fig, ax = plt.subplots(figsize=(6, 5))
        x_conf = np.asarray(curves["x"])
        ax.plot(x_conf, np.mean(curves["p_curve"], axis=0), label="Precision")
        ax.plot(x_conf, np.mean(curves["r_curve"], axis=0), label="Recall")
        ax.plot(x_conf, np.mean(curves["f1_curve"], axis=0), lw=2, label="F1")
        ax.axvline(result["best_conf"], ls="--", color="gray", label=f"best_conf={result['best_conf']:.2f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Score")
        ax.set_title(f"Confidence Curves (Epoch {epoch})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        conf_path = output_dir / f"confidence_curves_epoch_{epoch:03d}.png"
        fig.tight_layout()
        fig.savefig(conf_path, dpi=150)
        plt.close(fig)
        artifacts.append(conf_path)

    # 3. Confusion Matrix
    labels = [*class_names, "background"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(confusion.matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Confusion Matrix (Epoch {epoch})")
    for i in range(confusion.matrix.shape[0]):
        for j in range(confusion.matrix.shape[1]):
            ax.text(j, i, str(confusion.matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    cm_path = output_dir / f"confusion_matrix_epoch_{epoch:03d}.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    artifacts.append(cm_path)

    # 4. Markdown Report
    findings = detection_findings(result, confusion)
    report_path = output_dir / f"val_report_epoch_{epoch:03d}.md"
    report_lines = [f"# Validation Report — Epoch {epoch}", "", "## Findings", *[f"- {f}" for f in findings]]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    artifacts.append(report_path)

    return artifacts


class DetectionReportCallback(L.Callback):
    """Renderiza e envia curvas, matriz de confusão e relatório para o MLflow."""

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        result = getattr(pl_module, "last_val_result", None)
        confusion = getattr(pl_module, "val_confusion", None)
        if result is None or confusion is None:
            return

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment") or not hasattr(logger, "run_id"):
            return

        class_names = getattr(pl_module.val_metrics, "class_names", ["lesion"])
        with TemporaryDirectory() as tmp_dir:
            artifacts = render_detection_artifacts(
                output_dir=Path(tmp_dir),
                result=result,
                confusion=confusion,
                class_names=class_names,
                epoch=trainer.current_epoch,
            )
            for path in artifacts:
                sub = "detection/reports" if path.suffix == ".md" else "detection/plots"
                logger.experiment.log_artifact(logger.run_id, str(path), artifact_path=sub)
