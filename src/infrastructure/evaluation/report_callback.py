"""Callback Lightning para logging de curvas, matriz, visualizações TP/FP/FN e relatório no MLflow."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import lightning as L
import matplotlib
import numpy as np
import torch
from torchvision.ops import box_iou

from src.infrastructure.datasets.vindr_datamodule import IMAGENET_MEAN, IMAGENET_STD
from src.infrastructure.evaluation.detection_metrics import (
    ConfusionMatrix,
    clinical_operating_point,
    operating_point,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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
                "Alerta: excesso de falsos positivos no fundo — avalie elevar score_thresh ou ajustar razão pos:neg."
            )
    return findings


def _denormalize_clahe_gray(image_tensor: torch.Tensor) -> np.ndarray:
    """Extrai o canal CLAHE suave (canal 1) desnormalizado para visualização radiológica limpa."""
    ch1 = image_tensor[1].detach().cpu().float().numpy()
    restored = ch1 * IMAGENET_STD[1] + IMAGENET_MEAN[1]
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def render_predictions_grid(
    output_dir: Path,
    samples: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    conf_thresh: float,
    epoch: int,
    iou_thresh: float = 0.50,
) -> Path | None:
    """Renderiza grid TP (verde) / FP (vermelho) / FN (amarelo tracejado) sobre patches de validação."""
    if not samples:
        return None

    n = len(samples)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes_list = np.atleast_1d(axes).ravel()

    for ax, (img, det, tgt) in zip(axes_list, samples, strict=False):
        gray = _denormalize_clahe_gray(img)
        ax.imshow(gray, cmap="gray")

        gt_boxes = tgt["boxes"].detach().cpu()
        scores = det["scores"].detach().cpu()
        keep = scores >= max(0.05, conf_thresh)
        pred_boxes = det["boxes"].detach().cpu()[keep]
        pred_scores = scores[keep]

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        if gt_boxes.numel() > 0 and pred_boxes.numel() > 0:
            iou = box_iou(gt_boxes, pred_boxes).numpy()
            matches = np.array(np.nonzero(iou >= iou_thresh)).T
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matched_gt = set(matches[:, 0].tolist()) if matches.size else set()
            matched_pred = set(matches[:, 1].tolist()) if matches.size else set()

        # FN (amarelo tracejado) e GT casado (azul fino)
        for idx, box in enumerate(gt_boxes.tolist()):
            x1, y1, x2, y2 = box
            if idx not in matched_gt:
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", lw=2, ls="--"
                )
                ax.add_patch(rect)

        # TP (verde) e FP (vermelho)
        for idx, (box, sc) in enumerate(zip(pred_boxes.tolist(), pred_scores.tolist(), strict=True)):
            x1, y1, x2, y2 = box
            is_tp = idx in matched_pred
            color = "lime" if is_tp else "red"
            tag = "TP" if is_tp else "FP"
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, lw=2)
            ax.add_patch(rect)
            ax.text(
                x1,
                max(10.0, y1 - 4.0),
                f"{tag} {sc:.2f}",
                color=color,
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
            )

        ax.set_title(f"GT={len(gt_boxes)} | TP={len(matched_pred)} | FP={len(pred_boxes) - len(matched_pred)}")
        ax.axis("off")

    for ax in axes_list[n:]:
        ax.axis("off")

    fig.suptitle(f"Validation TP (green) / FP (red) / FN (yellow) — Epoch {epoch} (conf>={max(0.05, conf_thresh):.2f})")
    vis_path = output_dir / f"val_predictions_epoch_{epoch:03d}.png"
    fig.tight_layout()
    fig.savefig(vis_path, dpi=150)
    plt.close(fig)
    return vis_path


def render_detection_artifacts(
    output_dir: Path,
    result: Mapping[str, Any],
    confusion: ConfusionMatrix,
    class_names: list[str],
    epoch: int,
    visual_samples: Sequence[
        tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ] = (),
) -> list[Path]:
    """Renderiza PR curve, confidence curves, matriz de confusão, visualizações e relatório .md."""
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
        ax.axvline(
            result["best_conf"], ls="--", color="gray", label=f"best_conf={result['best_conf']:.2f}"
        )
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

    # 4. TP / FP / FN Visual Predictions Grid
    vis_path = render_predictions_grid(
        output_dir=output_dir,
        samples=visual_samples,
        conf_thresh=float(result.get("best_conf", 0.25) or 0.25),
        epoch=epoch,
    )
    if vis_path is not None:
        artifacts.append(vis_path)

    # 5. Markdown Report
    findings = detection_findings(result, confusion)
    report_path = output_dir / f"val_report_epoch_{epoch:03d}.md"
    report_lines = [
        f"# Validation Report — Epoch {epoch}",
        "",
        "## Findings",
        *[f"- {f}" for f in findings],
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    artifacts.append(report_path)

    return artifacts


class DetectionReportCallback(L.Callback):
    """Renderiza e envia curvas, matriz, visualizações e relatório ao MLflow (com tolerância a falhas)."""

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Executa após LightningModule.on_validation_epoch_end para garantir last_val_result preenchido."""
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        result = getattr(pl_module, "last_val_result", None)
        confusion = getattr(pl_module, "val_confusion", None)
        if result is None or confusion is None:
            return

        mlf_logger = trainer.logger
        if mlf_logger is None or not hasattr(mlf_logger, "experiment") or not hasattr(mlf_logger, "run_id"):
            return

        class_names = getattr(pl_module.val_metrics, "class_names", ["lesion"])
        visual_samples = getattr(pl_module, "val_visual_samples", ())

        with TemporaryDirectory() as tmp_dir:
            artifacts = render_detection_artifacts(
                output_dir=Path(tmp_dir),
                result=result,
                confusion=confusion,
                class_names=class_names,
                epoch=trainer.current_epoch,
                visual_samples=visual_samples,
            )
            for path in artifacts:
                if path.suffix == ".md":
                    sub = "detection/reports"
                elif path.name.startswith("val_predictions"):
                    sub = "detection/visualizations"
                else:
                    sub = "detection/plots"
                try:
                    mlf_logger.experiment.log_artifact(
                        mlf_logger.run_id, str(path), artifact_path=sub
                    )
                except Exception as exc:  # noqa: BLE001 - tolerância a falhas de rede/MLflow
                    logger.warning("Falha ao enviar artefato %s ao MLflow: %s", path.name, exc)
