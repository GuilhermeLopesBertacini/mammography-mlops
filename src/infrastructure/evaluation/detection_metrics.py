"""Métricas de detecção inspiradas no esquema AirisML (PR, F1, mAP, pontos de operação e ConfusionMatrix)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torchvision.ops import box_iou

IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)
_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz


def match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor = IOU_THRESHOLDS,
) -> np.ndarray:
    """Matriz de acertos [N_preds, len(iouv)] com matching guloso 1-para-1 por maior IoU."""
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return correct

    same_class = true_classes[:, None] == pred_classes[None, :]
    iou_masked = (iou * same_class).cpu().numpy()

    for i, threshold in enumerate(iouv.tolist()):
        matches = np.array(np.nonzero(iou_masked >= threshold)).T  # [K, (target, pred)]
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_masked[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def _smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    pad = np.ones(nf // 2)
    padded = np.concatenate((pad * y[0], y, pad * y[-1]), 0)
    return np.convolve(padded, np.ones(nf) / nf, mode="valid")


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """AP interpolada em 101 pontos (padrão COCO)."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_TRAPEZOID(np.interp(x, mrec, mpre), x)), mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16,
) -> dict[str, Any]:
    """Calcula AP por classe/IoU e curvas P/R/F1 ao longo da confiança."""
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]

    classes, n_targets = np.unique(target_cls, return_counts=True)
    nc = classes.shape[0]
    x = np.linspace(0, 1, 1000)
    prec_values: list[np.ndarray] = []
    ap = np.zeros((nc, tp.shape[1]))
    p_curve, r_curve = np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(classes):
        selected = pred_cls == c
        n_labels, n_preds = int(n_targets[ci]), int(selected.sum())
        if n_preds == 0 or n_labels == 0:
            continue
        fpc = (1 - tp[selected]).cumsum(0)
        tpc = tp[selected].cumsum(0)
        recall = tpc / (n_labels + eps)
        r_curve[ci] = np.interp(-x, -conf[selected], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[selected], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))

    prec_arr = np.array(prec_values) if prec_values else np.zeros((1, 1000))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    best = int(_smooth(f1_curve.mean(0), 0.1).argmax()) if nc else 0
    p, r, f1 = p_curve[:, best], r_curve[:, best], f1_curve[:, best]
    tp_count = (r * n_targets).round()
    fp_count = np.maximum(0.0, (tp_count / (p + eps) - tp_count).round())

    return {
        "tp": tp_count,
        "fp": fp_count,
        "p": p,
        "r": r,
        "f1": f1,
        "ap": ap,
        "classes": classes.astype(int),
        "p_curve": p_curve,
        "r_curve": r_curve,
        "f1_curve": f1_curve,
        "x": x,
        "prec_values": prec_arr,
        "best_index": best,
        "best_conf": float(x[best]),
    }


def operating_point(result: Mapping[str, Any], conf: float) -> dict[str, float]:
    """Lê Precision, Recall e F1 médios para um limiar de confiança específico."""
    curves = result.get("curves")
    if not curves:
        return {"conf": float(conf), "precision": 0.0, "recall": 0.0, "f1": 0.0}

    x = np.asarray(curves["x"])
    idx = int(np.argmin(np.abs(x - conf)))
    return {
        "conf": float(x[idx]),
        "precision": float(np.mean(curves["p_curve"][:, idx])),
        "recall": float(np.mean(curves["r_curve"][:, idx])),
        "f1": float(np.mean(curves["f1_curve"][:, idx])),
    }


def clinical_operating_point(
    result: Mapping[str, Any],
    min_recall: float = 0.90,
) -> dict[str, float]:
    """Encontra o maior threshold de confiança que ainda atinge Recall >= min_recall."""
    curves = result.get("curves")
    if not curves:
        return {"conf": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    mean_r = np.mean(curves["r_curve"], axis=0)
    eligible = np.nonzero(mean_r >= min_recall)[0]
    if eligible.size == 0:
        # Se nenhuma confiança atingir min_recall, retorna o ponto de maior recall disponível
        idx = int(np.argmax(mean_r))
    else:
        idx = int(eligible[-1])  # x é crescente -> maior threshold elegível

    x = np.asarray(curves["x"])
    return {
        "conf": float(x[idx]),
        "precision": float(np.mean(curves["p_curve"][:, idx])),
        "recall": float(mean_r[idx]),
        "f1": float(np.mean(curves["f1_curve"][:, idx])),
    }


class DetectionMetrics:
    """Acumula predições por imagem e computa mAP, P/R/F1 e curvas de operação."""

    def __init__(self, class_names: Sequence[str] = ("lesion",), label_offset: int = 1) -> None:
        self.class_names = list(class_names)
        self.label_offset = label_offset
        self.reset()

    def reset(self) -> None:
        self._tp: list[torch.Tensor] = []
        self._conf: list[torch.Tensor] = []
        self._pred_cls: list[torch.Tensor] = []
        self._target_cls: list[torch.Tensor] = []
        self._num_images = 0

    def update(
        self,
        detections: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        for det, target in zip(detections, targets, strict=True):
            self._num_images += 1
            pred_boxes = det["boxes"].detach().cpu()
            pred_scores = det["scores"].detach().cpu()
            pred_labels = det["labels"].detach().cpu() - self.label_offset

            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu() - self.label_offset

            self._target_cls.append(gt_labels)
            if pred_boxes.numel() == 0:
                continue

            if gt_boxes.numel() == 0:
                correct = torch.zeros((pred_boxes.shape[0], len(IOU_THRESHOLDS)), dtype=torch.bool)
            else:
                iou = box_iou(gt_boxes, pred_boxes)
                correct = torch.from_numpy(match_predictions(pred_labels, gt_labels, iou))

            self._tp.append(correct)
            self._conf.append(pred_scores)
            self._pred_cls.append(pred_labels)

    def compute(self) -> dict[str, Any]:
        target_cls = (
            torch.cat(self._target_cls).numpy() if self._target_cls else np.empty(0, dtype=int)
        )
        num_instances = int(target_cls.shape[0])
        num_predictions = int(sum(t.shape[0] for t in self._conf))

        base: dict[str, Any] = {
            "mAP50": 0.0,
            "mAP75": 0.0,
            "mAP50-95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "best_conf": 0.0,
            "fp_per_image": 0.0,
            "num_images": self._num_images,
            "num_instances": num_instances,
            "num_predictions": num_predictions,
            "curves": None,
        }
        if num_instances == 0 or num_predictions == 0:
            return base

        tp = torch.cat(self._tp).numpy()
        conf = torch.cat(self._conf).numpy()
        pred_cls = torch.cat(self._pred_cls).numpy()

        curves = ap_per_class(tp, conf, pred_cls, target_cls)
        ap = curves["ap"]
        total_fp = float(np.sum(curves["fp"]))
        base.update(
            {
                "mAP50": float(ap[:, 0].mean()),
                "mAP75": float(ap[:, 5].mean()),
                "mAP50-95": float(ap.mean()),
                "precision": float(curves["p"].mean()),
                "recall": float(curves["r"].mean()),
                "f1": float(curves["f1"].mean()),
                "best_conf": float(curves["best_conf"]),
                "fp_per_image": total_fp / max(1, self._num_images),
                "curves": curves,
            }
        )
        return base


class ConfusionMatrix:
    """Matriz de confusão [nc + 1, nc + 1] separando background_fp e missed (FN)."""

    def __init__(
        self,
        num_classes: int = 1,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.50,
        label_offset: int = 1,
    ) -> None:
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.label_offset = label_offset
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    def reset(self) -> None:
        self.matrix.fill(0)

    @property
    def background_fp(self) -> int:
        """Total de falsos positivos previstos sobre o fundo (sem objeto correspondente)."""
        return int(self.matrix[: self.num_classes, self.num_classes].sum())

    @property
    def missed(self) -> int:
        """Total de lesões reais perdidas (falsos negativos)."""
        return int(self.matrix[self.num_classes, : self.num_classes].sum())

    def update(
        self,
        detections: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        bg = self.num_classes
        for det, target in zip(detections, targets, strict=True):
            scores = det["scores"].detach().cpu()
            keep = scores >= self.conf_thresh
            pred_boxes = det["boxes"].detach().cpu()[keep]
            pred_labels = (det["labels"].detach().cpu()[keep] - self.label_offset).long()

            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = (target["labels"].detach().cpu() - self.label_offset).long()

            if gt_boxes.numel() == 0:
                for p_cls in pred_labels.tolist():
                    if 0 <= p_cls < self.num_classes:
                        self.matrix[p_cls, bg] += 1
                continue

            if pred_boxes.numel() == 0:
                for g_cls in gt_labels.tolist():
                    if 0 <= g_cls < self.num_classes:
                        self.matrix[bg, g_cls] += 1
                continue

            iou = box_iou(gt_boxes, pred_boxes).numpy()
            matches = np.array(np.nonzero(iou >= self.iou_thresh)).T
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            matched_gt = set(matches[:, 0].tolist()) if matches.size else set()
            matched_pred = set(matches[:, 1].tolist()) if matches.size else set()

            for g_idx, p_idx in matches.tolist():
                g_cls = int(gt_labels[g_idx])
                p_cls = int(pred_labels[p_idx])
                if 0 <= g_cls < self.num_classes and 0 <= p_cls < self.num_classes:
                    self.matrix[p_cls, g_cls] += 1

            for g_idx, g_cls in enumerate(gt_labels.tolist()):
                if g_idx not in matched_gt and 0 <= g_cls < self.num_classes:
                    self.matrix[bg, g_cls] += 1

            for p_idx, p_cls in enumerate(pred_labels.tolist()):
                if p_idx not in matched_pred and 0 <= p_cls < self.num_classes:
                    self.matrix[p_cls, bg] += 1
