"""Testes unitários para DetectionMetrics, ConfusionMatrix e renderização de relatórios."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.infrastructure.evaluation.detection_metrics import (
    ConfusionMatrix,
    DetectionMetrics,
    clinical_operating_point,
    match_predictions,
    operating_point,
)
from src.infrastructure.evaluation.report_callback import (
    detection_findings,
    render_detection_artifacts,
)


def test_match_predictions_is_one_to_one_and_class_aware() -> None:
    iou = torch.tensor([[0.9, 0.8, 0.0], [0.0, 0.0, 0.72]])
    correct = match_predictions(torch.tensor([0, 0, 1]), torch.tensor([0, 0]), iou)
    assert correct[:, 0].tolist() == [True, False, False]
    assert correct.shape == (3, 10)
    assert int(correct[0].sum()) == 9  # IoU 0.9 passa 0.50..0.90


def test_perfect_predictions_give_unit_metrics() -> None:
    box = torch.tensor([[10.0, 10.0, 50.0, 60.0]])
    metrics = DetectionMetrics(class_names=["lesion"], label_offset=1)
    metrics.update(
        [{"boxes": box, "scores": torch.tensor([0.95]), "labels": torch.tensor([1])}],
        [{"boxes": box, "labels": torch.tensor([1])}],
    )
    result = metrics.compute()
    assert result["mAP50"] == pytest.approx(0.995, abs=1e-3)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_confusion_matrix_tracks_background_fp_and_missed_lesions() -> None:
    cm = ConfusionMatrix(num_classes=1, conf_thresh=0.25, iou_thresh=0.50, label_offset=1)
    cm.update(
        [
            {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)},
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])},
        ],
        [
            {"boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)},
        ],
    )
    assert cm.missed == 1
    assert cm.background_fp == 1


def test_operating_points_and_artifact_rendering(tmp_path: Path) -> None:
    box = torch.tensor([[10.0, 10.0, 50.0, 60.0]])
    metrics = DetectionMetrics(class_names=["lesion"], label_offset=1)
    cm = ConfusionMatrix(num_classes=1, conf_thresh=0.25, iou_thresh=0.50, label_offset=1)

    dets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 60.0], [100.0, 100.0, 130.0, 130.0]]),
            "scores": torch.tensor([0.92, 0.40]),
            "labels": torch.tensor([1, 1]),
        }
    ]
    targs = [{"boxes": box, "labels": torch.tensor([1])}]
    metrics.update(dets, targs)
    cm.update(dets, targs)

    result = metrics.compute()
    op = operating_point(result, result["best_conf"])
    clin = clinical_operating_point(result, min_recall=0.90)
    assert op["f1"] >= 0.0
    assert clin["recall"] >= 0.90

    findings = detection_findings(result, cm)
    assert any("Ponto clínico" in line for line in findings)

    artifacts = render_detection_artifacts(tmp_path, result, cm, ["lesion"], epoch=0)
    assert len(artifacts) == 4
    assert all(p.is_file() for p in artifacts)
