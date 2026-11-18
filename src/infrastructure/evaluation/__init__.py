"""Módulo de métricas de detecção, matriz de confusão e relatórios para MLflow."""

from src.infrastructure.evaluation.detection_metrics import (
    ConfusionMatrix,
    DetectionMetrics,
    ap_per_class,
    clinical_operating_point,
    match_predictions,
    operating_point,
)
from src.infrastructure.evaluation.report_callback import (
    DetectionReportCallback,
    detection_findings,
)

__all__ = [
    "ConfusionMatrix",
    "DetectionMetrics",
    "DetectionReportCallback",
    "ap_per_class",
    "clinical_operating_point",
    "detection_findings",
    "match_predictions",
    "operating_point",
]
