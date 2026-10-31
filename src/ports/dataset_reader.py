"""Contratos abstratos (Portas) para ingestão de datasets e rastreamento de experimentos."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.domain.entities import PatientStudy


class DatasetReaderPort(ABC):
    """Porta para ingestão e desserialização de datasets radiológicos."""

    @abstractmethod
    def load_studies(self, data_dir: Path) -> list[PatientStudy]:
        """Carrega todos os estudos do dataset preservando o agregado de paciente."""


class ExperimentTrackerPort(ABC):
    """Porta para tracking e telemetria de experimentos (MLflow, etc.)."""

    @abstractmethod
    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> None:
        """Inicia uma execução rastreada."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Registra hiperparâmetros e configurações."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Registra métricas escalares em um determinado passo/época."""

    @abstractmethod
    def log_artifact(self, local_path: Path, artifact_path: str | None = None) -> None:
        """Envia um arquivo de artefato para o storage remoto."""

    @abstractmethod
    def end_run(self) -> None:
        """Encerra a execução ativa."""
