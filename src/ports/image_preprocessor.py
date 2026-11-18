"""Porta abstrata para pré-processamento de imagens mamográficas."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.domain.enums import PhotometricInterpretation


@dataclass(frozen=True)
class PreprocessResult:
    """Resultado do pré-processamento contendo imagem 3-canais e coordenadas do crop original."""

    image_rgb: npt.NDArray[np.uint8]
    crop_xmin: int
    crop_ymin: int
    crop_xmax: int
    crop_ymax: int


class ImagePreprocessorPort(ABC):
    """Contrato para leitura, correção fotométrica, crop de fundo e janelamento."""

    @abstractmethod
    def preprocess_dicom(
        self,
        dicom_path: Path,
        photometric: PhotometricInterpretation,
    ) -> PreprocessResult:
        """Lê um arquivo DICOM do disco e retorna a imagem pré-processada."""

    @abstractmethod
    def preprocess_array(
        self,
        raw_array: npt.NDArray[np.floating],
        photometric: PhotometricInterpretation,
    ) -> PreprocessResult:
        """Pré-processa uma matriz 2D já carregada em memória."""
