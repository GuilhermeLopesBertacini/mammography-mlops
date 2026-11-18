"""Adapter OpenCV + pydicom para pré-processamento de mamografias."""

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pydicom

try:
    from pydicom.pixels import apply_voi_lut
except ImportError:  # pragma: no cover - compatibilidade com pydicom < 3.0
    from pydicom.pixel_data_handlers.util import apply_voi_lut  # type: ignore[no-redef]

from src.domain.enums import PhotometricInterpretation
from src.ports.image_preprocessor import ImagePreprocessorPort, PreprocessResult


class OpenCVMammogramPreprocessor(ImagePreprocessorPort):
    """Executa correção MONOCHROME1, crop do tecido mamário, padding e 3 canais CLAHE."""

    def __init__(
        self,
        min_size: int = 1024,
        crop_black_background: bool = True,
        crop_padding: int = 32,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ) -> None:
        if min_size <= 0:
            raise ValueError(f"min_size deve ser positivo: {min_size}")
        if crop_padding < 0:
            raise ValueError(f"crop_padding não pode ser negativo: {crop_padding}")

        self.min_size = min_size
        self.crop_black_background = crop_black_background
        self.crop_padding = crop_padding
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

    def preprocess_dicom(
        self,
        dicom_path: Path,
        photometric: PhotometricInterpretation,
    ) -> PreprocessResult:
        """Lê o DICOM com VOI LUT e delega para o processamento em memória."""
        if not dicom_path.is_file():
            raise FileNotFoundError(f"Arquivo DICOM não encontrado: {dicom_path}")

        ds = pydicom.dcmread(dicom_path)
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
        return self.preprocess_array(arr, photometric)

    def preprocess_array(
        self,
        raw_array: npt.NDArray[np.floating],
        photometric: PhotometricInterpretation,
    ) -> PreprocessResult:
        """Normaliza fotometria, recorta fundo preto, aplica padding e gera 3 canais."""
        img = raw_array.astype(np.float32, copy=False)

        if photometric == PhotometricInterpretation.MONOCHROME1:
            img = float(img.max()) - img

        img_uint8 = self._normalize_to_uint8(img)
        orig_h, orig_w = img_uint8.shape

        if self.crop_black_background:
            xmin, ymin, xmax, ymax = self._detect_breast_roi(img_uint8, orig_w, orig_h)
        else:
            xmin, ymin, xmax, ymax = 0, 0, orig_w, orig_h

        cropped = img_uint8[ymin:ymax, xmin:xmax]
        padded = self._pad_bottom_right(cropped)
        image_rgb = self._build_three_channels(padded)

        return PreprocessResult(
            image_rgb=image_rgb,
            crop_xmin=xmin,
            crop_ymin=ymin,
            crop_xmax=xmax,
            crop_ymax=ymax,
        )

    def _normalize_to_uint8(self, img: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """Normaliza intensidades para [0, 255] com corte leve de outliers."""
        low = float(np.percentile(img, 0.1))
        high = float(np.percentile(img, 99.9))
        if high <= low:
            return np.zeros(img.shape, dtype=np.uint8)

        clipped = np.clip(img, low, high)
        scaled = (clipped - low) / (high - low) * 255.0
        return scaled.astype(np.uint8)

    def _detect_breast_roi(
        self,
        img_uint8: npt.NDArray[np.uint8],
        orig_w: int,
        orig_h: int,
    ) -> tuple[int, int, int, int]:
        """Localiza a maior componente conexa (mama), ignorando etiquetas soltas."""
        _, binary = cv2.threshold(
            img_uint8, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU # Calculates threshold for histograms with clear and distinct peaks
        )
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Label 0 é sempre o fundo; se não houver componente frontal, mantém imagem inteira
        if num_labels <= 1:
            return 0, 0, orig_w, orig_h

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1

        x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
        y = int(stats[largest_idx, cv2.CC_STAT_TOP])
        w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])

        xmin = max(0, x - self.crop_padding)
        ymin = max(0, y - self.crop_padding)
        xmax = min(orig_w, x + w + self.crop_padding)
        ymax = min(orig_h, y + h + self.crop_padding)
        return xmin, ymin, xmax, ymax

    def _pad_bottom_right(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Garante dimensões >= min_size sem deslocar a origem (0, 0) do crop."""
        h, w = img.shape
        pad_bottom = max(0, self.min_size - h)
        pad_right = max(0, self.min_size - w)
        if pad_bottom == 0 and pad_right == 0:
            return img

        bordered = cv2.copyMakeBorder(
            img,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        return bordered.astype(np.uint8, copy=False)

    def _build_three_channels(self, img_uint8: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Monta 3 canais complementares: [linear, CLAHE suave, CLAHE forte]."""
        grid = (self.clahe_tile_grid_size, self.clahe_tile_grid_size)
        clahe_soft = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=grid)
        clahe_strong = cv2.createCLAHE(clipLimit=self.clahe_clip_limit * 2.0, tileGridSize=grid)

        ch0 = img_uint8
        ch1 = clahe_soft.apply(img_uint8)
        ch2 = clahe_strong.apply(img_uint8)
        return np.stack([ch0, ch1, ch2], axis=-1)
