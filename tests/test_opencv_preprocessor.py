"""Testes unitários do OpenCVMammogramPreprocessor com imagens sintéticas."""

import numpy as np

from src.domain.bounding_box import BoundingBox
from src.domain.enums import PhotometricInterpretation
from src.infrastructure.preprocessing import OpenCVMammogramPreprocessor


def test_preprocessor_crops_black_background_and_ignores_small_marker() -> None:
    """Deve recortar ao redor da mama e ignorar etiqueta solta (ex: RCC) no fundo."""
    img = np.zeros((1600, 1400), dtype=np.float32)
    # Mama sintética à esquerda
    img[200:1300, 50:750] = 2000.0
    # Etiqueta pequena distante no canto superior direito
    img[50:90, 1250:1320] = 3000.0

    preprocessor = OpenCVMammogramPreprocessor(
        min_size=1024,
        crop_black_background=True,
        crop_padding=32,
    )
    result = preprocessor.preprocess_array(img, PhotometricInterpretation.MONOCHROME2)

    assert result.crop_xmin == 50 - 32
    assert result.crop_ymin == 200 - 32
    assert result.crop_xmax == 750 + 32
    assert result.crop_ymax == 1300 + 32
    assert result.image_rgb.shape[2] == 3
    assert result.image_rgb.dtype == np.uint8
    assert result.image_rgb.shape[0] >= 1024
    assert result.image_rgb.shape[1] >= 1024


def test_preprocessor_pads_bottom_right_when_crop_smaller_than_min_size() -> None:
    """Se a mama recortada for menor que 1024px, deve preencher apenas à direita/baixo."""
    img = np.zeros((1200, 1200), dtype=np.float32)
    img[300:700, 300:700] = 1500.0  # 400x400

    preprocessor = OpenCVMammogramPreprocessor(
        min_size=1024,
        crop_black_background=True,
        crop_padding=32,
    )
    result = preprocessor.preprocess_array(img, PhotometricInterpretation.MONOCHROME2)

    assert result.image_rgb.shape == (1024, 1024, 3)

    # BoundingBox dentro da mama deve ser transladada corretamente via clip_to_patch
    lesion = BoundingBox(xmin=400.0, ymin=400.0, xmax=500.0, ymax=500.0)
    cropped_lesion = lesion.clip_to_patch(
        result.crop_xmin, result.crop_ymin, result.crop_xmax, result.crop_ymax
    )
    assert cropped_lesion is not None
    assert cropped_lesion.xmin == 400.0 - result.crop_xmin
    assert cropped_lesion.ymin == 400.0 - result.crop_ymin


def test_preprocessor_inverts_monochrome1_photometry() -> None:
    """MONOCHROME1 tem fundo branco (alto) e mama escura (baixa); deve inverter."""
    img = np.full((1200, 1200), 4095.0, dtype=np.float32)
    img[100:1100, 100:900] = 500.0  # Mama mais escura que o fundo

    preprocessor = OpenCVMammogramPreprocessor(min_size=1024, crop_black_background=False)
    result = preprocessor.preprocess_array(img, PhotometricInterpretation.MONOCHROME1)

    # Após inversão, o fundo deve ser ~0 e a mama alta (>200) no canal linear (ch0)
    assert result.image_rgb[10, 10, 0] == 0
    assert result.image_rgb[500, 500, 0] > 200
