"""Testes unitários para BoundingBox e TilingEngine."""

import pytest

from src.application.tiling_engine import TilingEngine
from src.domain.bounding_box import BoundingBox
from src.domain.exceptions import InvalidBoundingBoxError


def test_bounding_box_creation_and_properties():
    box = BoundingBox(xmin=100.0, ymin=200.0, xmax=300.0, ymax=450.0)
    assert box.width == 200.0
    assert box.height == 250.0
    assert box.area == 50000.0
    assert box.center == (200.0, 325.0)
    assert box.to_coco_format() == (100.0, 200.0, 200.0, 250.0)
    assert box.to_pascal_voc() == (100.0, 200.0, 300.0, 450.0)


def test_bounding_box_negative_coords_raises_error():
    with pytest.raises(InvalidBoundingBoxError):
        BoundingBox(xmin=-1.0, ymin=10.0, xmax=50.0, ymax=60.0)


def test_bounding_box_inverted_coords_raises_error():
    with pytest.raises(InvalidBoundingBoxError):
        BoundingBox(xmin=100.0, ymin=10.0, xmax=50.0, ymax=60.0)


def test_bounding_box_translation():
    box = BoundingBox(xmin=150.0, ymin=100.0, xmax=250.0, ymax=200.0)
    translated = box.translate(offset_x=50.0, offset_y=30.0)
    assert translated.xmin == 100.0
    assert translated.ymin == 70.0
    assert translated.xmax == 200.0
    assert translated.ymax == 170.0


def test_bounding_box_iou():
    box1 = BoundingBox(xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0)
    box2 = BoundingBox(xmin=5.0, ymin=0.0, xmax=15.0, ymax=10.0)
    # Interseção: 5 x 10 = 50. União: 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
    assert pytest.approx(box1.compute_iou(box2), 0.001) == 1.0 / 3.0


def test_tiling_engine_patch_generation_and_clipping():
    engine = TilingEngine(patch_size=1000, overlap_ratio=0.20, min_bbox_coverage=0.30)

    # Imagem de 2500 x 3000
    box_inside = BoundingBox(xmin=200.0, ymin=200.0, xmax=400.0, ymax=400.0)

    patches = engine.generate_patches(
        image_width=2500,
        image_height=3000,
        target_bboxes=[box_inside],
    )

    assert len(patches) > 0
    # Patch 0 (canto superior esquerdo 0,0 a 1000,1000) deve conter a caixa inteira
    first_patch = patches[0]
    assert first_patch.xmin == 0.0
    assert first_patch.ymin == 0.0
    assert len(first_patch.local_bboxes) == 1
    assert first_patch.local_bboxes[0].xmin == 200.0
    assert first_patch.local_bboxes[0].ymin == 200.0
    assert first_patch.local_bboxes[0].width == 200.0
    assert first_patch.local_bboxes[0].height == 200.0
