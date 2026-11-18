"""Testes unitários do VinDrPatchDataset, collate_fn e VinDrLightningDataModule."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("albumentations")

from src.infrastructure.datasets.vindr_datamodule import (
    VinDrLightningDataModule,
    detection_collate_fn,
)


def _create_synthetic_cache(cache_dir: Path) -> None:
    patches_dir = cache_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        ("p_pos_1.png", "s1", "i1", "train", True, [[10.0, 20.0, 60.0, 80.0, 1]]),
        ("p_pos_2.png", "s2", "i2", "train", True, [[15.0, 25.0, 75.0, 85.0, 1], [5.0, 5.0, 25.0, 25.0, 1]]),
        ("p_neg_1.png", "s3", "i3", "train", False, []),
        ("p_neg_2.png", "s4", "i4", "train", False, []),
        ("p_neg_3.png", "s5", "i5", "train", False, []),
        ("p_neg_4.png", "s6", "i6", "train", False, []),
        ("p_val_1.png", "s7", "i7", "val", True, [[12.0, 12.0, 50.0, 50.0, 1]]),
        ("p_test_1.png", "s8", "i8", "test", False, []),
    ]

    with (cache_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["patch_path", "study_id", "image_id", "split", "has_lesion", "bboxes_json"]
        )
        writer.writeheader()
        for fname, study_id, img_id, split, has_lesion, boxes in rows:
            rel = Path("patches") / fname
            arr = np.full((128, 128, 3), 128, dtype=np.uint8)
            Image.fromarray(arr).save(cache_dir / rel)
            writer.writerow(
                {
                    "patch_path": str(rel),
                    "study_id": study_id,
                    "image_id": img_id,
                    "split": split,
                    "has_lesion": has_lesion,
                    "bboxes_json": json.dumps(boxes),
                }
            )


def test_datamodule_handles_variable_boxes_empty_targets_and_weighted_sampling(
    tmp_path: Path,
) -> None:
    _create_synthetic_cache(tmp_path)

    dm = VinDrLightningDataModule(
        cache_dir=tmp_path,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        positive_negative_ratio=1.0,
        seed=42,
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    # 2 positivos com ratio 1.0 => num_samples = 4 (2 batches de tamanho 2)
    batches = list(train_loader)
    assert len(batches) == 2

    images, targets = batches[0]
    assert isinstance(images, list) and len(images) == 2
    assert images[0].shape == (3, 128, 128)
    assert images[0].dtype == torch.float32

    for target in targets:
        assert target["boxes"].ndim == 2
        assert target["boxes"].shape[1] == 4
        assert target["boxes"].dtype == torch.float32
        assert target["labels"].ndim == 1
        assert target["labels"].dtype == torch.int64

    # Validação e teste devem retornar 100% do split sem sampler
    val_images, val_targets = next(iter(dm.val_dataloader()))
    assert len(val_images) == 1
    assert val_targets[0]["boxes"].shape == (1, 4)

    test_images, test_targets = next(iter(dm.test_dataloader()))
    assert len(test_images) == 1
    assert test_targets[0]["boxes"].shape == (0, 4)
    assert test_targets[0]["labels"].shape == (0,)


def test_detection_collate_fn_preserves_list_structure() -> None:
    sample_a = (torch.zeros(3, 64, 64), {"boxes": torch.zeros((2, 4)), "labels": torch.ones(2, dtype=torch.int64)})
    sample_b = (torch.zeros(3, 64, 64), {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)})

    images, targets = detection_collate_fn([sample_a, sample_b])
    assert len(images) == 2
    assert targets[0]["boxes"].shape == (2, 4)
    assert targets[1]["boxes"].shape == (0, 4)
