"""Dataset PyTorch, augmentations Albumentations e LightningDataModule para patches VinDr."""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import lightning as L
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms() -> A.Compose:
    """Pipeline conservador e regularizado de augmentation para mamografia."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                rotate=(-15.0, 15.0),
                p=0.6,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 48),
                hole_width_range=(16, 48),
                fill=0,
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=16.0,
            min_visibility=0.3,
        ),
    )


def build_eval_transforms() -> A.Compose:
    """Pipeline determinístico para validação e teste."""
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),
    )


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Agrupa batches com número variável de bounding boxes por imagem."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class VinDrPatchDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Dataset PyTorch que lê patches pré-processados do cache em disco."""

    def __init__(
        self,
        records: pd.DataFrame,
        cache_dir: Path,
        transforms: A.Compose,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.records.iloc[idx]
        img_path = self.cache_dir / str(row["patch_path"])
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Patch não encontrado ou ilegível: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        raw_boxes = json.loads(str(row["bboxes_json"]))
        bboxes = [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in raw_boxes]
        labels = [int(b[4]) for b in raw_boxes]

        transformed = self.transforms(image=img_rgb, bboxes=bboxes, labels=labels)
        image_tensor: torch.Tensor = transformed["image"]
        out_boxes = transformed["bboxes"]
        out_labels = transformed["labels"]

        if len(out_boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(out_boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(out_labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }
        return image_tensor, target


class VinDrLightningDataModule(L.LightningDataModule):
    """DataModule Lightning com balanceamento dinâmico pos:neg por época."""

    def __init__(
        self,
        cache_dir: Path,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        positive_negative_ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.positive_negative_ratio = positive_negative_ratio
        self.seed = seed

        self.train_dataset: VinDrPatchDataset | None = None
        self.val_dataset: VinDrPatchDataset | None = None
        self.test_dataset: VinDrPatchDataset | None = None
        self._train_sampler: WeightedRandomSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        """Carrega manifest.csv e instancia datasets por split."""
        manifest_path = self.cache_dir / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"manifest.csv não encontrado em {manifest_path}. Execute o pré-processamento antes."
            )

        df = pd.read_csv(manifest_path)

        if stage in (None, "fit"):
            train_df = df[df["split"] == "train"].reset_index(drop=True)
            val_df = df[df["split"] == "val"].reset_index(drop=True)
            self.train_dataset = VinDrPatchDataset(train_df, self.cache_dir, build_train_transforms())
            self.val_dataset = VinDrPatchDataset(val_df, self.cache_dir, build_eval_transforms())
            self._train_sampler = self._build_train_sampler(train_df)

        if stage in (None, "test"):
            test_df = df[df["split"] == "test"].reset_index(drop=True)
            self.test_dataset = VinDrPatchDataset(test_df, self.cache_dir, build_eval_transforms())

    def _build_train_sampler(self, train_df: pd.DataFrame) -> WeightedRandomSampler | None:
        """Cria sampler ponderado para manter proporção pos:neg controlada sem descartar negativos."""
        if train_df.empty or self.positive_negative_ratio <= 0:
            return None

        has_lesion = train_df["has_lesion"].astype(bool).tolist()
        n_pos = sum(has_lesion)
        n_neg = len(has_lesion) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None

        pos_weight = 1.0 / n_pos
        neg_weight = self.positive_negative_ratio / n_neg
        weights = [pos_weight if flag else neg_weight for flag in has_lesion]

        num_samples = max(1, round(n_pos * (1.0 + self.positive_negative_ratio)))
        generator = torch.Generator().manual_seed(self.seed)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        assert self.train_dataset is not None, "Chame setup('fit') antes de train_dataloader()."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self._train_sampler,
            shuffle=(self._train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=detection_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        assert self.val_dataset is not None, "Chame setup('fit') antes de val_dataloader()."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=detection_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        assert self.test_dataset is not None, "Chame setup('test') antes de test_dataloader()."
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=detection_collate_fn,
        )
