"""Cascade R-CNN sobre TimmFPNBackbone (ConvNeXt) encapsulado em LightningModule."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from src.infrastructure.models.backbone import TimmFPNBackbone


class CascadeRoIHeads(nn.Module):
    """Encadeia K estágios sequenciais de RoIHeads com limiares crescentes de IoU."""

    def __init__(
        self,
        box_roi_pool: MultiScaleRoIAlign,
        in_channels: int,
        num_classes: int,
        iou_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7),
        representation_size: int = 1024,
        batch_size_per_image: int = 512,
        positive_fraction: float = 0.25,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
    ) -> None:
        super().__init__()
        self.num_stages = len(iou_thresholds)
        self.stages = nn.ModuleList()

        resolution = box_roi_pool.output_size[0]
        for idx, thr in enumerate(iou_thresholds):
            scale = 2**idx
            reg_weights = (10.0 * scale, 10.0 * scale, 5.0 * scale, 5.0 * scale)
            box_head = TwoMLPHead(in_channels * (resolution**2), representation_size)
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
            stage = RoIHeads(
                box_roi_pool=box_roi_pool,
                box_head=box_head,
                box_predictor=box_predictor,
                fg_iou_thresh=thr,
                bg_iou_thresh=thr,
                batch_size_per_image=batch_size_per_image,
                positive_fraction=positive_fraction,
                bbox_reg_weights=reg_weights,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                detections_per_img=detections_per_img,
            )
            self.stages.append(stage)

    def _refine_proposals(
        self,
        stage: RoIHeads,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        filter_small: bool,
    ) -> list[torch.Tensor]:
        """Decodifica deltas do estágio atual para alimentar o próximo estágio (sem gradiente)."""
        with torch.no_grad():
            pred_boxes = stage.box_coder.decode(box_regression.detach(), proposals)
            # Seleciona a classe foreground mais provável para cada proposta
            fg_classes = class_logits.detach()[:, 1:].argmax(dim=1) + 1
            row_indices = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)
            chosen_boxes = pred_boxes[row_indices, fg_classes]

            boxes_per_image = [len(p) for p in proposals]
            split_boxes = chosen_boxes.split(boxes_per_image, dim=0)

            refined: list[torch.Tensor] = []
            for boxes, img_shape in zip(split_boxes, image_shapes, strict=True):
                clipped = box_ops.clip_boxes_to_image(boxes, img_shape)
                if filter_small:
                    keep = box_ops.remove_small_boxes(clipped, min_size=1.0)
                    clipped = clipped[keep]
                refined.append(clipped)
            return refined

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if self.training:
            if targets is None:
                raise ValueError("targets é obrigatório durante o treinamento do CascadeRoIHeads.")
            losses: dict[str, torch.Tensor] = {}
            curr_proposals = proposals

            for idx, stage in enumerate(self.stages):
                assert isinstance(stage, RoIHeads)
                sampled_props, _, labels, reg_targets = stage.select_training_samples(
                    curr_proposals, targets
                )
                box_features = stage.box_roi_pool(features, sampled_props, image_shapes)
                box_features = stage.box_head(box_features)
                class_logits, box_regression = stage.box_predictor(box_features)

                loss_cls, loss_box = fastrcnn_loss(
                    class_logits, box_regression, labels, reg_targets
                )
                losses[f"stage_{idx}_loss_classifier"] = loss_cls
                losses[f"stage_{idx}_loss_box_reg"] = loss_box

                if idx < self.num_stages - 1:
                    curr_proposals = self._refine_proposals(
                        stage,
                        class_logits,
                        box_regression,
                        sampled_props,
                        image_shapes,
                        filter_small=True,
                    )
            return [], losses

        # Inferência: refina propostas em cascata e combina probabilidades dos estágios
        curr_proposals = proposals
        stage_probs: list[torch.Tensor] = []
        last_box_reg: torch.Tensor | None = None
        proposals_for_last: list[torch.Tensor] = curr_proposals

        for idx, stage in enumerate(self.stages):
            assert isinstance(stage, RoIHeads)
            box_features = stage.box_roi_pool(features, curr_proposals, image_shapes)
            box_features = stage.box_head(box_features)
            class_logits, box_regression = stage.box_predictor(box_features)

            stage_probs.append(F.softmax(class_logits, dim=-1))
            if idx < self.num_stages - 1:
                curr_proposals = self._refine_proposals(
                    stage,
                    class_logits,
                    box_regression,
                    curr_proposals,
                    image_shapes,
                    filter_small=False,
                )
            else:
                last_box_reg = box_regression
                proposals_for_last = curr_proposals

        assert last_box_reg is not None
        avg_probs = sum(stage_probs) / float(self.num_stages)
        ensemble_logits = torch.log(avg_probs.clamp(min=1e-8))

        final_stage = self.stages[-1]
        assert isinstance(final_stage, RoIHeads)
        boxes, scores, labels = final_stage.postprocess_detections(
            ensemble_logits, last_box_reg, proposals_for_last, image_shapes
        )
        detections = [
            {"boxes": b, "scores": s, "labels": lbl}
            for b, s, lbl in zip(boxes, scores, labels, strict=True)
        ]
        return detections, {}


class CascadeRCNNDetector(nn.Module):
    """Detector completo: GeneralizedRCNNTransform + TimmFPNBackbone + RPN + CascadeRoIHeads."""

    def __init__(
        self,
        backbone: TimmFPNBackbone,
        num_classes: int = 2,
        cascade_iou_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7),
        min_size: int = 1024,
        max_size: int = 1024,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # Albumentations já normaliza e os patches já chegam em 1024x1024
        self.transform = GeneralizedRCNNTransform(
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
        )

        # 5 níveis na FPN (4 estágios do ConvNeXt + LastLevelMaxPool)
        anchor_generator = AnchorGenerator(
            sizes=((16, 32), (32, 64), (64, 128), (128, 256), (256, 512)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        rpn_head = RPNHead(
            backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
        )
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
        )

        roi_pool = MultiScaleRoIAlign(
            featmap_names=backbone.featmap_names,
            output_size=7,
            sampling_ratio=2,
        )
        self.roi_heads = CascadeRoIHeads(
            box_roi_pool=roi_pool,
            in_channels=backbone.out_channels,
            num_classes=num_classes,
            iou_thresholds=cascade_iou_thresholds,
        )

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
        image_list, targets = self.transform(images, targets)
        assert isinstance(image_list, ImageList)

        features = self.backbone(image_list.tensors)
        proposals, rpn_losses = self.rpn(image_list, features, targets)
        detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        detections = self.transform.postprocess(detections, image_list.image_sizes, original_image_sizes)

        losses: dict[str, torch.Tensor] = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)
        return detections, losses


class CascadeRCNNLightningModule(L.LightningModule):
    """Módulo PyTorch Lightning para treino, validação (mAP) e otimização."""

    def __init__(
        self,
        backbone: TimmFPNBackbone,
        num_classes: int = 2,
        cascade_iou_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7),
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        image_size: int = 1024,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.detector = CascadeRCNNDetector(
            backbone=backbone,
            num_classes=num_classes,
            cascade_iou_thresholds=cascade_iou_thresholds,
            min_size=image_size,
            max_size=image_size,
        )
        self._assert_no_batchnorm()
        self.val_map = MeanAveragePrecision(iou_type="bbox", class_metrics=False)

    def _assert_no_batchnorm(self) -> None:
        """Garante invariante arquitetural: nenhuma camada BatchNorm ativa no detector."""
        for name, module in self.detector.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                raise TypeError(
                    f"BatchNorm detectado em '{name}' — incompatível com micro-batches."
                )

    def training_step(
        self,
        batch: tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]],
        batch_idx: int,
    ) -> torch.Tensor:
        images, targets = batch
        _, loss_dict = self.detector(images, targets)
        total_loss = sum(loss_dict.values())
        assert isinstance(total_loss, torch.Tensor)

        batch_size = len(images)
        self.log("train/loss", total_loss, prog_bar=True, batch_size=batch_size)
        for key, val in loss_dict.items():
            self.log(f"train/{key}", val, batch_size=batch_size)
        return total_loss

    def validation_step(
        self,
        batch: tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]],
        batch_idx: int,
    ) -> None:
        images, targets = batch
        detections, _ = self.detector(images)
        self.val_map.update(detections, targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_map.compute()
        self.log("val/map", metrics["map"], prog_bar=True)
        self.log("val/map_50", metrics["map_50"], prog_bar=True)
        self.log("val/map_75", metrics["map_75"])
        self.val_map.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps <= 1:
            return {"optimizer": optimizer}

        warmup_steps = min(
            max(1, int(total_steps * (self.warmup_epochs / max(1, self.trainer.max_epochs or 1)))),
            max(1, total_steps - 1),
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=self.learning_rate * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
