"""CLI Hydra para geração offline do cache de patches e manifest.csv."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.application.preprocess_dataset import PreprocessDatasetUseCase
from src.application.tiling_engine import TilingEngine
from src.infrastructure.config import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    PreprocessingConfig,
    TrackingConfig,
    TrainerConfig,
)
from src.infrastructure.datasets.vindr_reader import VinDrDatasetReaderAdapter
from src.infrastructure.preprocessing import OpenCVMammogramPreprocessor

logger = logging.getLogger(__name__)


def _validate_config(cfg: DictConfig) -> PipelineConfig:
    """Converte DictConfig do Hydra em contrato Pydantic imutável (Fail-Fast)."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Configuração Hydra raiz deve ser um mapeamento.")

    return PipelineConfig(
        seed=raw["seed"],
        data=DataConfig(**raw["data"]),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        model=ModelConfig(**raw["model"]),
        trainer=TrainerConfig(**raw["trainer"]),
        tracking=TrackingConfig(**raw["tracking"]),
    )


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Composition root do pré-processamento offline."""
    pipeline = _validate_config(cfg)
    raw_images_dir = pipeline.data.require_raw_images_dir()
    annotations_dir = Path(pipeline.data.annotations_csv).parent

    reader = VinDrDatasetReaderAdapter(annotations_dir)
    preprocessor = OpenCVMammogramPreprocessor(
        min_size=pipeline.preprocessing.patch_size,
        crop_black_background=pipeline.preprocessing.crop_black_background,
        clahe_clip_limit=pipeline.preprocessing.clahe_clip_limit,
        clahe_tile_grid_size=pipeline.preprocessing.clahe_tile_grid_size,
    )
    tiling_engine = TilingEngine(
        patch_size=pipeline.preprocessing.patch_size,
        overlap_ratio=pipeline.preprocessing.overlap_ratio,
        min_bbox_coverage=pipeline.preprocessing.min_bbox_coverage,
    )

    use_case = PreprocessDatasetUseCase(
        reader=reader,
        preprocessor=preprocessor,
        tiling_engine=tiling_engine,
    )

    logger.info("Iniciando pré-processamento para cache em: %s", pipeline.data.cache_dir)
    manifest_path = use_case.execute(
        raw_images_dir=raw_images_dir,
        annotations_dir=annotations_dir,
        cache_dir=pipeline.data.cache_dir,
        val_fraction=pipeline.data.val_fraction,
        seed=pipeline.seed,
        exclude_birads_from_negatives=pipeline.data.exclude_birads_from_negatives,
        max_negative_studies=pipeline.data.max_negative_studies,
        max_workers=pipeline.data.num_workers,
    )
    logger.info("Pré-processamento concluído. Manifesto salvo em: %s", manifest_path)


if __name__ == "__main__":
    main()
