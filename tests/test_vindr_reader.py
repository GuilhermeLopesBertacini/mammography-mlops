"""Teste de integração para o VinDrDatasetReaderAdapter utilizando os CSVs reais."""

from pathlib import Path

from src.domain.enums import DatasetSplit, Laterality, ViewPosition
from src.infrastructure.datasets.vindr_reader import VinDrDatasetReaderAdapter


def test_vindr_reader_loads_sample_studies():
    data_dir = Path("/home/guilherme/Documents/Projects/mammography-mlops/data")
    adapter = VinDrDatasetReaderAdapter(data_dir=data_dir)

    # Executa o carregamento
    studies = adapter.load_studies()

    assert len(studies) == 5000

    # Amostra um estudo de teste
    first_study = studies[0]
    assert first_study.image_count == 4
    assert first_study.split in (DatasetSplit.TRAINING, DatasetSplit.TEST)

    # Verifica se as 4 incidências estão presentes
    assert first_study.get_image(Laterality.LEFT, ViewPosition.CC) is not None
    assert first_study.get_image(Laterality.LEFT, ViewPosition.MLO) is not None
    assert first_study.get_image(Laterality.RIGHT, ViewPosition.CC) is not None
    assert first_study.get_image(Laterality.RIGHT, ViewPosition.MLO) is not None
