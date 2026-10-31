"""Enumerações e Value Objects canônicos do domínio radiológico."""

from enum import Enum


class Laterality(str, Enum):
    """Lateralidade anatômica da mama."""

    LEFT = "L"
    RIGHT = "R"


class ViewPosition(str, Enum):
    """Projeção radiológica da incidência mamográfica."""

    CC = "CC"  # Crânio-Caudal (vista superior)
    MLO = "MLO"  # Médio-Lateral Oblíqua (vista angular incluindo peitoral e axila)


class BreastDensity(str, Enum):
    """Classificação de densidade mamária segundo o padrão ACR BI-RADS."""

    DENSITY_A = "DENSITY A"  # Predominantemente gordurosa (< 25% glandular)
    DENSITY_B = "DENSITY B"  # Fibroglandular dispersa (25-50% glandular)
    DENSITY_C = "DENSITY C"  # Heterogeneamente densa (51-75% glandular)
    DENSITY_D = "DENSITY D"  # Extremamente densa (> 75% glandular)


class BiRadsScore(str, Enum):
    """Categorias de avaliação diagnóstica BI-RADS."""

    BI_RADS_1 = "BI-RADS 1"  # Negativo / Normal
    BI_RADS_2 = "BI-RADS 2"  # Achado tipicamente benigno
    BI_RADS_3 = "BI-RADS 3"  # Achado provavelmente benigno (acompanhamento 6 meses)
    BI_RADS_4 = "BI-RADS 4"  # Suspeito de malignidade (indicação de biópsia)
    BI_RADS_5 = "BI-RADS 5"  # Altamente sugestivo de malignidade (> 95% risco)


class PhotometricInterpretation(str, Enum):
    """Interpretação radiológica dos valores numéricos dos pixels no DICOM."""

    MONOCHROME1 = "MONOCHROME1"  # Menor valor = branco (precisa de inversão)
    MONOCHROME2 = "MONOCHROME2"  # Menor valor = preto (padrão digital moderno)


class DatasetSplit(str, Enum):
    """Particionamento de dataset para treino e avaliação sem vazamento."""

    TRAINING = "training"
    TEST = "test"
    VALIDATION = "validation"
