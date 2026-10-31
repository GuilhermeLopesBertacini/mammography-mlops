"""Exceções canônicas de domínio para proteção de invariantes de negócio e geometria."""


class DomainException(Exception):
    """Exceção base para todas as violações de regras do domínio."""


class InvalidBoundingBoxError(DomainException):
    """Lançada quando uma BoundingBox viola coordenadas cartesianas ou limites da imagem."""


class PatientLeakageError(DomainException):
    """Lançada quando exames do mesmo paciente aparecem em partições (splits) distintas."""


class InconsistentStudyError(DomainException):
    """Lançada quando a composição anatômica de um estudo é inválida."""
