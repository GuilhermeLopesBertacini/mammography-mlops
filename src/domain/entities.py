"""Entidades de domínio para mamografia e estudos clínicos."""

from dataclasses import dataclass, field
from pathlib import Path

from src.domain.bounding_box import BoundingBox
from src.domain.enums import (
    BiRadsScore,
    BreastDensity,
    DatasetSplit,
    Laterality,
    PhotometricInterpretation,
    ViewPosition,
)
from src.domain.exceptions import InconsistentStudyError, InvalidBoundingBoxError


@dataclass
class FindingAnnotation:
    """Entidade que representa um achado patológico localizado na mamografia."""

    category: str
    birads: BiRadsScore | None = None
    bbox: BoundingBox | None = None

    @property
    def has_bbox(self) -> bool:
        """Indica se o achado possui localização espacial demarcada."""
        return self.bbox is not None


@dataclass
class MammogramImage:
    """Entidade principal representando uma incidência mamográfica individual."""

    image_id: str
    study_id: str
    series_id: str
    laterality: Laterality
    view_position: ViewPosition
    width: int
    height: int
    breast_birads: BiRadsScore
    breast_density: BreastDensity
    split: DatasetSplit
    photometric_interpretation: PhotometricInterpretation = PhotometricInterpretation.MONOCHROME2
    file_path: Path | None = None
    findings: list[FindingAnnotation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Dimensões inválidas para imagem {self.image_id}: {self.width}x{self.height}"
            )

    @property
    def has_findings(self) -> bool:
        """Indica se a imagem possui ao menos um achado patológico documentado."""
        return any(f.category != "No Finding" for f in self.findings)

    @property
    def bboxes(self) -> list[BoundingBox]:
        """Retorna todas as caixas delimitadoras associadas aos achados desta imagem."""
        return [f.bbox for f in self.findings if f.bbox is not None]

    def add_finding(self, finding: FindingAnnotation) -> None:
        """Adiciona um achado validando que a bounding box não extrapola a imagem."""
        if finding.bbox is not None and (
            finding.bbox.xmax > self.width or finding.bbox.ymax > self.height
        ):
            raise InvalidBoundingBoxError(
                f"Bounding box {finding.bbox} extrapola os limites da imagem {self.width}x{self.height}"
            )
        self.findings.append(finding)


@dataclass
class PatientStudy:
    """Agregado clínico (Aggregate Root) agrupando todas as incidências de um paciente.

    Invariantes:
    1. Todas as imagens devem pertencer ao mesmo study_id e mesmo split.
    2. O estudo idealmente reúne as 4 incidências canônicas (L-CC, L-MLO, R-CC, R-MLO).
    """

    study_id: str
    split: DatasetSplit
    images: list[MammogramImage] = field(default_factory=list)

    def add_image(self, image: MammogramImage) -> None:
        """Adiciona uma imagem assegurando consistência de estudo e ausência de vazamento de split."""
        if image.study_id != self.study_id:
            raise InconsistentStudyError(
                f"Imagem com study_id {image.study_id} incompatível com estudo {self.study_id}"
            )
        if image.split != self.split:
            raise InconsistentStudyError(
                f"Vazamento detectado: imagem split={image.split} diferente do estudo split={self.split}"
            )
        self.images.append(image)

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def lateralities(self) -> set[Laterality]:
        return {img.laterality for img in self.images}

    @property
    def view_positions(self) -> set[ViewPosition]:
        return {img.view_position for img in self.images}

    def get_image(
        self, laterality: Laterality, view_position: ViewPosition
    ) -> MammogramImage | None:
        """Recupera a imagem correspondente a uma projeção anatômica específica."""
        for img in self.images:
            if img.laterality == laterality and img.view_position == view_position:
                return img
        return None
