"""Value Object BoundingBox com proteção estrita de invariantes geométricas."""

from dataclasses import dataclass
from typing import Optional

from src.domain.exceptions import InvalidBoundingBoxError


@dataclass(frozen=True)
class BoundingBox:
    """Value Object imutável que representa uma caixa delimitadora em coordenadas cartesianas.

    Invariantes:
    1. xmin, ymin >= 0
    2. xmax > xmin e ymax > ymin (área estritamente positiva)
    3. Se limites de imagem fornecidos (max_width, max_height): xmax <= max_width, ymax <= max_height
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmin < 0 or self.ymin < 0:
            raise InvalidBoundingBoxError(
                f"Coordenadas não podem ser negativas: xmin={self.xmin}, ymin={self.ymin}"
            )
        if self.xmax <= self.xmin:
            raise InvalidBoundingBoxError(
                f"xmax ({self.xmax}) deve ser estritamente maior que xmin ({self.xmin})"
            )
        if self.ymax <= self.ymin:
            raise InvalidBoundingBoxError(
                f"ymax ({self.ymax}) deve ser estritamente maior que ymin ({self.ymin})"
            )

    @property
    def width(self) -> float:
        """Largura absoluta da caixa delimitadora."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Altura absoluta da caixa delimitadora."""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """Área geométrica da caixa delimitadora em pixels quadrados."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Ponto central (cx, cy) da caixa."""
        return (self.xmin + self.width / 2.0, self.ymin + self.height / 2.0)

    def to_coco_format(self) -> tuple[float, float, float, float]:
        """Retorna no padrão MS COCO: [xmin, ymin, width, height]."""
        return (self.xmin, self.ymin, self.width, self.height)

    def to_pascal_voc(self) -> tuple[float, float, float, float]:
        """Retorna no padrão Pascal VOC: [xmin, ymin, xmax, ymax]."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def translate(self, offset_x: float, offset_y: float) -> "BoundingBox":
        """Translada as coordenadas após uma operação geométrica (ex: cropping de imagem)."""
        new_xmin = self.xmin - offset_x
        new_ymin = self.ymin - offset_y
        new_xmax = self.xmax - offset_x
        new_ymax = self.ymax - offset_y

        return BoundingBox(
            xmin=max(0.0, new_xmin),
            ymin=max(0.0, new_ymin),
            xmax=new_xmax,
            ymax=new_ymax,
        )

    def clip_to_patch(
        self, patch_xmin: float, patch_ymin: float, patch_xmax: float, patch_ymax: float
    ) -> Optional["BoundingBox"]:
        """Calcula a interseção da caixa com uma janela (patch).

        Retorna a nova BoundingBox com coordenadas locais relativas ao patch,
        ou None se não houver sobreposição positiva com o patch.
        """
        inter_xmin = max(self.xmin, patch_xmin)
        inter_ymin = max(self.ymin, patch_ymin)
        inter_xmax = min(self.xmax, patch_xmax)
        inter_ymax = min(self.ymax, patch_ymax)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return None

        # Coordenadas relativas ao patch [0, patch_w] e [0, patch_h]
        local_xmin = inter_xmin - patch_xmin
        local_ymin = inter_ymin - patch_ymin
        local_xmax = inter_xmax - patch_xmin
        local_ymax = inter_ymax - patch_ymin

        return BoundingBox(
            xmin=local_xmin,
            ymin=local_ymin,
            xmax=local_xmax,
            ymax=local_ymax,
        )

    def compute_iou(self, other: "BoundingBox") -> float:
        """Calcula o índice de Interseção sobre União (IoU) com outra caixa."""
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
