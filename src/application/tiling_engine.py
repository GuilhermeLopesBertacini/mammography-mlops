"""Casos de uso para fatiamento de imagens em patches (Tiling) e remapeamento geométrico."""

from dataclasses import dataclass

from src.domain.bounding_box import BoundingBox


@dataclass(frozen=True)
class PatchProposal:
    """Representa um recorte espacial gerado pela janela deslizante."""

    patch_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    local_bboxes: list[BoundingBox]

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


class TilingEngine:
    """Gera janelas deslizantes (patches) preservando o pitch e resolução espacial nativa."""

    def __init__(
        self,
        patch_size: int = 1024,
        overlap_ratio: float = 0.20,
        min_bbox_coverage: float = 0.30,
    ) -> None:
        """
        Args:
            patch_size: Dimensão quadrada do patch (ex: 1024px).
            overlap_ratio: Fração de sobreposição entre janelas adjacentes (ex: 0.20 = 20%).
            min_bbox_coverage: Cobertura mínima da área da caixa delimitadora para ser mantida no patch.
        """
        if patch_size <= 0:
            raise ValueError(f"patch_size deve ser positivo: {patch_size}")
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError(f"overlap_ratio deve estar em [0.0, 1.0): {overlap_ratio}")

        self.patch_size = patch_size
        self.step_size = int(patch_size * (1.0 - overlap_ratio))
        self.min_bbox_coverage = min_bbox_coverage

    def generate_patches(
        self,
        image_width: int,
        image_height: int,
        target_bboxes: list[BoundingBox],
    ) -> list[PatchProposal]:
        """Calcula os offsets e gera patches com projeção local de caixas."""
        patches: list[PatchProposal] = []
        patch_idx = 0

        y_coords = self._calculate_coordinates(image_height)
        x_coords = self._calculate_coordinates(image_width)

        for y_start in y_coords:
            y_end = min(image_height, y_start + self.patch_size)
            y_start = max(0, y_end - self.patch_size)  # Garante tamanho fixo na borda

            for x_start in x_coords:
                x_end = min(image_width, x_start + self.patch_size)
                x_start = max(0, x_end - self.patch_size)

                # Mapeia caixas para o patch
                local_boxes: list[BoundingBox] = []
                for box in target_bboxes:
                    clipped = box.clip_to_patch(x_start, y_start, x_end, y_end)
                    if clipped is not None and (clipped.area / box.area) >= self.min_bbox_coverage:
                        local_boxes.append(clipped)

                patches.append(
                    PatchProposal(
                        patch_id=patch_idx,
                        xmin=float(x_start),
                        ymin=float(y_start),
                        xmax=float(x_end),
                        ymax=float(y_end),
                        local_bboxes=local_boxes,
                    )
                )
                patch_idx += 1

        return patches

    def _calculate_coordinates(self, max_dim: int) -> list[int]:
        """Calcula a lista de coordenadas iniciais garantindo cobertura total."""
        if max_dim <= self.patch_size:
            return [0]

        coords: list[int] = []
        curr = 0
        while curr + self.patch_size < max_dim:
            coords.append(curr)
            curr += self.step_size

        # Garante que o extremo final da imagem seja coberto
        coords.append(max_dim - self.patch_size)
        return sorted(set(coords))
