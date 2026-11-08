"""Geração de figura científica para banner acadêmico: Pipeline de Pré-processamento e Detecção.

Exibe as 4 fases essenciais:
  (a) Mamografia Bruta (DICOM) com fotometria padronizada
  (b) Isolamento da Silhueta Mamária (Cropping anatômico)
  (c) Realce Adaptativo de Contraste Local (CLAHE)
  (d) Detecção e Localização com Bounding Box e Zoom Focal na Massa
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from matplotlib import patches
from pydicom.pixel_data_handlers.util import apply_voi_lut


def load_dicom_as_uint8(dicom_path: Path) -> np.ndarray:
    """Carrega arquivo DICOM, aplica fotometria e VOI LUT, retornando matriz uint8 [0, 255]."""
    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array.astype(float)

    # 1. Tratamento Fotométrico: MONOCHROME1 vs MONOCHROME2
    photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    # 2. Janelamento Radiológico (VOI LUT)
    try:
        pixel_array = apply_voi_lut(pixel_array, dcm)
    except (ValueError, RuntimeError, AttributeError):
        # Fallback para percentis robustos se a tag VOI LUT falhar
        p_low, p_high = np.percentile(pixel_array, (0.5, 99.5))
        pixel_array = np.clip(pixel_array, p_low, p_high)

    # 3. Normalização min-max para [0, 255]
    p_min, p_max = np.min(pixel_array), np.max(pixel_array)
    if p_max > p_min:
        norm_img = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
    else:
        norm_img = np.zeros_like(pixel_array, dtype=np.uint8)

    return norm_img


def extract_breast_silhouette_crop(
    image: np.ndarray,
    padding: int = 50,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Segmenta o maior componente conexo (mama) e extrai a ROI retangular."""
    # Binarização inicial para isolar ar do tecido
    # Em imagens MONOCHROME2 o fundo é escuro (< 25)
    _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    # Abertura morfológica para desconectar etiquetas ou ruídos de digitalização
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Análise de Componentes Conexos (Largest Connected Component)
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        # Descarta o fundo (label 0) e acha o maior componente em área
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x = stats[largest_label, cv2.CC_STAT_LEFT]
        y = stats[largest_label, cv2.CC_STAT_TOP]
        w = stats[largest_label, cv2.CC_STAT_WIDTH]
        h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    else:
        # Fallback: imagem inteira
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]

    # Adiciona margem de segurança
    h_img, w_img = image.shape
    x_min = max(0, x - padding)
    y_min = max(0, y - padding)
    x_max = min(w_img, x + w + padding)
    y_max = min(h_img, y + h + padding)

    cropped = image[y_min:y_max, x_min:x_max]
    crop_box = (x_min, y_min, x_max, y_max)
    return cropped, crop_box


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Aplica CLAHE adaptativo."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def find_sample_image(
    annotations_csv: Path,
    images_dir: Path,
) -> tuple[Path, str, str, tuple[float, float, float, float], str]:
    """Localiza automaticamente um caso no VinDr-Mammo com massa confirmada e arquivo DICOM existente."""
    df = pd.read_csv(annotations_csv)
    # Filtra por Mass com Bounding Box e preferência por BI-RADS 4/5
    candidates = df[
        (df["finding_categories"].str.contains("Mass", na=False))
        & (df["xmin"].notna())
        & (df["finding_birads"].isin(["BI-RADS 4", "BI-RADS 5"]))
    ]

    for _, row in candidates.iterrows():
        study_id = str(row["study_id"])
        image_id = str(row["image_id"])
        category = str(row["finding_categories"])
        birads = str(row["finding_birads"])

        # Procura o arquivo DICOM nos padrões de diretório do VinDr
        possible_paths = [
            images_dir / study_id / f"{image_id}.dicom",
            images_dir / study_id / f"{image_id}.dcm",
            images_dir / study_id / image_id,
            images_dir / f"{image_id}.dicom",
        ]
        for p in possible_paths:
            if p.exists():
                bbox = (float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))
                label = f"{category} ({birads})"
                return p, study_id, image_id, bbox, label

    # Se não encontrar caminho DICOM existente, levanta FileNotFoundError com sugestão
    raise FileNotFoundError(
        f"Nenhum arquivo DICOM encontrado em {images_dir}. "
        "Verifique se o caminho do diretório de imagens está correto."
    )


def generate_pipeline_figure(
    dicom_path: Path,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    lesion_label: str = "Massa Tumoral (BI-RADS 5)",
    clip_limit: float = 2.0,
    dpi: int = 300,
) -> None:
    """Executa o pipeline completo e gera a figura comparativa em alta resolução."""
    print(f"-> Carregando DICOM: {dicom_path}")
    raw_img = load_dicom_as_uint8(dicom_path)
    orig_h, orig_w = raw_img.shape
    print(f"   Dimensões originais: {orig_w} x {orig_h} px")

    # 1. Cropping
    print("-> Executando isolamento da silhueta mamária (cropping)...")
    cropped_img, (c_xmin, c_ymin, _c_xmax, _c_ymax) = extract_breast_silhouette_crop(raw_img)
    crop_h, crop_w = cropped_img.shape
    print(f"   Dimensões recortadas: {crop_w} x {crop_h} px (Offset: dx={c_xmin}, dy={c_ymin})")

    # Translação da Bounding Box para coordenadas do recorte
    b_xmin, b_ymin, b_xmax, b_ymax = bbox
    t_xmin = max(0.0, b_xmin - c_xmin)
    t_ymin = max(0.0, b_ymin - c_ymin)
    t_xmax = min(float(crop_w), b_xmax - c_xmin)
    t_ymax = min(float(crop_h), b_ymax - c_ymin)

    # 2. CLAHE
    print(f"-> Aplicando CLAHE (clipLimit={clip_limit})...")
    clahe_img = apply_clahe(cropped_img, clip_limit=clip_limit)

    # 3. Inset Zoom da Lesão
    pad_zoom = 80
    z_xmin = max(0, int(t_xmin) - pad_zoom)
    z_ymin = max(0, int(t_ymin) - pad_zoom)
    z_xmax = min(crop_w, int(t_xmax) + pad_zoom)
    z_ymax = min(crop_h, int(t_ymax) + pad_zoom)
    zoom_patch = clahe_img[z_ymin:z_ymax, z_xmin:z_xmax]

    # Coordenadas da caixa relativas ao patch de zoom
    zoom_box_x = t_xmin - z_xmin
    zoom_box_y = t_ymin - z_ymin
    zoom_box_w = t_xmax - t_xmin
    zoom_box_h = t_ymax - t_ymin

    # 4. Renderização com Matplotlib (Grid de 5 colunas: 4 fases + Inset Zoom)
    print("-> Renderizando painel visual...")
    plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")
    fig = plt.figure(figsize=(20, 7), dpi=dpi)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.8, 0.8, 0.8, 0.9], wspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[0, 4])

    # Estilo dos títulos
    title_kwargs = {"fontsize": 13, "fontweight": "bold", "pad": 12, "color": "#111827"}

    # Painel (a): DICOM Bruto
    ax1.imshow(raw_img, cmap="gray")
    ax1.set_title("(a) Mamografia Bruta\n(DICOM Nativo)", **title_kwargs)
    ax1.set_xlabel(f"{orig_w} × {orig_h} px", fontsize=10, color="#4B5563")

    # Painel (b): Recorte da Silhueta
    ax2.imshow(cropped_img, cmap="gray")
    ax2.set_title("(b) Isolamento Anatômico\n(Cropping da Silhueta)", **title_kwargs)
    ax2.set_xlabel(f"{crop_w} × {crop_h} px", fontsize=10, color="#4B5563")

    # Painel (c): CLAHE
    ax3.imshow(clahe_img, cmap="gray")
    ax3.set_title("(c) Realce Adaptativo\n(Filtro CLAHE)", **title_kwargs)
    ax3.set_xlabel(f"clipLimit={clip_limit}", fontsize=10, color="#4B5563")

    # Painel (d): Detecção com Bounding Box
    ax4.imshow(clahe_img, cmap="gray")
    ax4.set_title("(d) Localização da Massa\n(Bounding Box)", **title_kwargs)
    rect = patches.Rectangle(
        (t_xmin, t_ymin),
        t_xmax - t_xmin,
        t_ymax - t_ymin,
        linewidth=2.5,
        edgecolor="#EF4444",  # Vermelho vibrante
        facecolor="none",
        linestyle="-",
    )
    ax4.add_patch(rect)
    ax4.set_xlabel("Cascade R-CNN / GT", fontsize=10, color="#4B5563")

    # Painel (e): Zoom Focal de Alta Resolução da Lesão
    ax5.imshow(zoom_patch, cmap="gray")
    ax5.set_title("(e) Detalhe da Margem\n(Zoom Focal do Tumor)", **title_kwargs)
    zoom_rect = patches.Rectangle(
        (zoom_box_x, zoom_box_y),
        zoom_box_w,
        zoom_box_h,
        linewidth=2.5,
        edgecolor="#EF4444",
        facecolor="none",
        linestyle="--",
    )
    ax5.add_patch(zoom_rect)
    ax5.text(
        zoom_box_x,
        max(0, zoom_box_y - 8),
        lesion_label,
        color="#EF4444",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#FEF2F2", "edgecolor": "#EF4444", "alpha": 0.9},
    )
    ax5.set_xlabel("Margens e Espiculações", fontsize=10, color="#4B5563")

    # Ocultar eixos para estética radiológica limpa
    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#E5E7EB")
            spine.set_linewidth(1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)

    # Também salva em PDF vetorial para impressão de alta qualidade
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    print(f"-> Figura salva com sucesso em:\n   PNG: {output_path}\n   PDF: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera figura do pipeline visual de pré-processamento e detecção para o banner."
    )
    parser.add_argument(
        "--dicom",
        type=Path,
        default=None,
        help="Caminho direto para o arquivo .dicom a ser processado.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/dados/vindr-mammo/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images"),
        help="Diretório raiz contendo as pastas de imagens DICOM do VinDr-Mammo.",
    )
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("data/finding_annotations.csv"),
        help="Caminho para o finding_annotations.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/pipeline_visual_banner.png"),
        help="Caminho de saída da imagem gerada.",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="Parâmetro clipLimit do CLAHE (default: 2.0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolução da imagem para impressão (default: 300 DPI).",
    )

    args = parser.parse_args()

    # Se um arquivo DICOM específico não foi passado, busca um caso com massa no dataset
    if args.dicom is None:
        print("-> Buscando automaticamente um caso com massa no dataset...")
        try:
            dicom_path, study_id, image_id, bbox, label = find_sample_image(
                args.annotations_csv, args.images_dir
            )
            print(f"   Caso selecionado: Estudo {study_id} | Imagem {image_id}")
            print(f"   Achado: {label} | BBox: {bbox}")
        except FileNotFoundError as e:
            print(f"Aviso: {e}")
            print("Especifique o arquivo com --dicom /caminho/para/imagem.dicom")
            return
    else:
        dicom_path = args.dicom
        # Se DICOM manual, tenta achar a anotação correspondente pelo nome do arquivo
        stem = dicom_path.stem
        df = pd.read_csv(args.annotations_csv)
        matches = df[df["image_id"] == stem]
        if not matches.empty and matches["xmin"].notna().any():
            row = matches[matches["xmin"].notna()].iloc[0]
            bbox = (float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))
            label = f"{row['finding_categories']} ({row.get('finding_birads', 'BI-RADS 4/5')})"
        else:
            # Caixa padrão centralizada se não houver anotação no CSV
            bbox = (500.0, 500.0, 1000.0, 1000.0)
            label = "Massa Tumoral"

    generate_pipeline_figure(
        dicom_path=dicom_path,
        bbox=bbox,
        output_path=args.output,
        lesion_label=label,
        clip_limit=args.clip_limit,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
