"""Adapter concreto para ingestão e estruturação dos metadados e anotações do VinDr-Mammo."""

import ast
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from src.domain.bounding_box import BoundingBox
from src.domain.entities import FindingAnnotation, MammogramImage, PatientStudy
from src.domain.enums import (
    BiRadsScore,
    BreastDensity,
    DatasetSplit,
    Laterality,
    PhotometricInterpretation,
    ViewPosition,
)
from src.ports.dataset_reader import DatasetReaderPort


class VinDrDatasetReaderAdapter(DatasetReaderPort):
    """Adapter que consome os arquivos CSV do VinDr-Mammo e instancia agregados do domínio."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.metadata_csv = data_dir / "metadata.csv"
        self.breast_csv = data_dir / "breast-level_annotations.csv"
        self.finding_csv = data_dir / "finding_annotations.csv"

    def load_studies(self, data_dir: Optional[Path] = None) -> List[PatientStudy]:
        """Lê os CSVs, faz o join relacional e retorna os PatientStudy agregados."""
        csv_dir = data_dir or self.data_dir
        meta_df = pd.read_csv(csv_dir / "metadata.csv")
        breast_df = pd.read_csv(csv_dir / "breast-level_annotations.csv")
        finding_df = pd.read_csv(csv_dir / "finding_annotations.csv")

        # Mapeamento rápido de fotometria por SOP Instance UID
        photometric_map: Dict[str, PhotometricInterpretation] = {}
        for _, row in meta_df.iterrows():
            sop_uid = str(row["SOP Instance UID"])
            photo_val = str(row["Photometric Interpretation"]).strip()
            photometric_map[sop_uid] = (
                PhotometricInterpretation.MONOCHROME1
                if photo_val == "MONOCHROME1"
                else PhotometricInterpretation.MONOCHROME2
            )

        # Mapeamento de findings agrupados por image_id
        findings_by_image: Dict[str, List[FindingAnnotation]] = {}
        for _, row in finding_df.iterrows():
            img_id = str(row["image_id"])
            cat_raw = row["finding_categories"]
            # Converter string de lista "['Mass']" para categorias limpas
            if isinstance(cat_raw, str) and cat_raw.startswith("["):
                try:
                    cats = ast.literal_eval(cat_raw)
                    category_name = ", ".join(cats) if isinstance(cats, list) else str(cat_raw)
                except Exception:
                    category_name = cat_raw
            else:
                category_name = str(cat_raw)

            finding_birads_raw = row.get("finding_birads")
            finding_birads = None
            if pd.notna(finding_birads_raw):
                try:
                    finding_birads = BiRadsScore(str(finding_birads_raw).strip())
                except ValueError:
                    finding_birads = None

            bbox = None
            if pd.notna(row["xmin"]) and pd.notna(row["ymin"]):
                # Sanitização na borda da infraestrutura: anotações manuais dos radiologistas
                # podem extrapolar ligeiramente as margens da imagem (ex: xmin = -6.25px).
                raw_xmin = float(row["xmin"])
                raw_ymin = float(row["ymin"])
                raw_xmax = float(row["xmax"])
                raw_ymax = float(row["ymax"])
                img_width = float(row["width"])
                img_height = float(row["height"])

                clipped_xmin = max(0.0, min(raw_xmin, img_width))
                clipped_ymin = max(0.0, min(raw_ymin, img_height))
                clipped_xmax = max(0.0, min(raw_xmax, img_width))
                clipped_ymax = max(0.0, min(raw_ymax, img_height))

                if clipped_xmax > clipped_xmin and clipped_ymax > clipped_ymin:
                    bbox = BoundingBox(
                        xmin=clipped_xmin,
                        ymin=clipped_ymin,
                        xmax=clipped_xmax,
                        ymax=clipped_ymax,
                    )

            annotation = FindingAnnotation(
                category=category_name,
                birads=finding_birads,
                bbox=bbox,
            )

            if img_id not in findings_by_image:
                findings_by_image[img_id] = []
            findings_by_image[img_id].append(annotation)

        # Construção dos agregados PatientStudy e MammogramImage
        studies: Dict[str, PatientStudy] = {}

        for _, row in breast_df.iterrows():
            study_id = str(row["study_id"])
            img_id = str(row["image_id"])
            series_id = str(row["series_id"])
            split_val = DatasetSplit(str(row["split"]).strip().lower())

            if study_id not in studies:
                studies[study_id] = PatientStudy(study_id=study_id, split=split_val)

            photo = photometric_map.get(img_id, PhotometricInterpretation.MONOCHROME2)

            image = MammogramImage(
                image_id=img_id,
                study_id=study_id,
                series_id=series_id,
                laterality=Laterality(str(row["laterality"]).strip()),
                view_position=ViewPosition(str(row["view_position"]).strip()),
                width=int(row["width"]),
                height=int(row["height"]),
                breast_birads=BiRadsScore(str(row["breast_birads"]).strip()),
                breast_density=BreastDensity(str(row["breast_density"]).strip()),
                split=split_val,
                photometric_interpretation=photo,
            )

            # Associar findings
            for f in findings_by_image.get(img_id, []):
                image.add_finding(f)

            studies[study_id].add_image(image)

        return list(studies.values())
