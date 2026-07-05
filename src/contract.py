"""Data contract : jeu de données JSON livré au Volet 2 (NLP).

Une entrée par page et par ligne, avec texte, géométrie (lignes de base et
polygones en pixels, origine en haut à gauche), confiance brute du modèle,
confiance calibrée et indicateur `needs_review`. Le schéma formel est
`schemas/data_contract.schema.json` ; `validate` le vérifie.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from src.htr import PredictedLine, PredictedPage
from src.metrics import isotonic_predict

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "data_contract.schema.json"

COORDINATES = {
    "origin": "top-left",
    "unit": "pixel",
    "x_axis": "right",
    "y_axis": "down",
    "reference": "image de la page référencée par le champ `image`, à sa résolution d'origine",
}


def inside_image(points, width: int, height: int, tolerance: int = 2) -> bool:
    """Vérifie que tous les points sont dans les limites de l'image.

    Args:
        points: Points (x, y).
        width: Largeur de l'image.
        height: Hauteur de l'image.
        tolerance: Marge tolérée en pixels.

    Returns:
        Vrai si tous les points sont dans [−tolérance, dimension + tolérance].

    Example:
        >>> inside_image([[0, 0], [10, 10]], 20, 20)
        True
    """
    a = np.asarray(points)
    return bool(len(a)) and bool((a[:, 0] >= -tolerance).all() and (a[:, 0] <= width + tolerance).all()
                                 and (a[:, 1] >= -tolerance).all() and (a[:, 1] <= height + tolerance).all())


def line_entry(line: PredictedLine, order: int, width: int, height: int, calibrator,
               review_probability: float, short_line_chars: int, normalizer=None) -> dict:
    """Construit l'entrée JSON d'une ligne et décide si elle doit être relue.

    Args:
        line: Ligne prédite.
        order: Rang de lecture dans la page.
        width: Largeur de l'image.
        height: Hauteur de l'image.
        calibrator: Modèle isotone (`metrics.isotonic_fit`) ou None.
        review_probability: Seuil de confiance calibrée sous lequel on relit.
        short_line_chars: Longueur sous laquelle une ligne est toujours relue.
        normalizer: Fonction texte → forme normalisée (facultative, pour le Volet 2).

    Returns:
        Dictionnaire conforme au schéma `line`.

    Example:
        >>> line_entry(line, 0, 2000, 3000, None, 0.5, 5)["needs_review"]
    """
    calibrated = float(isotonic_predict(calibrator, [line.confidence])[0]) if calibrator else line.confidence
    reasons = []
    if not line.text.strip():
        reasons.append("empty")
    if calibrated < review_probability:
        reasons.append("low_confidence")
    if len(line.text.strip()) < short_line_chars:
        reasons.append("short_line")
    if not (inside_image(line.polygon, width, height) and inside_image(line.baseline, width, height)):
        reasons.append("outside_image")
    entry = {
        "line_id": line.line_id,
        "order": order,
        "region_type": line.region_type,
        "text": line.text,
        "baseline": line.baseline,
        "polygon": line.polygon,
        "confidence_raw": round(float(line.confidence), 4),
        "confidence": round(min(max(calibrated, 0.0), 1.0), 4),
        "char_confidences": [round(min(max(float(c), 0.0), 1.0), 4) for c in line.char_confidences],
        "needs_review": bool(reasons),
        "review_reasons": reasons,
    }
    if normalizer is not None:
        entry["text_normalized"] = normalizer(line.text)
    return entry


def page_entry(page: PredictedPage, page_number: int, role: str, pagexml: str, image_rel: str,
               image_sha256: str | None, calibrator, review_probability: float, short_line_chars: int,
               normalizer=None) -> dict:
    """Construit l'entrée JSON d'une page.

    Args:
        page: Page prédite.
        page_number: Numéro de la page dans le document.
        role: Rôle de la page (page de titre, texte…).
        pagexml: Chemin du PAGE XML correspondant (relatif au dépôt).
        image_rel: Chemin de l'image (relatif au dépôt).
        image_sha256: Empreinte de l'image.
        calibrator: Modèle de calibration ou None.
        review_probability: Seuil de relecture.
        short_line_chars: Longueur minimale d'une ligne non suspecte.
        normalizer: Fonction de normalisation facultative.

    Returns:
        Dictionnaire conforme au schéma `page`.

    Example:
        >>> page_entry(page, 14, "texte", "segmentations/page_014.xml", "data/images/page_014.png", None, None, 0.5, 5)
    """
    entry = {
        "page_id": f"page_{page_number:03d}",
        "page_number": page_number,
        "image": image_rel,
        "width": page.width,
        "height": page.height,
        "role": role,
        "skew_angle": float(page.skew_angle),
        "pagexml": pagexml,
        "dropped_lines_in_illustrations": page.dropped,
        "regions": [{"region_id": f"r{k + 1:03d}", "type": r["type"], "polygon": r["polygon"]}
                    for k, r in enumerate(page.regions) if len(r["polygon"]) >= 3],
        "lines": [line_entry(l, k, page.width, page.height, calibrator, review_probability,
                             short_line_chars, normalizer)
                  for k, l in enumerate(page.lines) if len(l.polygon) >= 3 and len(l.baseline) >= 2],
    }
    if image_sha256:
        entry["image_sha256"] = image_sha256
    return entry


def build(document: dict, pages: list[dict], pipeline: dict) -> dict:
    """Assemble le data contract complet et ses statistiques.

    Args:
        document: Métadonnées du document (id, title, source, licence, url).
        pages: Entrées de pages (`page_entry`).
        pipeline: Description de la chaîne (modèles, prétraitement, calibration…).

    Returns:
        Le data contract.

    Example:
        >>> build({"id": "x", "title": "t", "source": "s", "licence": "l", "url": "u"}, [], {...})
    """
    n_lines = sum(len(p["lines"]) for p in pages)
    n_review = sum(l["needs_review"] for p in pages for l in p["lines"])
    return {
        "schema_version": "1.0",
        "document": document,
        "coordinate_system": COORDINATES,
        "pipeline": {"created": dt.datetime.now().isoformat(timespec="seconds")} | pipeline,
        "pages": pages,
        "stats": {"pages": len(pages), "lines": n_lines, "needs_review": n_review,
                  "needs_review_rate": round(n_review / n_lines, 4) if n_lines else 0.0},
    }


def validate(contract: dict, schema_path: Path = SCHEMA_PATH) -> None:
    """Valide un data contract contre le schéma JSON.

    Args:
        contract: Données à valider.
        schema_path: Fichier du schéma.

    Raises:
        jsonschema.ValidationError: Si le contrat ne respecte pas le schéma.

    Example:
        >>> validate(json.loads(Path("dataset_nlp/charles_viii_1494.json").read_text()))
    """
    import jsonschema
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(contract, schema, cls=jsonschema.Draft202012Validator)
