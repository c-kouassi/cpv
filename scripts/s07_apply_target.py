"""Étape 7 — Application au document cible (Charles VIII, 1494) et production des livrables.

Pour chaque page de texte (13 à 20) : prétraitement retenu → segmentation de
mise en page → reconnaissance par le modèle affiné → confiance calibrée →
PAGE XML (`segmentations/`) et data contract JSON validé (`dataset_nlp/`).

Usage :
    python -m scripts.s07_apply_target
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

import config
from src import contract, htr, preprocess
from src.corpus import sha256_file
from src.normalize_text import normalize
from src.repro import _git_commit, fixer_seeds, log_experiment


def main() -> None:
    """Transcrit les pages de texte du document cible et écrit les livrables."""
    fixer_seeds(config.SEED)
    t0 = time.time()
    if not config.IMAGES.exists() or len(list(config.IMAGES.glob("page_*.png"))) != config.N_PAGES:
        from src.extract_pages import extract_pages
        extract_pages(config.RAW_PDF, config.IMAGES, config.DPI)
    test = json.loads((config.RESULTS / "test" / "resultats.json").read_text())
    seg_choice = json.loads((config.RESULTS / "layout" / "segmenteurs_val.json").read_text())
    variant = test["selection_validation"]["pretraitement_inference"]
    calib = json.loads((config.MODELS / "calibration" / "isotone_val.json").read_text())
    import numpy as np
    calibrator = (np.array(calib["x"]), np.array(calib["y"]))
    model_path = config.MODELS / "kraken" / "charles8_htr_ft.mlmodel"
    model = htr.load_model(model_path)
    segmenter = htr.load_segmenter(seg_choice["chemin"])
    filt = json.loads((config.RESULTS / "layout" / "filtre_illustration_val.json").read_text())
    overlap = filt["seuil"] if filt["adopte"] else None

    pages = []
    for n in config.TARGET_TEXT_PAGES:
        image = config.IMAGES / f"page_{n:03d}.png"
        prep = preprocess.run(cv2.imread(str(image)), preprocess.VARIANTS[variant], config.PREPROCESS)
        pred = htr.predict_page(image, prep, model, segmenter, illustration_overlap=overlap)
        xml = config.SEGMENTATIONS / f"page_{n:03d}.xml"
        htr.to_pagexml(pred, xml)
        pages.append(contract.page_entry(
            pred, n, config.PAGE_ROLES[n], str(xml.relative_to(config.ROOT)),
            str(image.relative_to(config.ROOT)), sha256_file(image), calibrator,
            config.REVIEW_PROBABILITY, config.SHORT_LINE_CHARS, normalize))
        print(f"page {n} : {len(pred.lines)} lignes, {pred.dropped} écartées (illustration), angle {pred.skew_angle}", flush=True)

    data = contract.build(
        {"id": config.TARGET_ID, "title": config.TARGET_TITLE,
         "source": "gallica.bnf.fr / Bibliothèque nationale de France",
         "licence": "Réutilisation non commerciale libre (conditions Gallica) ; transcriptions : CC-BY 4.0",
         "url": f"https://gallica.bnf.fr/ark:/12148/{config.TARGET_ID}"},
        pages,
        {"preprocessing": {"variant": variant, "steps": preprocess.VARIANTS[variant], "params": config.PREPROCESS},
         "segmentation_model": {"name": seg_choice["meilleur"],
                                "path": str(Path(seg_choice["chemin"]).relative_to(config.ROOT)) if seg_choice["chemin"] else None,
                                "illustration_filter": {"adopted": filt["adopte"], "overlap": filt["seuil"]}},
         "htr_model": {"name": "charles8_htr_ft", "path": str(model_path.relative_to(config.ROOT)),
                       "sha256": sha256_file(model_path),
                       "base": json.loads((config.RESULTS / "baseline" / "classement_val.json").read_text())["meilleur"],
                       "cer_test": test["lignes_reference"]["affine"]["cer"]},
         "calibration": {"method": "régression isotone sur la validation", "target": calib["cible"],
                         "needs_review_if": f"confiance calibrée < {config.REVIEW_PROBABILITY} ou ligne < {config.SHORT_LINE_CHARS} caractères ou polygone hors image"},
         "git_commit": _git_commit(config.ROOT)})
    contract.validate(data)
    config.DATASET_NLP.mkdir(parents=True, exist_ok=True)
    out = config.DATASET_NLP / "charles_viii_1494.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in data["pages"]:
        (config.DATASET_NLP / "texte").mkdir(exist_ok=True)
        (config.DATASET_NLP / "texte" / f"{p['page_id']}.txt").write_text(
            "\n".join(l["text"] for l in p["lines"]) + "\n", encoding="utf-8")
    (config.SEGMENTATIONS / "COORDONNEES.json").write_text(json.dumps(
        contract.COORDINATES | {"format": "PAGE XML 2019 (kraken), une page par fichier",
                                "images": "data/images/page_XXX.png (PDF Gallica rendu à 300 DPI)"},
        indent=1, ensure_ascii=False))
    overlay_figures(data)
    log_experiment(config.EXPERIMENTS / "journal.jsonl", "application_document_cible",
                   {"pages": config.TARGET_TEXT_PAGES, "pretraitement": variant, "segmenteur": seg_choice["meilleur"]},
                   data["stats"], [str(out.relative_to(config.ROOT)), "segmentations/"], time.time() - t0)
    print(json.dumps(data["stats"], ensure_ascii=False))


def overlay_figures(data: dict) -> None:
    """Dessine, pour chaque page, les zones et les lignes (vert : validée, orange : à relire).

    Args:
        data: Data contract produit par `main`.

    Example:
        >>> overlay_figures(json.loads(Path("dataset_nlp/charles_viii_1494.json").read_text()))
    """
    import numpy as np
    out = config.FIGURES / "cible"
    out.mkdir(parents=True, exist_ok=True)
    colors = {"illustration": (180, 60, 160), "texte": (200, 120, 40), "text": (200, 120, 40), "paratexte": (60, 160, 200)}
    for p in data["pages"]:
        img = cv2.imread(str(config.ROOT / p["image"]))
        for r in p["regions"]:
            cv2.polylines(img, [np.asarray(r["polygon"], np.int32)], True, colors.get(r["type"], (128, 128, 128)), 3)
        for l in p["lines"]:
            c = (0, 140, 255) if l["needs_review"] else (60, 170, 60)
            cv2.polylines(img, [np.asarray(l["polygon"], np.int32)], True, c, 4)
        scale = 1600 / img.shape[0]
        cv2.imwrite(str(out / f"{p['page_id']}.jpg"), cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
