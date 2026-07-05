"""Étape 6b — Filtre géométrique des lignes situées dans les illustrations, décidé sur la VALIDATION.

Constat (document cible, page de titre) : le segmenteur détecte la gravure comme
zone « illustration » mais rattache les hachures, lues comme des lignes, à la
zone « text » héritée du modèle d'origine ; le filtre par type de zone ne les
écarte donc pas. Ce script compare, sur toute la validation et avec la chaîne
finale, la chaîne sans filtre et avec un filtre géométrique (ligne écartée si
au moins `config.ILLUSTRATION_OVERLAP` de sa surface est dans une illustration).

Le filtre a été conçu APRÈS l'évaluation unique du test : les chiffres du test
restent ceux de la chaîne sans filtre (results/test/resultats.json).

Usage :
    python -m scripts.s06b_illustration_filter_val
"""
from __future__ import annotations

import json
import time

import config
from src import evaluation, htr
from src.preprocess import VARIANTS
from src.repro import fixer_seeds, log_experiment


def main() -> None:
    """Compare la chaîne avec et sans filtre et enregistre la décision."""
    fixer_seeds(config.SEED)
    t0 = time.time()
    test = json.loads((config.RESULTS / "test" / "resultats.json").read_text())
    variant = test["selection_validation"]["pretraitement_inference"]
    seg_choice = json.loads((config.RESULTS / "layout" / "segmenteurs_val.json").read_text())
    model = htr.load_model(config.MODELS / "kraken" / "charles8_htr_ft.mlmodel")
    segmenter = htr.load_segmenter(seg_choice["chemin"])
    pages = evaluation.load_split_pages(config.CORPUS / "val")
    out = {}
    for name, overlap in [("sans_filtre", None), ("filtre_geometrique", config.ILLUSTRATION_OVERLAP)]:
        res = evaluation.run_end_to_end(pages, VARIANTS[variant], config.PREPROCESS, model, segmenter,
                                        "mps", config.IOU_THRESHOLD, illustration_overlap=overlap)
        out[name] = res.summary(config.BOOTSTRAP_N, config.ALPHA, config.SEED, config.IOU_THRESHOLD)
        print(name, json.dumps({k: out[name][k] for k in ["cer", "cer_ic95", "wer", "iou_lignes_moyenne",
                                                           "detection_f1_moyen"]}), flush=True)
    adopt = out["filtre_geometrique"]["cer"] <= out["sans_filtre"]["cer"]
    decision = {"adopte": adopt, "seuil": config.ILLUSTRATION_OVERLAP, "pretraitement": variant,
                "pages_val": len(pages), "resultats": out,
                "regle": "adopté si le CER de bout en bout de validation n'augmente pas"}
    (config.RESULTS / "layout" / "filtre_illustration_val.json").write_text(json.dumps(decision, indent=1, ensure_ascii=False))
    log_experiment(config.EXPERIMENTS / "journal.jsonl", "filtre_illustration_val",
                   {"seuil": config.ILLUSTRATION_OVERLAP, "pretraitement": variant, "segmenteur": seg_choice["meilleur"]},
                   {"cer_sans": out["sans_filtre"]["cer"], "cer_avec": out["filtre_geometrique"]["cer"], "adopte": adopt},
                   ["results/layout/filtre_illustration_val.json"], time.time() - t0)
    print("Filtre adopté :", adopt)


if __name__ == "__main__":
    main()
