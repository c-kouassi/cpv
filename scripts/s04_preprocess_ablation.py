"""Étape 4 — Ablation du prétraitement, sur la VALIDATION.

Chaque variante (`preprocess.VARIANTS`) est évaluée de bout en bout :
prétraitement → segmentation → reconnaissance → appariement aux lignes de
référence. On retient la variante au CER le plus bas ; l'intervalle bootstrap
dit si l'écart avec la suivante est significatif.

Usage :
    python -m scripts.s04_preprocess_ablation [--pages 12]
"""
from __future__ import annotations

import argparse
import json
import time

import config
from src import evaluation, htr, preprocess
from src.repro import fixer_seeds, log_experiment


def main() -> None:
    """Évalue chaque variante de prétraitement et enregistre le classement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=12)
    parser.add_argument("--accelerator", default="cpu")
    args = parser.parse_args()
    fixer_seeds(config.SEED)
    pages = evaluation.load_split_pages(config.CORPUS / "val")
    # Échantillon réparti sur les livres de validation (pas les premières pages d'un seul livre).
    by_book = {}
    for p in pages:
        by_book.setdefault(p.book, []).append(p)
    per_book = max(1, args.pages // len(by_book))
    sample = [p for book_pages in by_book.values() for p in book_pages[:: max(1, len(book_pages) // per_book)][:per_book]]
    base = json.loads((config.RESULTS / "baseline" / "classement_val.json").read_text())["meilleur"]
    seg_choice = json.loads((config.RESULTS / "layout" / "segmenteurs_val.json").read_text())
    model = htr.load_model(config.MODELS / "kraken" / f"{base}.mlmodel")
    segmenter = htr.load_segmenter(seg_choice["chemin"])
    table = {}
    for name, steps in preprocess.VARIANTS.items():
        t0 = time.time()
        res = evaluation.run_end_to_end(sample, steps, config.PREPROCESS, model, segmenter,
                                        args.accelerator, config.IOU_THRESHOLD)
        table[name] = res.summary(config.BOOTSTRAP_N, config.ALPHA, config.SEED, config.IOU_THRESHOLD)
        log_experiment(config.EXPERIMENTS / "journal.jsonl", f"ablation_pretraitement_{name}",
                       {"etapes": steps, "params": config.PREPROCESS, "reconnaisseur": base,
                        "segmenteur": seg_choice["meilleur"], "pages_val": [f"{p.book}/{p.name}" for p in sample]},
                       table[name], [], time.time() - t0)
        print(f"{name:16s} CER {table[name]['cer']:.4f} IC95 {[round(x, 4) for x in table[name]['cer_ic95']]} "
              f"IoU lignes {table[name]['iou_lignes_moyenne']:.3f}", flush=True)
    best = min(table, key=lambda k: table[k]["cer"])
    (config.RESULTS / "preprocess").mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "preprocess" / "ablation_val.json").write_text(json.dumps(
        {"meilleure": best, "pages": [f"{p.book}/{p.name}" for p in sample], "variantes": table},
        indent=1, ensure_ascii=False))
    print("Variante retenue :", best)


if __name__ == "__main__":
    main()
