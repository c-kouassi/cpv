"""Étape 2 — Ligne de base sans fine-tuning : CER de chaque modèle candidat sur la VALIDATION.

Reconnaissance sur les lignes de référence (géométrie de la vérité terrain) :
l'écart mesuré ne vient que du modèle de reconnaissance. Le test n'est pas utilisé.

Usage :
    python -m scripts.s02_baseline_models [--device mps]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from PIL import Image

import config
from src import htr, metrics
from src.corpus import group_by_page, read_manifest
from src.repro import fixer_seeds, log_experiment


def evaluate_model(model, lines_by_page: dict) -> list[dict]:
    """Transcrit toutes les lignes de référence d'un ensemble de pages.

    Args:
        model: Reconnaisseur Kraken.
        lines_by_page: (livre, page) → lignes de référence.

    Returns:
        Résultats ligne par ligne (voir `htr.recognize_reference_lines`), avec le livre.

    Example:
        >>> res = evaluate_model(model, group_by_page(read_manifest(path)))
    """
    out = []
    for (book, _), lines in lines_by_page.items():
        image = Image.open(config.DATA / lines[0].image)
        for r in htr.recognize_reference_lines(image, lines, model):
            out.append(r | {"book": book})
    return out


def summarize(res: list[dict], seed: int) -> dict:
    """CER, WER et intervalle bootstrap d'un ensemble de lignes transcrites.

    Args:
        res: Résultats ligne par ligne.
        seed: Graine du bootstrap.

    Returns:
        Dictionnaire de métriques.

    Example:
        >>> summarize([{"ref": "ab", "hyp": "ab"}], 42)["cer"]
        0.0
    """
    pairs = [(r["ref"], r["hyp"]) for r in res]
    d, n = metrics.per_line_counts(pairs)
    dw, nw = metrics.per_line_counts(pairs, unit="word")
    lo, hi = metrics.bootstrap_ci(d, n, config.BOOTSTRAP_N, config.ALPHA, seed)
    return {"lignes": len(pairs), "cer": float(d.sum() / n.sum()), "cer_ic95": [lo, hi],
            "wer": float(dw.sum() / nw.sum())}


def main() -> None:
    """Évalue chaque modèle candidat sur la validation et enregistre le classement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rank-only", action="store_true", help="reclasser à partir des résultats enregistrés")
    args = parser.parse_args()
    fixer_seeds(config.SEED)
    val = group_by_page(read_manifest(config.CORPUS / "lines_val.jsonl"))
    out_dir = config.RESULTS / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, doi in config.KRAKEN_CANDIDATES.items():
        cached = out_dir / f"val_{name}.jsonl"
        if args.rank_only and cached.exists():
            res = [json.loads(l) for l in cached.read_text().splitlines() if l.strip()]
            table[name] = summarize(res, config.SEED) | {
                "par_livre": {b: summarize([r for r in res if r["book"] == b], config.SEED)["cer"]
                              for b in sorted({r["book"] for r in res})}}
            continue
        t0 = time.time()
        path = htr.fetch_model(doi, config.MODELS / "kraken" / f"{name}.mlmodel")
        res = evaluate_model(htr.load_model(path, args.device), val)
        (out_dir / f"val_{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n")
        table[name] = summarize(res, config.SEED) | {
            "par_livre": {b: summarize([r for r in res if r["book"] == b], config.SEED)["cer"]
                          for b in sorted({r["book"] for r in res})}}
        log_experiment(config.EXPERIMENTS / "journal.jsonl", f"baseline_val_{name}",
                       {"modele": name, "doi": doi, "partition": "val", "geometrie": "reference"},
                       table[name], [f"results/baseline/val_{name}.jsonl"], time.time() - t0)
        print(f"{name:22s} CER {table[name]['cer']:.4f}  IC95 {np.round(table[name]['cer_ic95'], 4)}  "
              f"WER {table[name]['wer']:.4f}", flush=True)
    for name in table:
        table[name]["contamination"] = config.KRAKEN_CONTAMINATION.get(name, "inconnu")
    eligible = [k for k in table if table[k]["contamination"] == "propre"]
    best = min(eligible, key=lambda k: table[k]["cer"])
    (out_dir / "classement_val.json").write_text(json.dumps(
        {"meilleur": best, "regle": "meilleur CER de validation parmi les modèles sans recouvrement documenté "
                                    "avec le corpus (config.KRAKEN_CONTAMINATION)", "modeles": table}, indent=1))
    for k in sorted(table, key=lambda k: table[k]["cer"]):
        print(f"{k:22s} CER {table[k]['cer']:.4f}  {table[k]['contamination']}")
    print("Modèle de base retenu :", best)


if __name__ == "__main__":
    main()
