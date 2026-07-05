"""Étape 3 — Segmentation de mise en page : segmenteur par défaut vs segmenteur affiné, sur la VALIDATION.

Mesures : détection des lignes (précision, rappel, F1, IoU des polygones),
IoU au pixel des zones (texte, illustration, paratexte), et CER de bout en bout
avec un même reconnaisseur, pour voir l'effet de la segmentation sur la transcription.

Usage :
    python -m scripts.s03_layout_val [--pages N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import config
from src import evaluation, htr
from src.repro import fixer_seeds, log_experiment


def best_checkpoint(directory: Path) -> Path | None:
    """Renvoie le meilleur fichier de poids produit par `ketos` (`best_<score>.safetensors`).

    Args:
        directory: Dossier de sortie de l'entraînement.

    Returns:
        Le fichier de meilleur score, ou None s'il n'y en a pas.

    Example:
        >>> best_checkpoint(Path("models/kraken/segmentation"))
    """
    files = sorted(directory.glob("best_*.safetensors"), key=lambda p: float(p.stem.split("_")[1]))
    return files[-1] if files else None


def main() -> None:
    """Compare les segmenteurs sur la validation et retient le meilleur."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=0, help="0 = toutes les pages de validation")
    parser.add_argument("--accelerator", default="cpu")
    args = parser.parse_args()
    fixer_seeds(config.SEED)
    pages = evaluation.load_split_pages(config.CORPUS / "val")
    if args.pages:
        pages = pages[:: max(1, len(pages) // args.pages)][: args.pages]
    base = json.loads((config.RESULTS / "baseline" / "classement_val.json").read_text())["meilleur"]
    model = htr.load_model(config.MODELS / "kraken" / f"{base}.mlmodel")
    seg_ft = best_checkpoint(config.MODELS / "kraken" / "segmentation")
    candidates = {"blla_defaut": None, "blla_affine": seg_ft}
    out = {}
    for name, path in candidates.items():
        if name != "blla_defaut" and path is None:
            print("Pas de segmenteur affiné : lancer `ketos segtrain` (voir README).")
            continue
        t0 = time.time()
        res = evaluation.run_end_to_end(pages, [], config.PREPROCESS, model, htr.load_segmenter(path),
                                        args.accelerator, config.IOU_THRESHOLD)
        out[name] = res.summary(config.BOOTSTRAP_N, config.ALPHA, config.SEED, config.IOU_THRESHOLD)
        log_experiment(config.EXPERIMENTS / "journal.jsonl", f"layout_val_{name}",
                       {"segmenteur": str(path) if path else "kraken blla.mlmodel", "reconnaisseur": base,
                        "pages_val": len(pages), "pretraitement": "brut"}, out[name], [], time.time() - t0)
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
    best = max(out, key=lambda k: (out[k]["part_lignes_iou_sup_seuil"], -out[k]["cer"]))
    (config.RESULTS / "layout").mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "layout" / "segmenteurs_val.json").write_text(json.dumps(
        {"meilleur": best, "chemin": str(candidates[best]) if candidates[best] else None,
         "pages": len(pages), "resultats": out}, indent=1, ensure_ascii=False))
    print("Segmenteur retenu :", best)


if __name__ == "__main__":
    main()
