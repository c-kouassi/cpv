"""Étape 8 — Annotations humaines : accord inter-annotateurs (IAA) et vérité terrain du document cible.

Deux campagnes, chacune en deux temps (`prepare` puis `compute`) :

- **iaa** : 50 lignes tirées du test scellé (après l'évaluation unique de
  l'étape 6). Deux membres du groupe les transcrivent INDÉPENDAMMENT, à partir
  de l'image seule, sans voir ni la vérité terrain ni la sortie du modèle.
  CER(A, B) mesure l'ambiguïté irréductible : c'est le plafond auquel comparer
  le CER du modèle sur les mêmes lignes.
- **cible** : les lignes détectées sur les pages 13-20 de Charles VIII.
  Transcription humaine → CER du modèle dans le domaine visé (hors corpus
  d'entraînement).

Usage :
    python -m scripts.s08_annotations prepare iaa
    (les annotateurs remplissent annotations/iaa/A/*.txt et annotations/iaa/B/*.txt)
    python -m scripts.s08_annotations compute iaa
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from PIL import Image

import config
from src import metrics
from src.corpus import read_manifest

ROOT = config.ROOT / "annotations"
PAD = 12


def crop_line(image: Image.Image, polygon, pad: int = PAD) -> Image.Image:
    """Découpe le rectangle englobant d'une ligne, avec une marge.

    Args:
        image: Image de la page.
        polygon: Contour de la ligne (x, y).
        pad: Marge en pixels.

    Returns:
        Image de la ligne.

    Example:
        >>> crop_line(Image.new("L", (100, 100)), [[10, 10], [50, 10], [50, 30]]).size
        (62, 42)
    """
    a = np.asarray(polygon)
    x0, y0 = max(int(a[:, 0].min()) - pad, 0), max(int(a[:, 1].min()) - pad, 0)
    x1, y1 = min(int(a[:, 0].max()) + pad, image.width), min(int(a[:, 1].max()) + pad, image.height)
    return image.crop((x0, y0, x1, y1))


def prepare(campaign: str, n: int) -> None:
    """Exporte les images de lignes et des fichiers de transcription vides pour deux annotateurs.

    Args:
        campaign: « iaa » ou « cible ».
        n: Nombre de lignes (campagne « iaa »).

    Raises:
        SystemExit: Si le test n'a pas encore été évalué (campagne « iaa »).

    Example:
        >>> prepare("iaa", 50)
    """
    out = ROOT / campaign
    items = []
    if campaign == "iaa":
        if not (config.RESULTS / "test" / "EVALUE").exists():
            raise SystemExit("Évaluer le test (étape 6) avant de préparer l'IAA.")
        lines = [l for l in read_manifest(config.CORPUS / "lines_test.jsonl") if len(l.text) >= 10]
        sample = random.Random(config.SEED).sample(lines, n)
        for k, l in enumerate(sample):
            items.append({"id": f"iaa_{k:03d}", "image": config.DATA / l.image, "polygon": l.polygon,
                          "book": l.book, "page": l.page, "line_id": l.line_id})
    else:
        data = json.loads((config.DATASET_NLP / "charles_viii_1494.json").read_text())
        for p in data["pages"]:
            for l in p["lines"]:
                items.append({"id": f"{p['page_id']}_{l['line_id']}", "image": config.ROOT / p["image"],
                              "polygon": l["polygon"], "hyp": l["text"]})
    (out / "lignes").mkdir(parents=True, exist_ok=True)
    for who in ["A", "B"]:
        (out / who).mkdir(exist_ok=True)
    cache = {}
    for it in items:
        im = cache.setdefault(str(it["image"]), Image.open(it["image"]))
        crop_line(im, it["polygon"]).save(out / "lignes" / f"{it['id']}.png")
        for who in ["A", "B"]:
            f = out / who / f"{it['id']}.txt"
            if not f.exists():
                f.write_text("", encoding="utf-8")
    meta = [{k: (str(v.relative_to(config.ROOT)) if hasattr(v, "relative_to") else v)
             for k, v in it.items() if k != "polygon"} for it in items]
    (out / "lignes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"{len(items)} lignes exportées dans {out.relative_to(config.ROOT)}/lignes ; "
          f"transcrire dans {campaign}/A et {campaign}/B (voir CONVENTIONS_TRANSCRIPTION.md).")


def compute(campaign: str) -> dict:
    """Calcule l'accord inter-annotateurs et compare le modèle aux annotateurs.

    Args:
        campaign: « iaa » ou « cible ».

    Returns:
        Métriques (CER entre annotateurs, CER du modèle, IC bootstrap).

    Raises:
        SystemExit: Si des transcriptions manquent.

    Example:
        >>> compute("iaa")["cer_A_B"]
    """
    out = ROOT / campaign
    meta = json.loads((out / "lignes.json").read_text())
    read = lambda who, i: (out / who / f"{i}.txt").read_text(encoding="utf-8").strip()
    missing = [m["id"] for m in meta for who in "AB" if not read(who, m["id"])]
    if missing:
        raise SystemExit(f"{len(missing)} transcriptions vides (ex. {missing[:3]}).")
    a = [read("A", m["id"]) for m in meta]
    b = [read("B", m["id"]) for m in meta]

    def _stat(pairs):
        d, n = metrics.per_line_counts(pairs)
        lo, hi = metrics.bootstrap_ci(d, n, config.BOOTSTRAP_N, config.ALPHA, config.SEED)
        return {"cer": float(d.sum() / n.sum()), "ic95": [lo, hi]}

    res = {"lignes": len(meta), "cer_A_B": _stat(list(zip(a, b))), "cer_B_A": _stat(list(zip(b, a)))}
    if campaign == "iaa":
        model = {json.loads(l)["line_id"] + "|" + json.loads(l)["book"]: json.loads(l)["hyp"]
                 for l in (config.RESULTS / "test" / "lignes_affine.jsonl").read_text().splitlines() if l.strip()}
        hyp = [model[f"{m['line_id']}|{m['book']}"] for m in meta]
        # La vérité terrain n'est pas dans lignes.json (les annotateurs ne doivent pas la voir).
        gt = {(l.book, l.page, l.line_id): l.text for l in read_manifest(config.CORPUS / "lines_test.jsonl")}
        ref = [gt[(m["book"], m["page"], m["line_id"])] for m in meta]
        res |= {"cer_A_reference": _stat(list(zip(ref, a))), "cer_B_reference": _stat(list(zip(ref, b))),
                "cer_modele_reference": _stat(list(zip(ref, hyp)))}
        res["modele_sous_iaa"] = res["cer_modele_reference"]["cer"] < res["cer_A_B"]["cer"]
    else:
        hyp = [m["hyp"] for m in meta]
        res |= {"cer_modele_vs_A": _stat(list(zip(a, hyp))), "cer_modele_vs_B": _stat(list(zip(b, hyp)))}
    (config.RESULTS / "annotations").mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "annotations" / f"{campaign}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "compute"])
    parser.add_argument("campaign", choices=["iaa", "cible"])
    parser.add_argument("-n", type=int, default=50)
    args = parser.parse_args()
    prepare(args.campaign, args.n) if args.action == "prepare" else compute(args.campaign)


if __name__ == "__main__":
    main()
