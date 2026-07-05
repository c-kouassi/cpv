"""Étape 6 — Sélection finale sur la validation, calibration, puis évaluation UNIQUE sur le test scellé.

1. Validation : choix du prétraitement à l'inférence pour le modèle affiné
   (brut ou meilleure variante de l'étape 4), calibration isotone des confiances.
2. Test scellé (une seule exécution, verrouillée par `results/test/EVALUE`) :
   - CER / WER du modèle de base et du modèle affiné sur les lignes de référence,
     intervalles bootstrap (N = 1000), test de McNemar ;
   - chaîne complète de bout en bout (prétraitement + segmentation + HTR) :
     CER / WER, IoU des lignes et des zones, taux needs_review ;
   - calibration (ECE, diagramme de fiabilité) et analyse des erreurs.

Usage :
    python -m scripts.s06_evaluate_test
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
from PIL import Image

import config
from src import evaluation, htr, metrics
from src.corpus import group_by_page, read_manifest, split_digest
from src.repro import fixer_seeds, log_experiment

OUT = config.RESULTS / "test"


def reference_lines(model, lines_by_page) -> list[dict]:
    """Transcrit les lignes de référence d'un ensemble de pages.

    Args:
        model: Reconnaisseur.
        lines_by_page: (livre, page) → lignes.

    Returns:
        Résultats ligne par ligne, avec le livre.

    Example:
        >>> reference_lines(model, group_by_page(lines))
    """
    out = []
    for (book, _), lines in lines_by_page.items():
        im = Image.open(config.DATA / lines[0].image)
        out += [r | {"book": book} for r in htr.recognize_reference_lines(im, lines, model)]
    return out


def describe(res: list[dict]) -> dict:
    """CER, WER et intervalle bootstrap de résultats ligne par ligne.

    Args:
        res: Résultats (clés « ref », « hyp »).

    Returns:
        Métriques.

    Example:
        >>> describe([{"ref": "a", "hyp": "a"}])["cer"]
        0.0
    """
    pairs = [(r["ref"], r["hyp"]) for r in res]
    d, n = metrics.per_line_counts(pairs)
    dw, nw = metrics.per_line_counts(pairs, unit="word")
    lo, hi = metrics.bootstrap_ci(d, n, config.BOOTSTRAP_N, config.ALPHA, config.SEED)
    wlo, whi = metrics.bootstrap_ci(dw, nw, config.BOOTSTRAP_N, config.ALPHA, config.SEED)
    return {"lignes": len(pairs), "caracteres": int(n.sum()), "cer": float(d.sum() / n.sum()),
            "cer_ic95": [lo, hi], "wer": float(dw.sum() / nw.sum()), "wer_ic95": [wlo, whi]}


def check_sealed_test() -> str:
    """Vérifie que la partition de test n'a pas changé depuis son scellement.

    Returns:
        L'empreinte SHA-256 du test.

    Raises:
        RuntimeError: Si l'empreinte ne correspond pas à `data/corpus/TEST_SCELLE.sha256`.

    Example:
        >>> check_sealed_test()
    """
    from src.corpus import read_alto
    xmls = sorted((config.CORPUS / "test").glob("*.xml"))
    files = xmls + [read_alto(x, "corpus", x.name.split("__")[0]).image for x in xmls]
    digest, _ = split_digest(files, config.DATA)
    sealed = (config.CORPUS / "TEST_SCELLE.sha256").read_text().strip()
    if digest != sealed:
        raise RuntimeError("Le test a changé depuis son scellement : évaluation refusée.")
    return digest


def main() -> None:
    """Sélection sur la validation puis évaluation unique sur le test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="réévaluer malgré le verrou (à justifier)")
    args = parser.parse_args()
    fixer_seeds(config.SEED)
    lock = OUT / "EVALUE"
    if lock.exists() and not args.force:
        raise SystemExit(f"Le test a déjà été évalué ({lock.read_text().strip()}). Voir results/test/.")
    t0 = time.time()
    base_name = json.loads((config.RESULTS / "baseline" / "classement_val.json").read_text())["meilleur"]
    seg_choice = json.loads((config.RESULTS / "layout" / "segmenteurs_val.json").read_text())
    ablation = json.loads((config.RESULTS / "preprocess" / "ablation_val.json").read_text())
    base = htr.load_model(config.MODELS / "kraken" / f"{base_name}.mlmodel")
    ft = htr.load_model(config.MODELS / "kraken" / "charles8_htr_ft.mlmodel")
    segmenter = htr.load_segmenter(seg_choice["chemin"])
    from src.preprocess import VARIANTS

    # ── 1. Validation : prétraitement d'inférence et calibration ─────────────
    val_pages = evaluation.load_split_pages(config.CORPUS / "val")
    sample = [p for p in val_pages if f"{p.book}/{p.name}" in set(ablation["pages"])]
    choix = {}
    for variant in sorted({"brut", ablation["meilleure"]}):
        res = evaluation.run_end_to_end(sample, VARIANTS[variant], config.PREPROCESS, ft, segmenter,
                                        iou_threshold=config.IOU_THRESHOLD)
        choix[variant] = res.summary(config.BOOTSTRAP_N, config.ALPHA, config.SEED)["cer"]
    variant = min(choix, key=choix.get)

    val_ref = reference_lines(ft, group_by_page(read_manifest(config.CORPUS / "lines_val.jsonl")))
    conf_val = np.array([r["confidence"] for r in val_ref])
    ok_val = np.array([metrics.cer(r["ref"], r["hyp"]) <= config.CALIBRATION_TARGET_CER for r in val_ref], float)
    calibrator = metrics.isotonic_fit(conf_val, ok_val)
    ece_val_raw, _ = metrics.expected_calibration_error(conf_val, ok_val)
    ece_val_cal, _ = metrics.expected_calibration_error(metrics.isotonic_predict(calibrator, conf_val), ok_val)
    calib_dir = config.MODELS / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    (calib_dir / "isotone_val.json").write_text(json.dumps(
        {"cible": f"P(CER ligne <= {config.CALIBRATION_TARGET_CER})", "x": calibrator[0].tolist(),
         "y": calibrator[1].tolist(), "ajuste_sur": "validation (lignes de référence, modèle affiné)"}))
    selection = {"pretraitement_inference": variant, "cer_val_par_variante": choix,
                 "ece_val_brute": ece_val_raw, "ece_val_calibree": ece_val_cal}
    print("Sélection sur la validation :", selection, flush=True)

    # ── 2. Test scellé ───────────────────────────────────────────────────────
    digest = check_sealed_test()
    test_lines = group_by_page(read_manifest(config.CORPUS / "lines_test.jsonl"))
    OUT.mkdir(parents=True, exist_ok=True)
    res_base = reference_lines(base, test_lines)
    res_ft = reference_lines(ft, test_lines)
    for name, res in [("base", res_base), ("affine", res_ft)]:
        (OUT / f"lignes_{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n")
    correct_b = np.array([metrics.cer(r["ref"], r["hyp"]) <= config.LINE_CORRECT_CER for r in res_base])
    correct_f = np.array([metrics.cer(r["ref"], r["hyp"]) <= config.LINE_CORRECT_CER for r in res_ft])
    reference = {"base": describe(res_base) | {"modele": base_name},
                 "affine": describe(res_ft) | {"modele": "charles8_htr_ft"},
                 "mcnemar": metrics.mcnemar(correct_b, correct_f) | {
                     "critere": f"ligne correcte si CER <= {config.LINE_CORRECT_CER}",
                     "lignes_correctes_base": int(correct_b.sum()), "lignes_correctes_affine": int(correct_f.sum())}}
    by_book = defaultdict(list)
    for r in res_ft:
        by_book[r["book"]].append(r)
    reference["affine_par_livre"] = {b: describe(v) for b, v in sorted(by_book.items())}
    by_len = defaultdict(list)
    for r in res_ft:
        n = len(r["ref"])
        by_len["<10" if n < 10 else "10-29" if n < 30 else "30-49" if n < 50 else ">=50"].append(r)
    reference["affine_par_longueur"] = {k: describe(v) for k, v in by_len.items()}
    reference["confusions_affine"] = [[a, b, k] for (a, b), k in metrics.confusions([(r["ref"], r["hyp"]) for r in res_ft], 25)]

    conf_t = np.array([r["confidence"] for r in res_ft])
    ok_t = np.array([metrics.cer(r["ref"], r["hyp"]) <= config.CALIBRATION_TARGET_CER for r in res_ft], float)
    cal_t = metrics.isotonic_predict(calibrator, conf_t)
    ece_raw, rel_raw = metrics.expected_calibration_error(conf_t, ok_t)
    ece_cal, rel_cal = metrics.expected_calibration_error(cal_t, ok_t)
    review = (cal_t < config.REVIEW_PROBABILITY) | np.array([len(r["hyp"].strip()) < config.SHORT_LINE_CHARS for r in res_ft])
    calibration = {"ece_brute": ece_raw, "ece_calibree": ece_cal, "fiabilite_brute": rel_raw,
                   "fiabilite_calibree": rel_cal, "taux_needs_review": float(review.mean()),
                   "cer_lignes_a_relire": describe([r for r, f in zip(res_ft, review) if f])["cer"] if review.any() else None,
                   "cer_lignes_non_relues": describe([r for r, f in zip(res_ft, review) if not f])["cer"] if (~review).any() else None}

    test_pages = evaluation.load_split_pages(config.CORPUS / "test")
    e2e = evaluation.run_end_to_end(test_pages, VARIANTS[variant], config.PREPROCESS, ft, segmenter,
                                    iou_threshold=config.IOU_THRESHOLD)
    bout_en_bout = e2e.summary(config.BOOTSTRAP_N, config.ALPHA, config.SEED, config.IOU_THRESHOLD)

    results = {"test_sha256": digest, "selection_validation": selection, "lignes_reference": reference,
               "bout_en_bout": bout_en_bout, "calibration": calibration,
               "seuils": {"cer_validation": config.CER_VALIDATION_THRESHOLD,
                          "cer_excellence": config.CER_EXCELLENCE_THRESHOLD}}
    (OUT / "resultats.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    _figures(rel_raw, rel_cal, reference, ece_raw, ece_cal)
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log_experiment(config.EXPERIMENTS / "journal.jsonl", "evaluation_test_scelle",
                   {"base": base_name, "affine": "charles8_htr_ft", "segmenteur": seg_choice["meilleur"],
                    "pretraitement": variant, "bootstrap_n": config.BOOTSTRAP_N},
                   {"cer_base": reference["base"]["cer"], "cer_affine": reference["affine"]["cer"],
                    "mcnemar_p": reference["mcnemar"]["p_value"], "cer_bout_en_bout": bout_en_bout["cer"],
                    "iou_lignes": bout_en_bout["iou_lignes_moyenne"], "needs_review": calibration["taux_needs_review"]},
                   ["results/test/resultats.json"], time.time() - t0)
    print(json.dumps({k: results[k] for k in ["selection_validation", "bout_en_bout"]}, indent=1, ensure_ascii=False))
    print("base", reference["base"]["cer"], "affiné", reference["affine"]["cer"], "McNemar", reference["mcnemar"])


def _figures(rel_raw, rel_cal, reference, ece_raw, ece_cal) -> None:
    import matplotlib.pyplot as plt
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for rel, label, ece in [(rel_raw, "confiance brute", ece_raw), (rel_cal, "confiance calibrée", ece_cal)]:
        a1.plot([b["confiance"] for b in rel], [b["exactitude"] for b in rel], marker="o", label=f"{label} (ECE {ece:.3f})")
    a1.plot([0, 1], [0, 1], ls="--", c="gray")
    a1.set_xlabel("confiance annoncée"); a1.set_ylabel(f"part des lignes à CER ≤ {config.CALIBRATION_TARGET_CER:.0%}")
    a1.set_title("Diagramme de fiabilité (test)"); a1.legend()
    names = ["base", "affine"]
    cers = [reference[n]["cer"] for n in names]
    err = np.array([[reference[n]["cer"] - reference[n]["cer_ic95"][0], reference[n]["cer_ic95"][1] - reference[n]["cer"]] for n in names]).T
    a2.bar(["sans fine-tuning", "affiné"], cers, yerr=err, capsize=6, color=["#9aa7b8", "#4a78b5"])
    a2.axhline(config.CER_VALIDATION_THRESHOLD, ls="--", c="#c4473a", label="seuil de validation 15 %")
    a2.axhline(config.CER_EXCELLENCE_THRESHOLD, ls=":", c="#2f8f5b", label="seuil d'excellence 8 %")
    a2.set_ylabel("CER (test, IC 95 %)"); a2.set_title("CER sur le test scellé"); a2.legend()
    plt.tight_layout(); plt.savefig(config.FIGURES / "test_cer_calibration.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
