"""Non-régression du CER sur un sous-ensemble de référence fixe.

Le sous-ensemble (`tests/fixtures/reference_subset.json`) est tiré de la
VALIDATION, jamais du test scellé. Le modèle affiné doit y rester sous le seuil
de validation du sujet (CER < 15 %). Le test est sauté si les données ou le
modèle ne sont pas présents (voir README, « Reproduire les résultats »).
"""
import json
from pathlib import Path

import pytest

import config

FIXTURE = Path(__file__).parent / "fixtures" / "reference_subset.json"
MODEL = config.MODELS / "kraken" / "charles8_htr_ft.mlmodel"

pytestmark = pytest.mark.skipif(
    not (FIXTURE.exists() and MODEL.exists() and (config.CORPUS / "lines_val.jsonl").exists()),
    reason="modèle affiné ou corpus absent")


def test_cer_sous_le_seuil_de_validation():
    pytest.importorskip("kraken")
    from PIL import Image
    from src import htr, metrics
    from src.corpus import group_by_page, read_manifest
    subset = json.loads(FIXTURE.read_text())
    wanted = {(x["book"], x["page"], x["line_id"]) for x in subset["lignes"]}
    lines = [l for l in read_manifest(config.CORPUS / "lines_val.jsonl") if (l.book, l.page, l.line_id) in wanted]
    assert len(lines) == len(wanted), "sous-ensemble de référence introuvable dans la validation"
    model = htr.load_model(MODEL)
    pairs = []
    for (_, _), page_lines in group_by_page(lines).items():
        res = htr.recognize_reference_lines(Image.open(config.DATA / page_lines[0].image), page_lines, model)
        pairs += [(r["ref"], r["hyp"]) for r in res]
    cer = metrics.corpus_counts(pairs).error_rate
    assert cer < config.CER_VALIDATION_THRESHOLD, f"CER {cer:.3f} ≥ {config.CER_VALIDATION_THRESHOLD}"
    assert cer <= subset["cer_reference"] + 0.01, f"régression : CER {cer:.4f} > {subset['cer_reference']:.4f} + 1 pt"
