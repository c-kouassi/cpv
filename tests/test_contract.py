"""Data contract : cohérence avec le schéma JSON, géométrie, needs_review."""
import copy
import json

import numpy as np
import pytest

import config
from src import contract
from src.htr import PredictedLine, PredictedPage

jsonschema = pytest.importorskip("jsonschema")


def _page(text="le roy nostre sire", conf=0.97, poly=None):
    poly = poly or [[10, 10], [500, 10], [500, 60], [10, 60]]
    line = PredictedLine("l001", text, conf, [conf] * len(text), [[10, 50], [500, 50]], poly, "texte")
    return PredictedPage("data/images/page_014.png", 1000, 1500, [line],
                         [{"type": "texte", "polygon": [[0, 0], [900, 0], [900, 900], [0, 900]]}], 0, 0.4)


def _contract(page):
    calibrator = (np.array([0.5, 0.9, 1.0]), np.array([0.1, 0.6, 0.95]))
    entry = contract.page_entry(page, 14, "texte", "segmentations/page_014.xml", "data/images/page_014.png",
                                "0" * 64, calibrator, config.REVIEW_PROBABILITY, config.SHORT_LINE_CHARS, str.upper)
    return contract.build({"id": "x", "title": "t", "source": "s", "licence": "l", "url": "u"}, [entry],
                          {"preprocessing": {}, "segmentation_model": {}, "htr_model": {}, "calibration": {}})


def test_contrat_valide():
    c = _contract(_page())
    contract.validate(c)
    line = c["pages"][0]["lines"][0]
    assert line["text_normalized"] == "LE ROY NOSTRE SIRE"
    assert 0 <= line["confidence"] <= 1 and not line["needs_review"]
    assert c["coordinate_system"]["origin"] == "top-left" and c["coordinate_system"]["unit"] == "pixel"
    assert c["stats"]["needs_review_rate"] == 0.0


def test_contrat_relu_depuis_json():
    contract.validate(json.loads(json.dumps(_contract(_page()), ensure_ascii=False)))


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("coordinate_system"),
    lambda c: c["pages"][0]["lines"][0].update(confidence=1.5),
    lambda c: c["pages"][0]["lines"][0].update(polygon=[[0, 0], [1, 1]]),
    lambda c: c["pages"][0]["lines"][0].update(baseline=[[0, 0, 0]]),
    lambda c: c["pages"][0]["lines"][0].pop("needs_review"),
    lambda c: c["pages"][0]["regions"][0].update(type="gravure"),
])
def test_contrat_invalide_rejete(mutation):
    c = copy.deepcopy(_contract(_page()))
    mutation(c)
    with pytest.raises(jsonschema.ValidationError):
        contract.validate(c)


def test_needs_review_confiance_basse():
    line = _contract(_page(conf=0.55))["pages"][0]["lines"][0]
    assert line["needs_review"] and "low_confidence" in line["review_reasons"]


def test_needs_review_ligne_courte():
    line = _contract(_page(text="et"))["pages"][0]["lines"][0]
    assert "short_line" in line["review_reasons"]


def test_needs_review_polygone_hors_image():
    line = _contract(_page(poly=[[10, 10], [1200, 10], [1200, 60], [10, 60]]))["pages"][0]["lines"][0]
    assert "outside_image" in line["review_reasons"]


def test_toute_ligne_a_un_polygone():
    c = _contract(_page())
    assert all(len(l["polygon"]) >= 3 for p in c["pages"] for l in p["lines"])
