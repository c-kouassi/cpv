"""Métriques : Levenshtein, CER/WER, bootstrap, McNemar, IoU, calibration."""
import random

import numpy as np
import pytest

from src import metrics


@pytest.mark.parametrize("a,b,d", [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3),
                                   ("abc", "abc", 0), ("quatorre", "quatorze", 1)])
def test_levenshtein_cas_connus(a, b, d):
    assert metrics.levenshtein(a, b) == d


def test_cer_est_levenshtein_sur_longueur_reference():
    assert metrics.cer("abcd", "ab") == 0.5
    assert metrics.cer("ab", "abcd") == 1.0          # non symétrique
    assert metrics.cer("ab", "abxxxxxx") == 3.0      # peut dépasser 1


def test_alignement_coherent_avec_la_distance():
    random.seed(0)
    for _ in range(200):
        a = "".join(random.choices("abe ̃", k=random.randint(0, 12)))
        b = "".join(random.choices("abe ̃", k=random.randint(0, 12)))
        c = metrics.edit_counts(a, b)
        assert c.distance == metrics.levenshtein(a, b) and c.ref_length == len(a)


def test_cer_identique_a_editdistance():
    editdistance = pytest.importorskip("editdistance")
    random.seed(1)
    for _ in range(100):
        a = "".join(random.choices("abcdeſ⁊", k=random.randint(1, 30)))
        b = "".join(random.choices("abcdeſ⁊", k=random.randint(0, 30)))
        assert metrics.levenshtein(a, b) == editdistance.eval(a, b)


def test_cer_corpus_pondere_par_longueur():
    pairs = [("a", "b"), ("abcdefghij", "abcdefghij")]
    assert metrics.corpus_counts(pairs).error_rate == pytest.approx(1 / 11)
    d, n = metrics.per_line_counts(pairs)
    assert d.sum() / n.sum() == pytest.approx(1 / 11)


def test_nfc_avant_comparaison():
    assert metrics.cer("ũg", "ũg") == 0.0


def test_wer():
    assert metrics.wer("le roy nostre sire", "le roy nosre sire") == 0.25


def test_bootstrap_encadre_le_cer_et_est_reproductible():
    d = np.array([0] * 80 + [3] * 20)
    n = np.array([30] * 100)
    lo, hi = metrics.bootstrap_ci(d, n, 1000, 0.05, 42)
    assert lo < d.sum() / n.sum() < hi
    assert (lo, hi) == metrics.bootstrap_ci(d, n, 1000, 0.05, 42)


def test_mcnemar():
    a = np.array([1] * 30 + [0] * 10, bool)
    b = np.array([1] * 20 + [0] * 20, bool)
    r = metrics.mcnemar(a, b)
    assert (r["b"], r["c"]) == (10, 0)
    assert r["p_value"] == pytest.approx(2 / 2 ** 10)
    assert metrics.mcnemar(a, a)["p_value"] == 1.0
    with pytest.raises(ValueError):
        metrics.mcnemar(a, b[:5])


def test_iou_polygones():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert metrics.polygon_iou(sq, sq) == pytest.approx(1.0)
    assert metrics.polygon_iou(sq, [[20, 20], [30, 20], [30, 30]]) == 0.0
    half = [[5, 0], [15, 0], [15, 10], [5, 10]]
    assert 0.3 < metrics.polygon_iou(sq, half) < 0.4


def test_appariement_polygones():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    far = [[50, 50], [60, 50], [60, 60], [50, 60]]
    r = metrics.match_polygons([sq], [sq, far])
    assert r["recall"] == 0.5 and r["precision"] == 1.0 and r["ious"][1] == 0.0


def test_calibration_isotone_croissante_et_ece():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < x ** 3).astype(float)       # modèle surconfiant
    model = metrics.isotonic_fit(x, y)
    pred = metrics.isotonic_predict(model, np.sort(x))
    assert np.all(np.diff(pred) >= -1e-12)
    ece_raw, _ = metrics.expected_calibration_error(x, y)
    ece_cal, _ = metrics.expected_calibration_error(metrics.isotonic_predict(model, x), y)
    assert ece_cal < ece_raw


def test_confusions():
    assert metrics.confusions([("ſus", "fus")])[0] == (("ſ", "f"), 1)
