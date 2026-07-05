"""Prétraitement : formes, types, plages de valeurs, redressement."""
import cv2
import numpy as np
import pytest

import config
from src import preprocess


@pytest.fixture
def page():
    """Page synthétique : lignes de « texte », tache sombre, bruit gaussien (BGR uint8)."""
    rng = np.random.default_rng(0)
    img = np.full((600, 800), 220, np.uint8)
    for k in range(8):
        cv2.putText(img, "noſtre sire le roy", (40, 80 + k * 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 3)
    yy, xx = np.mgrid[:600, :800]
    img = cv2.subtract(img, (60 * np.exp(-((xx - 600) ** 2 + (yy - 300) ** 2) / (2 * 120 ** 2))).astype(np.uint8))
    noisy = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("name", list(preprocess.VARIANTS))
def test_variantes_formes_types_plages(page, name):
    out = preprocess.run(page, preprocess.VARIANTS[name], config.PREPROCESS)
    assert out.image.dtype == np.uint8
    assert out.image.shape[:2] == page.shape[:2]
    assert out.image.ndim == (3 if name == "brut" else 2)
    assert 0 <= out.image.min() and out.image.max() <= 255
    assert out.matrix.shape == (2, 3)


def test_sauvola_binaire(page):
    out = preprocess.run(page, preprocess.VARIANTS["sauvola"], config.PREPROCESS).image
    assert preprocess.is_binary(out)


def test_clahe_augmente_le_contraste_local():
    gray = np.tile(np.linspace(100, 140, 256, dtype=np.uint8), (256, 1))
    assert preprocess.clahe(gray).std() > gray.std()


def test_debruitage_rapproche_de_l_image_propre():
    rng = np.random.default_rng(1)
    clean = np.full((128, 128), 200, np.uint8)
    cv2.putText(clean, "abc", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 30, 4)
    noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255).astype(np.uint8)
    err = lambda a: np.abs(a.astype(int) - clean.astype(int)).mean()
    assert err(preprocess.denoise(noisy)) < err(noisy)


def test_fond_aplati(page):
    gray = preprocess.to_gray(page)
    flat = preprocess.normalize_background(preprocess.denoise(gray), 51)
    # fond près de la tache (x≈600) vs fond loin de la tache, entre deux lignes
    assert abs(int(flat[555:590, 560:640].mean()) - int(flat[555:590, 40:120].mean())) < \
        abs(int(gray[555:590, 560:640].mean()) - int(gray[555:590, 40:120].mean()))


@pytest.mark.parametrize("angle", [-2.0, 1.4, 3.0])
def test_redressement_retrouve_l_angle(page, angle):
    h, w = page.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    tilted = cv2.warpAffine(page, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    assert preprocess.estimate_skew(preprocess.to_gray(tilted)) == pytest.approx(-angle, abs=0.4)


def test_inversion_des_points_redresses(page):
    h, w = page.shape[:2]
    tilted = cv2.warpAffine(page, cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0), (w, h),
                            borderMode=cv2.BORDER_REPLICATE)
    out = preprocess.deskew(tilted)
    pts = [[100, 100], [700, 500]]
    moved = (np.asarray(pts) @ out.matrix[:, :2].T + out.matrix[:, 2]).tolist()
    assert preprocess.invert_points(moved, out.matrix) == pts


def test_etape_inconnue(page):
    with pytest.raises(ValueError):
        preprocess.run(page, ["flou"], {})
