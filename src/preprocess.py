"""Prétraitement des images de pages.

Chaîne (chaque étape est optionnelle, l'ablation choisit la combinaison) :

1. redressement : estimation de l'inclinaison par profil de projection ;
2. niveaux de gris ;
3. débruitage NL-means sur l'image EN GRIS (avant toute binarisation : le
   filtre moyenne des voisinages qui se ressemblent et a besoin de niveaux
   de gris continus) ;
4. normalisation du fond : division par une estimation morphologique du fond
   (taches, auréoles, jaunissement) ;
5. CLAHE : égalisation d'histogramme adaptative par tuiles, contraste borné
   pour ne pas amplifier le bruit ;
6. binarisation de Sauvola : seuil local T = m · (1 + k · (s / R − 1)).

Toutes les sorties sont des tableaux uint8 ; les images sont enregistrées en
PNG (sans perte). Le redressement renvoie la matrice affine utilisée, pour
ramener les polygones détectés dans le repère de l'image d'origine.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Preprocessed:
    """Image prétraitée et transformation géométrique appliquée.

    Attributes:
        image: Image uint8 (2D en gris, 3D en couleur).
        matrix: Matrice affine 2×3 de l'image d'origine vers l'image prétraitée.
        angle: Angle de redressement appliqué, en degrés.
    """
    image: np.ndarray
    matrix: np.ndarray
    angle: float = 0.0


IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convertit en niveaux de gris.

    Args:
        img: Image BGR (H, W, 3) ou déjà en gris (H, W), uint8.

    Returns:
        Image (H, W) uint8.

    Example:
        >>> to_gray(np.zeros((4, 4, 3), np.uint8)).shape
        (4, 4)
    """
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0, step: float = 0.2) -> float:
    """Estime l'inclinaison des lignes de texte par maximisation de la variance du profil horizontal.

    Quand les lignes sont horizontales, la quantité d'encre par rangée alterne
    nettement entre lignes et interlignes : la variance du profil est maximale.
    On teste les angles de −max à +max sur une image réduite.

    Args:
        gray: Image en niveaux de gris.
        max_angle: Angle maximal testé, en degrés.
        step: Pas de recherche, en degrés.

    Returns:
        Angle (degrés) à appliquer pour redresser ; positif = rotation antihoraire.

    Example:
        >>> estimate_skew(np.full((100, 100), 255, np.uint8))
        0.0
    """
    scale = min(1000 / max(gray.shape), 1)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if ink.mean() < 1:
        return 0.0
    h, w = ink.shape
    best, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
        rot = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score * (1 + 1e-9):
            best, best_score = float(angle), score
    return round(best, 2) + 0.0


def deskew(img: np.ndarray, max_angle: float = 5.0) -> Preprocessed:
    """Redresse une page.

    Args:
        img: Image (gris ou couleur).
        max_angle: Angle maximal recherché, en degrés.

    Returns:
        Image redressée (même taille, bords répliqués) et matrice affine appliquée.

    Example:
        >>> deskew(np.full((50, 50), 255, np.uint8)).angle
        0.0
    """
    angle = estimate_skew(to_gray(img), max_angle)
    if angle == 0:
        return Preprocessed(img, IDENTITY.copy(), 0.0)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Preprocessed(out, m, angle)


def invert_points(points, matrix: np.ndarray) -> list[list[int]]:
    """Ramène des points de l'image prétraitée dans le repère de l'image d'origine.

    Args:
        points: Liste de (x, y) dans l'image prétraitée.
        matrix: Matrice affine 2×3 origine → prétraitée.

    Returns:
        Points (x, y) entiers dans l'image d'origine.

    Example:
        >>> invert_points([[1, 2]], IDENTITY)
        [[1, 2]]
    """
    if not len(points):
        return []
    inv = cv2.invertAffineTransform(np.asarray(matrix, np.float64))
    pts = np.asarray(points, np.float64)
    return np.rint(pts @ inv[:, :2].T + inv[:, 2]).astype(int).tolist()


def denoise(gray: np.ndarray, h: float = 7) -> np.ndarray:
    """Débruitage NL-means (à appliquer sur l'image en gris, avant binarisation).

    Args:
        gray: Image en niveaux de gris.
        h: Force du filtrage (plus grand = plus lisse, mais traits plus flous).

    Returns:
        Image débruitée, même forme et type.

    Example:
        >>> denoise(np.zeros((32, 32), np.uint8)).dtype
        dtype('uint8')
    """
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def normalize_background(gray: np.ndarray, kernel: int = 51) -> np.ndarray:
    """Retire les variations lentes du fond (texte sombre sur fond clair).

    Le fond est estimé par une fermeture morphologique avec un élément plus
    large que les traits : les lettres disparaissent, les taches restent. On
    divise ensuite l'image par ce fond.

    Args:
        gray: Image en niveaux de gris.
        kernel: Taille de l'élément structurant (pixels).

    Returns:
        Image à fond aplati, uint8.

    Example:
        >>> int(normalize_background(np.full((64, 64), 200, np.uint8)).max())
        255
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    out = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Égalisation d'histogramme adaptative à contraste limité (CLAHE).

    Args:
        gray: Image en niveaux de gris.
        clip: Limite de contraste (évite d'amplifier le bruit des zones uniformes).
        tile: Nombre de tuiles par côté.

    Returns:
        Image égalisée, uint8.

    Example:
        >>> clahe(np.zeros((32, 32), np.uint8)).shape
        (32, 32)
    """
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, r: float = 128) -> np.ndarray:
    """Binarisation de Sauvola (seuil local).

    T(x, y) = m · (1 + k · (s / R − 1)), m et s étant la moyenne et l'écart-type
    dans une fenêtre centrée. Dans les zones sombres (taches), m baisse et le
    seuil suit, là où un seuil global effacerait ou noircirait la zone.

    Args:
        gray: Image en niveaux de gris.
        window: Taille de la fenêtre (impaire).
        k: Sensibilité à l'écart-type local.
        r: Dynamique de l'écart-type (128 pour des images 8 bits).

    Returns:
        Image binaire (0 = encre, 255 = fond), uint8.

    Example:
        >>> sorted(np.unique(sauvola(np.arange(1600, dtype=np.uint8).reshape(40, 40))).tolist())
        [0, 255]
    """
    g = gray.astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(g * g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return np.where(g > mean * (1 + k * (std / r - 1)), 255, 0).astype(np.uint8)


VARIANTS = {
    # nom : étapes appliquées, dans l'ordre
    "brut": [],
    "gris": ["gray"],
    "redresse": ["deskew", "gray"],
    "clahe": ["deskew", "gray", "clahe"],
    "debruite_clahe": ["deskew", "gray", "denoise", "clahe"],
    "fond_clahe": ["deskew", "gray", "denoise", "background", "clahe"],
    "sauvola": ["deskew", "gray", "denoise", "background", "sauvola"],
}


def run(img: np.ndarray, steps: list[str], params: dict) -> Preprocessed:
    """Applique une suite d'étapes de prétraitement.

    Args:
        img: Image BGR d'origine, uint8.
        steps: Étapes parmi « deskew », « gray », « denoise », « background », « clahe », « sauvola ».
        params: Paramètres (voir `config.PREPROCESS`).

    Returns:
        Image prétraitée et matrice affine origine → prétraitée.

    Raises:
        ValueError: Si une étape est inconnue.

    Example:
        >>> run(np.zeros((32, 32, 3), np.uint8), ["gray", "clahe"], {}).image.ndim
        2
    """
    out = Preprocessed(img, IDENTITY.copy(), 0.0)
    for step in steps:
        if step == "deskew":
            out = deskew(out.image, params.get("deskew_max_angle", 5.0))
        elif step == "gray":
            out.image = to_gray(out.image)
        elif step == "denoise":
            out.image = denoise(to_gray(out.image), params.get("denoise_h", 7))
        elif step == "background":
            out.image = normalize_background(to_gray(out.image), params.get("background_kernel", 51))
        elif step == "clahe":
            out.image = clahe(to_gray(out.image), params.get("clahe_clip", 2.0), params.get("clahe_tile", 8))
        elif step == "sauvola":
            out.image = sauvola(to_gray(out.image), params.get("sauvola_window", 31), params.get("sauvola_k", 0.2))
        else:
            raise ValueError(f"étape inconnue : {step}")
    return out


def is_binary(img: np.ndarray) -> bool:
    """Vrai si l'image ne contient que les valeurs 0 et 255.

    Args:
        img: Image uint8.

    Returns:
        Booléen.

    Example:
        >>> is_binary(np.array([[0, 255]], np.uint8))
        True
    """
    return set(np.unique(img).tolist()) <= {0, 255}
