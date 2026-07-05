"""Métriques : Levenshtein, CER, WER, intervalles bootstrap, McNemar, IoU, calibration.

Le CER n'est pas une mesure à part : c'est la distance de Levenshtein entre la
transcription de référence et l'hypothèse, divisée par la longueur de la
référence.

    CER = (S + D + I) / N

    S, D, I = substitutions, suppressions, insertions du chemin d'édition minimal
    N       = nombre de caractères de la RÉFÉRENCE

Conséquences : le CER n'est pas symétrique, il peut dépasser 1, et sur un corpus
on somme les éditions et les longueurs avant de diviser (une ligne de 3
caractères ne pèse pas autant qu'une ligne de 60).
"""
from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class EditCounts:
    """Décompte des opérations d'édition entre une référence et une hypothèse.

    Attributes:
        substitutions: Nombre de substitutions.
        deletions: Nombre de suppressions (caractères de la référence absents).
        insertions: Nombre d'insertions (caractères en trop dans l'hypothèse).
        ref_length: Longueur de la référence.
    """
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_length: int = 0

    @property
    def distance(self) -> int:
        """Distance de Levenshtein (S + D + I)."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        """Taux d'erreur (S + D + I) / N ; 1.0 si la référence est vide et l'hypothèse non."""
        return self.distance / self.ref_length if self.ref_length else float(self.distance > 0)

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(self.substitutions + other.substitutions, self.deletions + other.deletions,
                          self.insertions + other.insertions, self.ref_length + other.ref_length)


def _matrix(ref, hyp) -> list[list[int]]:
    # Programmation dynamique de Wagner-Fischer : d[i][j] = distance entre
    # les i premiers éléments de ref et les j premiers de hyp.
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d


def levenshtein(a, b) -> int:
    """Distance de Levenshtein entre deux séquences.

    Args:
        a: Première séquence (chaîne ou liste de mots).
        b: Seconde séquence.

    Returns:
        Nombre minimal d'insertions, suppressions et substitutions pour passer de a à b.

    Example:
        >>> levenshtein("kitten", "sitting")
        3
    """
    return _matrix(a, b)[len(a)][len(b)]


def align(ref, hyp) -> list[tuple[str, object, object]]:
    """Chemin d'édition minimal entre une référence et une hypothèse.

    Args:
        ref: Séquence de référence.
        hyp: Séquence produite.

    Returns:
        Liste de (opération, élément_ref, élément_hyp) ; opérations « = » (identité),
        « S » (substitution), « D » (suppression), « I » (insertion).

    Example:
        >>> align("ab", "ac")
        [('=', 'a', 'a'), ('S', 'b', 'c')]
    """
    d = _matrix(ref, hyp)
    i, j, ops = len(ref), len(hyp), []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def edit_counts(ref, hyp) -> EditCounts:
    """Décompte S, D, I sur le chemin d'édition minimal.

    Args:
        ref: Séquence de référence.
        hyp: Séquence produite.

    Returns:
        Les décomptes et la longueur de la référence.

    Example:
        >>> edit_counts("abc", "abd").substitutions
        1
    """
    c = Counter(op for op, _, _ in align(ref, hyp))
    return EditCounts(c["S"], c["D"], c["I"], len(ref))


def prepare(text: str) -> str:
    """Forme canonique avant comparaison : NFC et espaces normalisés.

    Sans NFC, « ũ » précomposé et « u + tilde combinant » compteraient comme une
    erreur alors qu'ils s'affichent à l'identique.

    Args:
        text: Texte à comparer.

    Returns:
        Texte en NFC, espaces multiples réduits.

    Example:
        >>> prepare("u\\u0303  g") == prepare("\\u0169 g")
        True
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def cer(ref: str, hyp: str) -> float:
    """CER d'une ligne.

    Args:
        ref: Transcription de référence.
        hyp: Transcription produite.

    Returns:
        Distance de Levenshtein / longueur de la référence.

    Example:
        >>> cer("abcd", "ab")
        0.5
    """
    return edit_counts(prepare(ref), prepare(hyp)).error_rate


def wer(ref: str, hyp: str) -> float:
    """WER d'une ligne (même calcul que le CER, sur les mots).

    Args:
        ref: Transcription de référence.
        hyp: Transcription produite.

    Returns:
        Distance de Levenshtein entre listes de mots / nombre de mots de la référence.

    Example:
        >>> wer("le roy nostre sire", "le roy noſtre sire")
        0.25
    """
    return edit_counts(prepare(ref).split(), prepare(hyp).split()).error_rate


def per_line_counts(pairs, unit: str = "char") -> tuple[np.ndarray, np.ndarray]:
    """Distances et longueurs de référence ligne par ligne.

    Args:
        pairs: Itérable de (référence, hypothèse).
        unit: « char » ou « word ».

    Returns:
        Deux tableaux : distances, longueurs de référence.

    Example:
        >>> per_line_counts([("ab", "ac")])
        (array([1]), array([2]))
    """
    dist, length = [], []
    for ref, hyp in pairs:
        ref, hyp = prepare(ref), prepare(hyp)
        if unit == "word":
            ref, hyp = ref.split(), hyp.split()
        dist.append(levenshtein(ref, hyp))
        length.append(len(ref))
    return np.array(dist), np.array(length)


def corpus_counts(pairs, unit: str = "char") -> EditCounts:
    """Somme des éditions sur un ensemble de lignes (CER/WER micro-moyennés).

    Args:
        pairs: Itérable de (référence, hypothèse).
        unit: « char » (CER) ou « word » (WER).

    Returns:
        Décomptes cumulés ; `.error_rate` donne le CER ou WER global.

    Example:
        >>> corpus_counts([("a", "b"), ("abcdefghij", "abcdefghij")]).error_rate
        0.09090909090909091
    """
    total = EditCounts()
    for ref, hyp in pairs:
        ref, hyp = prepare(ref), prepare(hyp)
        if unit == "word":
            ref, hyp = ref.split(), hyp.split()
        total = total + edit_counts(ref, hyp)
    return total


def bootstrap_ci(dist: np.ndarray, length: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """Intervalle de confiance du taux d'erreur global par rééchantillonnage des lignes.

    On tire n lignes avec remise, n_boot fois, et on recalcule à chaque fois
    somme(distances) / somme(longueurs). Les bornes sont les quantiles α/2 et 1−α/2.

    Args:
        dist: Distances par ligne (voir `per_line_counts`).
        length: Longueurs de référence par ligne.
        n_boot: Nombre de rééchantillonnages (1000 dans le sujet).
        alpha: Risque (0.05 pour un intervalle à 95 %).
        seed: Graine du tirage.

    Returns:
        (borne basse, borne haute).

    Raises:
        ValueError: Si aucune ligne n'est fournie.

    Example:
        >>> bootstrap_ci(np.array([0, 1, 2]), np.array([10, 10, 10]))
        (0.0, 0.2)
    """
    if len(dist) == 0:
        raise ValueError("aucune ligne")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dist), size=(n_boot, len(dist)))
    rates = dist[idx].sum(axis=1) / np.maximum(length[idx].sum(axis=1), 1)
    return float(np.quantile(rates, alpha / 2)), float(np.quantile(rates, 1 - alpha / 2))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Test de McNemar exact entre deux systèmes évalués sur les mêmes lignes.

    Seules comptent les lignes discordantes : b = juste pour A et faux pour B,
    c = faux pour A et juste pour B. Sous H0 (même taux d'erreur), b suit une
    loi binomiale B(b + c, 1/2) ; la p-valeur bilatérale est exacte.

    Args:
        correct_a: Booléens, ligne correcte pour le système A.
        correct_b: Booléens, ligne correcte pour le système B (mêmes lignes).

    Returns:
        {"b", "c", "p_value"}.

    Raises:
        ValueError: Si les deux tableaux n'ont pas la même longueur.

    Example:
        >>> mcnemar(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))["b"]
        1
    """
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    if a.shape != b_.shape:
        raise ValueError("les deux systèmes doivent être évalués sur les mêmes lignes")
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"b": b, "c": c, "p_value": min(1.0, 2 * p)}


def confusions(pairs, top: int = 20) -> list[tuple[tuple[str, str], int]]:
    """Erreurs les plus fréquentes au caractère.

    Args:
        pairs: Itérable de (référence, hypothèse).
        top: Nombre de confusions renvoyées.

    Returns:
        Liste de ((attendu, obtenu), nombre) ; « ∅ » = caractère absent.

    Example:
        >>> confusions([("ſus", "fus")])
        [(('ſ', 'f'), 1)]
    """
    c = Counter()
    for ref, hyp in pairs:
        for op, r, h in align(prepare(ref), prepare(hyp)):
            if op != "=":
                c[(r or "∅", h or "∅")] += 1
    return c.most_common(top)


# ── Géométrie ────────────────────────────────────────────────────────────────
def polygon_mask(polygon, shape: tuple[int, int]) -> np.ndarray:
    """Rastérise un polygone en masque booléen.

    Args:
        polygon: Liste de points (x, y) en pixels, origine en haut à gauche.
        shape: (hauteur, largeur) de l'image.

    Returns:
        Masque booléen de forme `shape`.

    Example:
        >>> polygon_mask([[0, 0], [9, 0], [9, 9], [0, 9]], (20, 20)).sum()
        100
    """
    import cv2
    mask = np.zeros(shape, np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def polygon_iou(p1, p2, shape: tuple[int, int] | None = None) -> float:
    """Intersection sur union de deux polygones, calculée par rastérisation.

    Args:
        p1: Premier polygone (points x, y).
        p2: Second polygone.
        shape: Taille du canevas ; par défaut, la boîte englobante des deux polygones.

    Returns:
        IoU entre 0 et 1.

    Example:
        >>> round(polygon_iou([[0,0],[10,0],[10,10],[0,10]], [[5,0],[15,0],[15,10],[5,10]]), 2)
        0.38
    """
    pts = np.asarray(list(p1) + list(p2))
    if len(p1) < 3 or len(p2) < 3:
        return 0.0
    off = pts.min(axis=0)
    h, w = (pts.max(axis=0) - off + 2)[::-1]
    m1 = polygon_mask(np.asarray(p1) - off, (int(h), int(w)))
    m2 = polygon_mask(np.asarray(p2) - off, (int(h), int(w)))
    union = np.logical_or(m1, m2).sum()
    return float(np.logical_and(m1, m2).sum() / union) if union else 0.0


def match_polygons(pred: list, ref: list, threshold: float = 0.5) -> dict:
    """Apparie prédictions et références par IoU décroissante (appariement glouton un-à-un).

    Args:
        pred: Polygones prédits.
        ref: Polygones de référence.
        threshold: IoU minimale pour qu'un appariement compte comme détection.

    Returns:
        {"ious": IoU de chaque référence avec sa meilleure prédiction appariée (0 si aucune),
         "precision", "recall", "f1", "mean_iou"}.

    Example:
        >>> match_polygons([[[0,0],[9,0],[9,9],[0,9]]], [[[0,0],[9,0],[9,9],[0,9]]])["mean_iou"]
        1.0
    """
    def _bbox(p):
        a = np.asarray(p)
        return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
    cand = []
    for i, p in enumerate(pred):
        bp = _bbox(p) if len(p) else None
        for j, r in enumerate(ref):
            if bp is None or not len(r):
                continue
            br = _bbox(r)
            if bp[0] > br[2] or br[0] > bp[2] or bp[1] > br[3] or br[1] > bp[3]:
                continue
            iou = polygon_iou(p, r)
            if iou > 0:
                cand.append((iou, i, j))
    used_p, used_r, ious = set(), set(), [0.0] * len(ref)
    for iou, i, j in sorted(cand, reverse=True):
        if i not in used_p and j not in used_r:
            used_p.add(i); used_r.add(j); ious[j] = iou
    tp = sum(i >= threshold for i in ious)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"ious": ious, "precision": precision, "recall": recall, "f1": f1,
            "mean_iou": float(np.mean(ious)) if ious else 0.0}


# ── Calibration des confiances ───────────────────────────────────────────────
def isotonic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Régression isotone (algorithme PAV) : fonction croissante de x approchant y.

    Sert à calibrer la confiance brute d'une ligne en probabilité d'exactitude,
    sur la validation.

    Args:
        x: Scores bruts (ex. confiance moyenne de la ligne).
        y: Cibles dans [0, 1] (ex. 1 si la ligne est correcte).

    Returns:
        (seuils x triés, valeurs calibrées correspondantes), à utiliser avec
        `isotonic_predict`.

    Example:
        >>> xs, ys = isotonic_fit(np.array([0.1, 0.5, 0.9]), np.array([0, 1, 1]))
    """
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    blocks = [[v, 1.0] for v in ys]          # [moyenne, poids]
    starts = list(range(len(ys)))
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [m, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]; del starts[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.empty(len(ys))
    for k, (m, w) in enumerate(blocks):
        fitted[starts[k]:starts[k] + int(w)] = m
    return xs, fitted


def isotonic_predict(model: tuple[np.ndarray, np.ndarray], x) -> np.ndarray:
    """Applique une calibration isotone (interpolation linéaire entre seuils).

    Args:
        model: Sortie de `isotonic_fit`.
        x: Scores bruts.

    Returns:
        Probabilités calibrées.

    Example:
        >>> isotonic_predict((np.array([0., 1.]), np.array([0.2, 0.8])), [0.5])
        array([0.5])
    """
    xs, ys = model
    return np.interp(np.asarray(x, float), xs, ys)


def expected_calibration_error(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Erreur de calibration attendue (ECE) et données du diagramme de fiabilité.

    Args:
        prob: Probabilités annoncées.
        outcome: Résultats observés (0/1 ou valeurs dans [0, 1]).
        n_bins: Nombre de tranches de probabilité.

    Returns:
        (ECE, liste par tranche de {"bin", "n", "confiance", "exactitude"}).

    Example:
        >>> expected_calibration_error(np.array([0.9, 0.9]), np.array([1, 1]))[0]
        0.1
    """
    prob, outcome = np.asarray(prob, float), np.asarray(outcome, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, table = 0.0, []
    for k in range(n_bins):
        sel = (prob >= edges[k]) & ((prob < edges[k + 1]) | (k == n_bins - 1))
        if sel.any():
            conf, acc = prob[sel].mean(), outcome[sel].mean()
            ece += sel.mean() * abs(conf - acc)
            table.append({"bin": round(float(edges[k]), 2), "n": int(sel.sum()),
                          "confiance": float(conf), "exactitude": float(acc)})
    return round(float(ece), 6), table
