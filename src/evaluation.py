"""Évaluation de la chaîne complète sur une partition du corpus (pages ALTO harmonisées)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src import htr, metrics, preprocess
from src.corpus import Page, read_alto

REGION_CLASSES = ("texte", "illustration", "paratexte")
# Le segmenteur par défaut de Kraken ne connaît qu'une classe « text ».
DEFAULT_MODEL_ALIASES = {"text": "texte"}


def load_split_pages(split_dir: Path) -> list[Page]:
    """Relit les pages ALTO harmonisées d'une partition.

    Args:
        split_dir: Dossier `data/corpus/<partition>`.

    Returns:
        Pages avec zones (classes regroupées) et lignes de référence.

    Example:
        >>> pages = load_split_pages(Path("data/corpus/val"))
    """
    pages = []
    for xml in sorted(split_dir.glob("*.xml")):
        book = xml.name.split("__")[0]
        pages.append(read_alto(xml, "corpus", book))
    return pages


def region_iou_counts(pred_regions: list[dict], ref_regions: list[dict], shape: tuple[int, int],
                      scale: float = 0.25) -> dict[str, tuple[int, int]]:
    """Intersection et union au pixel, par classe de zone, sur une page.

    Les masques sont calculés sur une image réduite (`scale`) pour la vitesse ;
    l'IoU est un rapport de surfaces, peu sensible à cette réduction.

    Args:
        pred_regions: Zones prédites {"type", "polygon"}.
        ref_regions: Zones de référence {"type", "polygon"}.
        shape: (hauteur, largeur) de l'image d'origine.
        scale: Facteur de réduction.

    Returns:
        Classe → (intersection, union) en pixels réduits.

    Example:
        >>> region_iou_counts([], [], (100, 100))["texte"]
        (0, 0)
    """
    h, w = int(shape[0] * scale), int(shape[1] * scale)
    out = {}
    for cls in REGION_CLASSES:
        mp = np.zeros((h, w), bool)
        mr = np.zeros((h, w), bool)
        for r in pred_regions:
            if DEFAULT_MODEL_ALIASES.get(r["type"], r["type"]) == cls:
                mp |= metrics.polygon_mask((np.asarray(r["polygon"]) * scale).astype(int), (h, w))
        for r in ref_regions:
            if r["type"] == cls:
                mr |= metrics.polygon_mask((np.asarray(r["polygon"]) * scale).astype(int), (h, w))
        out[cls] = (int((mp & mr).sum()), int((mp | mr).sum()))
    return out


@dataclass
class EndToEndResult:
    """Résultats de bout en bout sur un ensemble de pages.

    Attributes:
        pairs: Paires (référence, hypothèse) après appariement des lignes.
        line_ious: IoU de chaque ligne de référence avec sa ligne prédite appariée.
        detection: Par page, précision / rappel / F1 de la détection de lignes.
        regions: Classe → [intersection, union] cumulées.
    """
    pairs: list[tuple[str, str]] = field(default_factory=list)
    line_ious: list[float] = field(default_factory=list)
    detection: list[dict] = field(default_factory=list)
    regions: dict = field(default_factory=lambda: {c: [0, 0] for c in REGION_CLASSES})

    def summary(self, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42,
                iou_threshold: float = 0.5) -> dict:
        """Métriques globales.

        Args:
            n_boot: Rééchantillonnages bootstrap.
            alpha: Risque de l'intervalle.
            seed: Graine.
            iou_threshold: Seuil d'IoU pour compter une ligne comme détectée.

        Returns:
            CER, WER, IC du CER, IoU moyenne des lignes, part des lignes à IoU > seuil,
            F1 moyen de détection, IoU par classe de zone.

        Example:
            >>> EndToEndResult(pairs=[("a", "a")]).summary()["cer"]
            0.0
        """
        d, n = metrics.per_line_counts(self.pairs)
        dw, nw = metrics.per_line_counts(self.pairs, unit="word")
        lo, hi = metrics.bootstrap_ci(d, n, n_boot, alpha, seed)
        return {
            "lignes_reference": sum(1 for r, _ in self.pairs if r),
            "cer": float(d.sum() / max(n.sum(), 1)), "cer_ic95": [lo, hi],
            "wer": float(dw.sum() / max(nw.sum(), 1)),
            "iou_lignes_moyenne": float(np.mean(self.line_ious)) if self.line_ious else 0.0,
            "part_lignes_iou_sup_seuil": float(np.mean(np.asarray(self.line_ious) >= iou_threshold)) if self.line_ious else 0.0,
            "detection_f1_moyen": float(np.mean([x["f1"] for x in self.detection])) if self.detection else 0.0,
            "iou_zones": {c: (i / u if u else None) for c, (i, u) in self.regions.items()},
        }


def run_end_to_end(pages: list[Page], steps: list[str], params: dict, model, segmenter,
                   accelerator: str = "cpu", iou_threshold: float = 0.5,
                   drop_types: tuple[str, ...] = ("illustration",),
                   illustration_overlap: float | None = None) -> EndToEndResult:
    """Applique la chaîne complète aux pages et la compare à la vérité terrain.

    Args:
        pages: Pages de référence.
        steps: Étapes de prétraitement (`preprocess.VARIANTS`).
        params: Paramètres de prétraitement.
        model: Reconnaisseur.
        segmenter: Segmenteur (`htr.load_segmenter`).
        accelerator: Périphérique de segmentation.
        iou_threshold: IoU minimale d'appariement des lignes.
        drop_types: Classes de zones dont les lignes sont écartées.
        illustration_overlap: Seuil de recouvrement avec une illustration (voir `htr.predict_page`).

    Returns:
        Résultats cumulés.

    Example:
        >>> res = run_end_to_end(load_split_pages(Path("data/corpus/val"))[:2], [], {}, model, segmenter)
    """
    res = EndToEndResult()
    for page in pages:
        img = cv2.imread(str(page.image))
        prep = preprocess.run(img, steps, params)
        pred = htr.predict_page(page.image, prep, model, segmenter, drop_types, accelerator, illustration_overlap)
        pairs, det = htr.match_lines(pred.lines, page.lines, iou_threshold)
        res.pairs += pairs
        res.detection.append(det)
        res.line_ious += metrics.match_polygons([l.polygon for l in pred.lines],
                                                [l.polygon for l in page.lines], iou_threshold)["ious"]
        for cls, (i, u) in region_iou_counts(pred.regions, page.regions, img.shape[:2]).items():
            res.regions[cls][0] += i
            res.regions[cls][1] += u
    return res
