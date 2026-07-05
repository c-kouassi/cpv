"""Segmentation et reconnaissance avec Kraken (API Python, testé avec kraken 7.1).

Deux façons d'évaluer un modèle de reconnaissance :

- **lignes de référence** (`recognize_reference_lines`) : on donne au modèle les
  lignes de base et polygones de la vérité terrain ; l'erreur mesurée ne vient
  que de la reconnaissance ;
- **bout en bout** (`predict_page` puis `match_lines`) : prétraitement,
  segmentation et reconnaissance automatiques, lignes prédites appariées aux
  lignes de référence par IoU ; une ligne de référence non détectée compte
  comme entièrement supprimée, une ligne prédite en trop comme insérée.

Rappel : `kraken segment` sans `-bl` utilise l'ancien segmenteur par boîtes ;
ici on appelle toujours le segmenteur neuronal par lignes de base (`blla`).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image

from src.metrics import match_polygons, polygon_iou, polygon_mask
from src.preprocess import Preprocessed, invert_points


def fetch_model(doi: str, dest: Path) -> Path:
    """Télécharge un modèle du dépôt Kraken (`kraken get`) et le copie en `dest`.

    Args:
        doi: DOI Zenodo du modèle.
        dest: Chemin du fichier `.mlmodel` à créer.

    Returns:
        Le chemin `dest`.

    Raises:
        RuntimeError: Si la sortie de `kraken get` ne donne pas le fichier téléchargé.

    Example:
        >>> fetch_model("10.5281/zenodo.10592716", Path("models/kraken/catmus_print_large.mlmodel"))
    """
    if dest.exists():
        return dest
    out = subprocess.run(["kraken", "get", doi], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Model dir: (.+?) \(model files: ([^)]+)\)", " ".join(out.split()))
    if not m:
        raise RuntimeError(f"Sortie inattendue de `kraken get {doi}` :\n{out}")
    src = Path(m.group(1)) / m.group(2).split(",")[0].strip()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    return dest


def load_model(path, device: str = "cpu"):
    """Charge un modèle de reconnaissance Kraken.

    Args:
        path: Fichier `.mlmodel`.
        device: « cpu », « mps » ou « cuda:0 ».

    Returns:
        Le reconnaisseur (TorchSeqRecognizer).

    Example:
        >>> model = load_model("models/kraken/catmus_print_large.mlmodel")
    """
    from kraken.lib import models
    return models.load_any(str(path), device=device)


def load_segmenter(path=None):
    """Charge un modèle de segmentation (None = modèle par défaut de Kraken, « blla »).

    Args:
        path: Fichier de poids (`.safetensors` produit par `ketos segtrain`, ou `.mlmodel`), ou None.

    Returns:
        Un `SegmentationTaskModel` Kraken.

    Example:
        >>> segmenter = load_segmenter("models/kraken/segmentation/best.safetensors")
    """
    from kraken.tasks import SegmentationTaskModel
    return SegmentationTaskModel.load_model(str(path) if path else None)


def segment(segmenter, image: Image.Image, accelerator: str = "cpu"):
    """Segmente une page : zones typées et lignes (lignes de base + polygones).

    Args:
        segmenter: Modèle chargé par `load_segmenter`.
        image: Image PIL de la page.
        accelerator: « cpu », « mps » ou « gpu ».

    Returns:
        Un objet `Segmentation` de Kraken.

    Example:
        >>> seg = segment(load_segmenter(), Image.open("data/images/page_014.png"))
    """
    from kraken.configs import SegmentationInferenceConfig
    cfg = SegmentationInferenceConfig(accelerator=accelerator, device=1 if accelerator != "cpu" else "auto")
    return segmenter.predict(image, cfg)


@dataclass
class PredictedLine:
    """Ligne produite par la chaîne automatique, dans le repère de l'image d'origine.

    Attributes:
        line_id: Identifiant.
        text: Transcription.
        confidence: Moyenne des probabilités des caractères décodés.
        char_confidences: Probabilité de chaque caractère.
        baseline: Ligne de base (x, y).
        polygon: Contour (x, y).
        region_type: Classe de la zone contenant la ligne (si le segmenteur en prédit).
    """
    line_id: str
    text: str
    confidence: float
    char_confidences: list[float]
    baseline: list[list[int]]
    polygon: list[list[int]]
    region_type: str | None = None


@dataclass
class PredictedPage:
    """Résultat de la chaîne sur une page.

    Attributes:
        image: Chemin de l'image d'origine.
        width: Largeur (pixels).
        height: Hauteur (pixels).
        lines: Lignes conservées (hors illustrations).
        regions: Zones prédites : {"type", "polygon"}.
        dropped: Lignes écartées car situées dans une illustration.
        skew_angle: Angle de redressement appliqué.
    """
    image: str
    width: int
    height: int
    lines: list[PredictedLine] = field(default_factory=list)
    regions: list[dict] = field(default_factory=list)
    dropped: int = 0
    skew_angle: float = 0.0


def to_pil(img: np.ndarray) -> Image.Image:
    """Convertit une image OpenCV (BGR ou gris) en image PIL.

    Args:
        img: Tableau uint8.

    Returns:
        Image PIL (RGB ou L).

    Example:
        >>> to_pil(np.zeros((2, 2), np.uint8)).mode
        'L'
    """
    return Image.fromarray(img) if img.ndim == 2 else Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))


def _region_types(seg) -> dict[str, str]:
    return {r.id: rtype for rtype, regs in (seg.regions or {}).items() for r in regs}


def predict_page(image_path: Path, prep: Preprocessed, model, segmenter,
                 drop_types: tuple[str, ...] = ("illustration",), accelerator: str = "cpu",
                 illustration_overlap: float | None = None) -> PredictedPage:
    """Segmente et transcrit une page prétraitée ; coordonnées ramenées à l'image d'origine.

    Les lignes rattachées à une zone d'illustration (gravure, lettrine, estampille)
    sont écartées : un segmenteur générique prend volontiers les hachures d'une
    gravure pour des lignes de texte.

    Args:
        image_path: Image d'origine (pour les métadonnées).
        prep: Image prétraitée et matrice de transformation (`preprocess.run`).
        model: Reconnaisseur Kraken.
        segmenter: Modèle de segmentation (`load_segmenter`).
        drop_types: Classes de zones dont les lignes sont écartées.
        accelerator: Périphérique pour la segmentation.
        illustration_overlap: Si défini, écarte aussi toute ligne dont cette part
            (ou plus) de la surface tombe dans une zone d'illustration prédite.
            Nécessaire car le segmenteur rattache souvent ces lignes à la zone
            « text » héritée du modèle d'origine plutôt qu'à l'illustration.

    Returns:
        La page prédite.

    Example:
        >>> page = predict_page(Path("data/images/page_014.png"), preprocess.run(img, [], {}), model, load_segmenter())
    """
    from kraken import rpred
    im = to_pil(prep.image)
    seg = segment(segmenter, im, accelerator)
    types = _region_types(seg)
    ill = None
    if illustration_overlap is not None:
        h0, w0 = prep.image.shape[:2]
        ill = np.zeros((h0 // 4 + 1, w0 // 4 + 1), bool)
        for rtype, regs in (seg.regions or {}).items():
            if rtype in drop_types:
                for r in regs:
                    ill |= polygon_mask((np.asarray(r.boundary) / 4).astype(int), ill.shape)
    keep, dropped = [], 0
    for line in seg.lines:
        rtype = next((types[r] for r in (line.regions or []) if r in types), None)
        inside = False
        if ill is not None and line.boundary and len(line.boundary) >= 3:
            m = polygon_mask((np.asarray(line.boundary) / 4).astype(int), ill.shape)
            inside = m.any() and (m & ill).sum() / m.sum() >= illustration_overlap
        if rtype in drop_types or inside:
            dropped += 1
        else:
            keep.append(line)
    h, w = prep.image.shape[:2]
    out = PredictedPage(str(image_path), w, h, dropped=dropped, skew_angle=prep.angle)
    for rtype, regs in (seg.regions or {}).items():
        for r in regs:
            out.regions.append({"type": rtype, "polygon": invert_points(r.boundary, prep.matrix)})
    if not keep:
        return out
    for rec in rpred.rpred(model, im, replace(seg, lines=keep)):
        confs = [float(c) for c in rec.confidences]
        if not confs:
            continue
        rtype = next((types[r] for r in (rec.regions or []) if r in types), None)
        out.lines.append(PredictedLine(
            line_id=f"l{len(out.lines) + 1:03d}", text=rec.prediction,
            confidence=float(np.mean(confs)), char_confidences=confs,
            baseline=invert_points(rec.baseline, prep.matrix),
            polygon=invert_points(rec.boundary, prep.matrix), region_type=rtype))
    return out


def recognize_reference_lines(image: Image.Image, lines: list, model) -> list[dict]:
    """Transcrit les lignes de la vérité terrain avec leur géométrie de référence.

    Args:
        image: Image de la page (repère des polygones de référence).
        lines: Lignes de référence (objets `corpus.Line`).
        model: Reconnaisseur Kraken.

    Returns:
        [{"line_id", "ref", "hyp", "confidence"}] dans l'ordre des lignes.

    Example:
        >>> res = recognize_reference_lines(Image.open(img), page_lines, model)
    """
    from kraken import rpred
    from kraken.containers import BaselineLine, Segmentation
    bl = [BaselineLine(id=l.line_id, baseline=[tuple(p) for p in l.baseline],
                       boundary=[tuple(p) for p in l.polygon])
          for l in lines if len(l.baseline) >= 2 and len(l.polygon) >= 3]
    seg = Segmentation(type="baselines", imagename="page", text_direction="horizontal-lr",
                       script_detection=False, lines=bl)
    preds = {}
    for rec in rpred.rpred(model, image, seg):
        confs = list(rec.confidences)
        preds[rec.id] = (rec.prediction, float(np.mean(confs)) if confs else 0.0)
    return [{"line_id": l.line_id, "ref": l.text, "hyp": preds.get(l.line_id, ("", 0.0))[0],
             "confidence": preds.get(l.line_id, ("", 0.0))[1]} for l in lines]


def _bbox_overlap(a, b) -> bool:
    if len(a) < 3 or len(b) < 3:
        return False
    a, b = np.asarray(a), np.asarray(b)
    return not (a[:, 0].max() < b[:, 0].min() or b[:, 0].max() < a[:, 0].min()
                or a[:, 1].max() < b[:, 1].min() or b[:, 1].max() < a[:, 1].min())


def match_lines(pred: list[PredictedLine], ref: list, iou_threshold: float = 0.5) -> tuple[list[tuple[str, str]], dict]:
    """Apparie lignes prédites et lignes de référence par IoU de polygones.

    Args:
        pred: Lignes prédites.
        ref: Lignes de référence (`corpus.Line`).
        iou_threshold: IoU minimale d'un appariement.

    Returns:
        (paires (référence, hypothèse) pour le CER de bout en bout, mesures de
        segmentation : précision, rappel, F1, IoU moyenne des références).
        Référence non détectée → (ref, "") ; prédiction en trop → ("", hyp).

    Example:
        >>> pairs, seg = match_lines(page.lines, ref_lines)
    """
    m = match_polygons([p.polygon for p in pred], [r.polygon for r in ref], iou_threshold)
    cand = sorted(((polygon_iou(p.polygon, r.polygon), i, j)
                   for i, p in enumerate(pred) for j, r in enumerate(ref)
                   if _bbox_overlap(p.polygon, r.polygon)), reverse=True)
    used_p, used_r, pairs = set(), set(), []
    for iou, i, j in cand:
        if iou < iou_threshold:
            break
        if i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
            pairs.append((ref[j].text, pred[i].text))
    pairs += [(r.text, "") for j, r in enumerate(ref) if j not in used_r]
    pairs += [("", p.text) for i, p in enumerate(pred) if i not in used_p]
    return pairs, {k: m[k] for k in ["precision", "recall", "f1", "mean_iou"]}


def to_pagexml(page: PredictedPage, out: Path) -> None:
    """Écrit une page prédite en PAGE XML (zones, lignes de base, polygones, texte, confiances).

    Les coordonnées sont en pixels de l'image d'origine, origine en haut à
    gauche. Le chemin de l'image est relatif au fichier XML.

    Args:
        page: Page prédite.
        out: Fichier `.xml` à écrire.

    Example:
        >>> to_pagexml(page, Path("segmentations/page_014.xml"))
    """
    from kraken import serialization
    from kraken.containers import BaselineLine, BaselineOCRRecord, Region, Segmentation
    regions = {}
    for k, r in enumerate(page.regions):
        if len(r["polygon"]) >= 3:
            regions.setdefault(r["type"], []).append(
                Region(id=f"r{k + 1:03d}", boundary=[tuple(p) for p in r["polygon"]],
                       tags={"type": [{"type": r["type"]}]}))
    records = []
    for l in page.lines:
        line = BaselineLine(id=l.line_id, baseline=[tuple(p) for p in l.baseline],
                            boundary=[tuple(p) for p in l.polygon])
        records.append(BaselineOCRRecord(prediction=l.text, cuts=[], confidences=l.char_confidences,
                                         line=line, display_order=False))
    rel = os.path.relpath(Path(page.image).resolve(), out.parent.resolve())
    seg = Segmentation(type="baselines", imagename=rel, text_direction="horizontal-lr",
                       script_detection=False, lines=records, regions=regions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(serialization.serialize(seg, image_size=(page.width, page.height), template="pagexml",
                                           sub_line_segmentation=False), encoding="utf-8")


def illustration_mask(page: PredictedPage) -> np.ndarray:
    """Masque des zones d'illustration prédites (utile pour les figures).

    Args:
        page: Page prédite.

    Returns:
        Masque booléen (hauteur, largeur).

    Example:
        >>> illustration_mask(page).shape
    """
    mask = np.zeros((page.height, page.width), bool)
    for r in page.regions:
        if r["type"] == "illustration":
            mask |= polygon_mask(r["polygon"], mask.shape)
    return mask
