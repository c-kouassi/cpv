"""Corpus d'entraînement : lecture des ALTO, harmonisation, partition, empreintes.

Les jeux GalliCorpora sont fournis en ALTO v4 (une page = une image + un XML),
avec des zones typées selon SegmOnto (MainZone, GraphicZone, DropCapitalZone…).
Ce module les lit, harmonise les conventions de transcription des deux jeux,
partitionne PAR LIVRE et calcule les empreintes SHA-256 de chaque partition.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

ALTO_NS = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}

# Harmonisation entre les deux jeux (voir CONVENTIONS_TRANSCRIPTION.md).
# Le jeu « incunables » (le plus gros) suit CREMMALAB : s long transcrit « s ».
# Le jeu « gothiques 16e » garde « ſ » : on l'aligne sur le premier, sinon le
# modèle apprendrait deux graphies pour un même glyphe selon le livre.
HARMONIZE = [
    ("ſ", "s"),
    ("⁋", "¶"),      # pied-de-mouche inversé → pied-de-mouche
    ("¬", "-"),      # trait de coupure en fin de ligne
    ("◌", ""),       # cercle pointillé (support de diacritique isolé, artefact de saisie)
]

ABBREVIATION_MARKS = set("ꝑꝓꝗꝙꝯꝰ⁊") | {chr(c) for c in range(0x0300, 0x0370)} | {chr(c) for c in range(0x1DC0, 0x1E00)}


@dataclass
class Line:
    """Une ligne de vérité terrain.

    Attributes:
        dataset: Clé du jeu de données (voir `config.DATASETS`).
        book: Identifiant Gallica du livre (dossier).
        page: Nom de la page (sans extension).
        line_id: Identifiant de la ligne dans l'ALTO.
        text: Transcription harmonisée (NFC).
        baseline: Ligne de base, liste de points (x, y) en pixels.
        polygon: Contour de la ligne, liste de points (x, y) en pixels.
        region_type: Type SegmOnto de la zone contenant la ligne.
        image: Chemin de l'image de la page (relatif au dossier `data/`).
    """
    dataset: str
    book: str
    page: str
    line_id: str
    text: str
    baseline: list[list[int]]
    polygon: list[list[int]]
    region_type: str
    image: str = ""


@dataclass
class Page:
    """Une page de vérité terrain : image, dimensions, zones et lignes.

    Attributes:
        dataset: Clé du jeu de données.
        book: Identifiant du livre.
        name: Nom de la page.
        image: Chemin de l'image.
        alto: Chemin du fichier ALTO lu.
        width: Largeur en pixels.
        height: Hauteur en pixels.
        regions: Zones (type SegmOnto et polygone).
        lines: Lignes transcrites.
    """
    dataset: str
    book: str
    name: str
    image: Path
    alto: Path
    width: int
    height: int
    regions: list[dict] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)


def harmonize(text: str) -> str:
    """Harmonise une transcription selon les conventions du projet.

    Args:
        text: Transcription brute.

    Returns:
        Transcription en NFC, conventions des deux jeux alignées, espaces normalisés.

    Example:
        >>> harmonize("noſtre  ſire ⁋")
        'nostre sire ¶'
    """
    text = unicodedata.normalize("NFD", text)
    for src, dst in HARMONIZE:
        text = text.replace(src, dst)
    return unicodedata.normalize("NFC", " ".join(text.split()))


def _points(s: str | None) -> list[list[int]]:
    if not s:
        return []
    nums = [int(float(v)) for v in re.split(r"[\s,]+", s.strip()) if v]
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]


def read_alto(alto: Path, dataset: str, book: str) -> Page:
    """Lit une page ALTO v4 (format eScriptorium / GalliCorpora).

    Args:
        alto: Chemin du fichier ALTO.
        dataset: Clé du jeu de données.
        book: Identifiant du livre.

    Returns:
        La page avec ses zones typées et ses lignes harmonisées.

    Raises:
        FileNotFoundError: Si l'image référencée par l'ALTO est absente.

    Example:
        >>> page = read_alto(Path("…/bpt6k8710026t_f11.chocomufin.xml"), "incunables_15e", "bpt6k8710026t")
        >>> len(page.lines)
        26
    """
    root = ET.parse(alto).getroot()
    tags = {t.get("ID"): t.get("LABEL") for t in root.iterfind(".//a:Tags/a:OtherTag", ALTO_NS)}
    file_name = root.findtext(".//a:sourceImageInformation/a:fileName", namespaces=ALTO_NS).strip()
    image = alto.parent / file_name
    if not image.exists():
        raise FileNotFoundError(image)
    p = root.find(".//a:Page", ALTO_NS)
    page = Page(dataset, book, alto.name.split(".")[0], image, alto,
                int(float(p.get("WIDTH"))), int(float(p.get("HEIGHT"))))
    for block in root.iterfind(".//a:TextBlock", ALTO_NS):
        rtype = tags.get((block.get("TAGREFS") or "").split(" ")[0], "MainZone").split("#")[0]
        poly = _points(block.findtext("a:Shape/a:Polygon", namespaces=ALTO_NS)
                       or (block.find("a:Shape/a:Polygon", ALTO_NS).get("POINTS")
                           if block.find("a:Shape/a:Polygon", ALTO_NS) is not None else None))
        page.regions.append({"type": rtype, "polygon": poly})
        for tl in block.iterfind("a:TextLine", ALTO_NS):
            text = " ".join(s.get("CONTENT", "") for s in tl.iterfind("a:String", ALTO_NS))
            text = harmonize(text)
            shape = tl.find("a:Shape/a:Polygon", ALTO_NS)
            if not text or shape is None:
                continue
            page.lines.append(Line(dataset, book, page.name, tl.get("ID"), text,
                                   _points(tl.get("BASELINE")), _points(shape.get("POINTS")), rtype,
                                   image.as_posix().split("/data/", 1)[-1]))
    # Zones sans texte (illustrations…) : ALTO les encode en Illustration/GraphicalElement.
    for el in list(root.iterfind(".//a:Illustration", ALTO_NS)) + list(root.iterfind(".//a:GraphicalElement", ALTO_NS)):
        rtype = tags.get((el.get("TAGREFS") or "").split(" ")[0], "GraphicZone").split("#")[0]
        shape = el.find("a:Shape/a:Polygon", ALTO_NS)
        page.regions.append({"type": rtype, "polygon": _points(shape.get("POINTS")) if shape is not None else []})
    return page


def load_dataset(root: Path, key: str, skip_books: set[str] = frozenset()) -> list[Page]:
    """Lit toutes les pages d'un jeu GalliCorpora (versions `.chocomufin.xml`).

    Args:
        root: Dossier du dépôt cloné.
        key: Clé du jeu de données.
        skip_books: Livres à ignorer (doublons entre jeux).

    Returns:
        Les pages, triées par livre puis par nom.

    Example:
        >>> pages = load_dataset(Path("data/external/HTR-incunable-15e-siecle"), "incunables_15e")
    """
    pages = []
    for book_dir in sorted(p for p in (root / "data").iterdir() if p.is_dir()):
        if book_dir.name in skip_books:
            continue
        for alto in sorted(book_dir.glob("*.chocomufin.xml")):
            pages.append(read_alto(alto, key, book_dir.name))
    return pages


def split_books(pages: list[Page], fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Attribue chaque livre à une partition, par tirage stratifié par jeu de données.

    Les livres sont triés par nombre de lignes puis tirés avec une graine fixe ;
    chaque jeu de données contribue au moins un livre au test lorsqu'il en a
    trois ou plus.

    Args:
        pages: Pages du corpus.
        fractions: Part visée (en lignes) de chaque partition hors « train ».
        seed: Graine du tirage.

    Returns:
        Dictionnaire livre → partition (« train », « val » ou « test »).

    Example:
        >>> split_books(pages, {"test": 0.15, "val": 0.15}, 42)["bpt6k8710026t"]
        'train'
    """
    rng = random.Random(seed)
    lines_per_book = Counter()
    dataset_of = {}
    for p in pages:
        lines_per_book[p.book] += len(p.lines)
        dataset_of[p.book] = p.dataset
    assignment = {}
    for ds in sorted(set(dataset_of.values())):
        books = sorted(b for b in lines_per_book if dataset_of[b] == ds)
        rng.shuffle(books)
        total = sum(lines_per_book[b] for b in books)
        filled = Counter()
        for b in books:
            for part in ["test", "val"]:
                if filled[part] < fractions[part] * total and len(books) >= 3:
                    assignment[b] = part
                    filled[part] += lines_per_book[b]
                    break
            else:
                assignment[b] = "train"
    return assignment


def sha256_file(path: Path) -> str:
    """Empreinte SHA-256 d'un fichier.

    Args:
        path: Fichier à hacher.

    Returns:
        Empreinte hexadécimale.

    Example:
        >>> len(sha256_file(Path("config.py")))
        64
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_digest(files: list[Path], base: Path) -> tuple[str, list[str]]:
    """Empreinte d'une partition : SHA-256 de la liste triée « empreinte  chemin ».

    Args:
        files: Fichiers de la partition (images et XML).
        base: Dossier de référence pour les chemins relatifs.

    Returns:
        L'empreinte globale et les lignes au format `sha256sum`.

    Example:
        >>> digest, lines = split_digest([Path("a.xml")], Path("."))
    """
    # Chemins normalisés : « corpus/test/../../external/x.jpg » et « external/x.jpg »
    # désignent le même fichier et doivent donner la même empreinte.
    base = base.resolve()
    lines = sorted(f"{sha256_file(f)}  {f.resolve().relative_to(base).as_posix()}" for f in files)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), lines


def write_alto_harmonized(page: Page, out: Path, region_classes: dict[str, str] | None = None,
                          line_class: str | None = None) -> None:
    """Réécrit l'ALTO d'une page avec les transcriptions harmonisées.

    L'image n'est pas copiée : `fileName` pointe vers l'original par un chemin
    relatif au nouveau fichier, ce que Kraken sait résoudre. Les étiquettes de
    zones et de lignes peuvent être regroupées pour l'entraînement du segmenteur.

    Args:
        page: Page lue par `read_alto`.
        out: Chemin du nouveau fichier ALTO.
        region_classes: Type SegmOnto → classe de zone (types absents : zone ignorée).
        line_class: Classe unique donnée à toutes les lignes.

    Example:
        >>> write_alto_harmonized(page, Path("data/corpus/train/x.xml"))
    """
    import os
    ET.register_namespace("", ALTO_NS["a"])
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(page.alto)
    root = tree.getroot()
    fn = root.find(".//a:sourceImageInformation/a:fileName", ALTO_NS)
    fn.text = os.path.relpath(page.image.resolve(), out.parent.resolve())
    for s in root.iterfind(".//a:String", ALTO_NS):
        s.set("CONTENT", harmonize(s.get("CONTENT", "")))
    for tag in root.iterfind(".//a:Tags/a:OtherTag", ALTO_NS):
        label = tag.get("LABEL", "").split("#")[0]
        if region_classes is not None and tag.get("ID", "").startswith("BT"):
            tag.set("LABEL", region_classes.get(label, "autre"))
        elif line_class is not None and tag.get("ID", "").startswith("LT"):
            tag.set("LABEL", line_class)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)


def line_stats(lines: list[Line]) -> dict:
    """Statistiques descriptives d'un ensemble de lignes.

    Args:
        lines: Lignes de vérité terrain.

    Returns:
        Nombre de lignes et de caractères, longueur moyenne, taux d'abréviation
        (part des caractères qui sont des signes d'abréviation ou des diacritiques
        combinants), part des lignes contenant au moins une abréviation.

    Example:
        >>> line_stats(page.lines)["lignes"]
        26
    """
    chars = sum(len(unicodedata.normalize("NFD", l.text)) for l in lines)
    abbr = sum(sum(c in ABBREVIATION_MARKS for c in unicodedata.normalize("NFD", l.text)) for l in lines)
    with_abbr = sum(any(c in ABBREVIATION_MARKS for c in unicodedata.normalize("NFD", l.text)) for l in lines)
    return {
        "lignes": len(lines),
        "caracteres": chars,
        "longueur_moyenne": round(chars / len(lines), 1) if lines else 0,
        "taux_abreviation": round(abbr / chars, 4) if chars else 0,
        "lignes_avec_abreviation": round(with_abbr / len(lines), 3) if lines else 0,
    }


def write_manifest(lines: list[Line], path: Path) -> None:
    """Écrit les lignes au format JSONL (une ligne JSON par ligne de texte).

    Args:
        lines: Lignes à écrire.
        path: Fichier de sortie.

    Example:
        >>> write_manifest(page.lines, Path("data/corpus/lines_train.jsonl"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(asdict(l), ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[Line]:
    """Relit un manifeste JSONL écrit par `write_manifest`.

    Args:
        path: Fichier JSONL.

    Returns:
        Les lignes.

    Example:
        >>> lines = read_manifest(Path("data/corpus/lines_val.jsonl"))
    """
    with path.open(encoding="utf-8") as f:
        return [Line(**json.loads(l)) for l in f if l.strip()]


def group_by_page(lines: list[Line]) -> dict[tuple[str, str], list[Line]]:
    """Regroupe des lignes par (livre, page).

    Args:
        lines: Lignes.

    Returns:
        Dictionnaire (livre, page) → lignes, dans l'ordre d'origine.

    Example:
        >>> pages = group_by_page(read_manifest(Path("data/corpus/lines_val.jsonl")))
    """
    out = defaultdict(list)
    for l in lines:
        out[(l.book, l.page)].append(l)
    return dict(out)
