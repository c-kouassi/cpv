"""Extraction des pages du PDF Gallica en images PNG (sans perte)."""
from __future__ import annotations

from pathlib import Path

import pymupdf


def extract_pages(pdf_path: Path, out_dir: Path, dpi: int = 300) -> list[Path]:
    """Rend chaque page du PDF en image PNG.

    PNG et non JPG : la compression JPG ajoute des artefacts autour des traits,
    c'est-à-dire précisément le bruit que l'on cherche ensuite à retirer.

    Args:
        pdf_path: PDF Gallica.
        out_dir: Dossier de sortie (créé au besoin).
        dpi: Résolution de rendu.

    Returns:
        Chemins des images `page_001.png`, `page_002.png`…

    Example:
        >>> extract_pages(Path("data/raw/document.pdf"), Path("data/images"), 300)[0].name
        'page_001.png'
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with pymupdf.open(pdf_path) as pdf:
        for i, page in enumerate(pdf, start=1):
            path = out_dir / f"page_{i:03d}.png"
            page.get_pixmap(dpi=dpi).save(path)
            paths.append(path)
    return paths
