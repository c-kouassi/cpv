"""Forme normalisée du texte, fournie au Volet 2 (champ `text_normalized` du data contract).

La transcription livrée reste graphématique (CONVENTIONS_TRANSCRIPTION.md) :
abréviations conservées. Cette fonction en dérive une forme lisible par des
outils NLP modernes ; elle change de convention d'écriture, elle ne corrige
pas d'erreurs. La normalisation orthographique complète relève du Volet 2.
"""
from __future__ import annotations

import re
import unicodedata

TILDE = "̃"
VOWELS = "aeiouy"

# Signes d'abréviation à valeur fixe (convention CREMMALAB). Les formes longues d'abord.
ABBREVIATIONS = [
    ("q" + TILDE, "que"),
    ("ꝗ", "que"),
    ("ꝑ", "par"),
    ("ꝓ", "pro"),
    ("ꝯ", "con"),
    ("ꝰ", "us"),
    ("⁊", "et"),
    ("&", "et"),
    ("ſ", "s"),
]


def _expand_tildes(text: str) -> str:
    # Voyelle + tilde = nasale omise : n, ou m devant p, b, m ; rien si un n suit déjà.
    out, chars = [], list(text)
    for i, ch in enumerate(chars):
        if ch != TILDE:
            out.append(ch)
            continue
        prev = out[-1].lower() if out else ""
        nxt = chars[i + 1].lower() if i + 1 < len(chars) else ""
        if not prev or prev not in VOWELS:
            out.append(ch)
        elif nxt != "n":
            out.append("m" if nxt in "pbm" else "n")
    return "".join(out)


def normalize(text: str) -> str:
    """Dérive une forme normalisée d'une transcription graphématique.

    Args:
        text: Transcription (conventions du projet).

    Returns:
        Texte en NFC : abréviations à valeur fixe développées, tildes nasaux
        développés, coupures marquées en fin de ligne recollées, pieds-de-mouche retirés.

    Example:
        >>> normalize("cõpter ⁊ ꝑler")
        'compter et parler'
    """
    text = unicodedata.normalize("NFD", text)
    for src, dst in ABBREVIATIONS:
        text = text.replace(src, dst)
    text = _expand_tildes(text)
    text = text.replace("¶", "")
    text = re.sub(r"(\w)[-=⸗]\n(\w)", r"\1\2", text)
    return unicodedata.normalize("NFC", " ".join(text.split()))
