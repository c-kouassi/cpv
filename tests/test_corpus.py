"""Corpus : lecture ALTO, harmonisation, partition par livre, empreintes, normalisation."""
from pathlib import Path

import pytest
from PIL import Image

from src import corpus
from src.normalize_text import normalize

ALTO = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Description><sourceImageInformation><fileName>p.jpg</fileName></sourceImageInformation></Description>
  <Tags>
    <OtherTag ID="BT1" LABEL="MainZone#1"/><OtherTag ID="BT2" LABEL="GraphicZone"/>
    <OtherTag ID="BT3" LABEL="RunningTitleZone"/><OtherTag ID="LT1" LABEL="DefaultLine"/>
  </Tags>
  <Layout><Page WIDTH="200" HEIGHT="300"><PrintSpace>
    <TextBlock ID="b1" TAGREFS="BT1"><Shape><Polygon POINTS="0 0 200 0 200 200 0 200"/></Shape>
      <TextLine ID="l1" TAGREFS="LT1" BASELINE="10 40 190 40"><Shape><Polygon POINTS="10 10 190 10 190 45 10 45"/></Shape>
        <String CONTENT="noſtre  ſire ⁋"/></TextLine>
      <TextLine ID="l2" TAGREFS="LT1" BASELINE="10 90 190 90"><Shape><Polygon POINTS="10 60 190 60 190 95 10 95"/></Shape>
        <String CONTENT=""/></TextLine>
    </TextBlock>
    <TextBlock ID="b2" TAGREFS="BT3"><Shape><Polygon POINTS="0 250 200 250 200 300 0 300"/></Shape></TextBlock>
  </PrintSpace></Page></Layout>
</alto>"""


@pytest.fixture
def alto(tmp_path):
    Image.new("L", (200, 300), 255).save(tmp_path / "p.jpg")
    path = tmp_path / "book__p.xml"
    path.write_text(ALTO, encoding="utf-8")
    return path


def test_harmonisation():
    assert corpus.harmonize("noſtre  ſire ⁋") == "nostre sire ¶"
    assert corpus.harmonize("fu¬") == "fu-"


def test_lecture_alto(alto):
    page = corpus.read_alto(alto, "ds", "book")
    assert (page.width, page.height) == (200, 300)
    assert [l.text for l in page.lines] == ["nostre sire ¶"]          # ligne vide écartée
    assert page.lines[0].region_type == "MainZone"
    assert page.lines[0].baseline == [[10, 40], [190, 40]]
    assert {r["type"] for r in page.regions} == {"MainZone", "RunningTitleZone"}


def test_image_manquante(tmp_path):
    (tmp_path / "x.xml").write_text(ALTO, encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        corpus.read_alto(tmp_path / "x.xml", "ds", "book")


def test_reecriture_alto_harmonise(alto, tmp_path):
    page = corpus.read_alto(alto, "ds", "book")
    out = tmp_path / "split" / "book__p.xml"
    corpus.write_alto_harmonized(page, out, {"MainZone": "texte", "RunningTitleZone": "paratexte"}, "ligne")
    again = corpus.read_alto(out, "ds", "book")                        # image retrouvée par chemin relatif
    assert again.lines[0].text == "nostre sire ¶"
    assert {r["type"] for r in again.regions} == {"texte", "paratexte"}


def _pages(n_books=10, ds="a"):
    return [corpus.Page(ds, f"{ds}{b}", "p", Path("x"), Path("x"), 1, 1,
                        lines=[corpus.Line(ds, f"{ds}{b}", "p", str(i), "t", [], [], "MainZone") for i in range(10 + b)])
            for b in range(n_books)]


def test_partition_par_livre_sans_fuite():
    pages = _pages(10, "a") + _pages(4, "b")
    split = corpus.split_books(pages, {"test": 0.15, "val": 0.15}, 42)
    assert set(split) == {p.book for p in pages}
    assert {"train", "val", "test"} <= set(split.values())
    assert split == corpus.split_books(pages, {"test": 0.15, "val": 0.15}, 42)      # reproductible
    for ds in "ab":                                                                  # chaque jeu au test
        assert any(v == "test" for k, v in split.items() if k.startswith(ds))


def test_empreinte_partition(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("1"); b.write_text("2")
    d1, lines = corpus.split_digest([a, b], tmp_path)
    d2, _ = corpus.split_digest([b, a], tmp_path)
    assert d1 == d2 and len(lines) == 2
    b.write_text("3")
    assert corpus.split_digest([a, b], tmp_path)[0] != d1


def test_statistiques():
    st = corpus.line_stats([corpus.Line("a", "b", "p", "1", "cõpter ⁊ ꝑler", [], [], "MainZone")])
    assert st["lignes"] == 1 and st["lignes_avec_abreviation"] == 1.0 and st["taux_abreviation"] > 0


@pytest.mark.parametrize("src,dst", [("cõpter", "compter"), ("cõmencer", "commencer"), ("ũg", "ung"),
                                     ("ũn", "un"), ("⁊ ꝑ terre", "et par terre"), ("q̃ ie", "que ie"),
                                     ("¶ Le roy", "Le roy"), ("noſtre", "nostre")])
def test_forme_normalisee(src, dst):
    assert normalize(src) == dst
