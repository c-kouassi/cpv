"""Étape 1 — Corpus : téléchargement, harmonisation, partition par livre, scellement du test.

Usage :
    python -m scripts.s01_prepare_corpus

Produit :
    data/corpus/{train,val,test}/*.xml   ALTO harmonisés (images : chemins relatifs vers data/external)
    data/corpus/lines_{split}.jsonl      une ligne JSON par ligne de texte
    data/corpus/SHA256SUMS_{split}.txt   empreintes des fichiers de chaque partition
    results/corpus/*.csv, figures        description du corpus
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter

import config
from src.corpus import (line_stats, load_dataset, split_books, split_digest,
                        write_alto_harmonized, write_manifest)
from src.repro import fixer_seeds, log_experiment


def fetch_datasets() -> None:
    """Clone chaque jeu de données au commit figé dans `config.DATASETS`."""
    config.EXTERNAL.mkdir(parents=True, exist_ok=True)
    for d in config.DATASETS.values():
        target = config.EXTERNAL / d["dir"]
        if not target.exists():
            subprocess.run(["git", "clone", "-q", d["repo"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "-q", d["commit"]], check=True)


def main() -> None:
    """Prépare le corpus et écrit ses statistiques."""
    t0 = time.time()
    fixer_seeds(config.SEED)
    fetch_datasets()

    pages = []
    for key, d in config.DATASETS.items():
        skip = {b for b, keep in config.DUPLICATE_BOOKS.items() if keep != key}
        pages += load_dataset(config.EXTERNAL / d["dir"], key, skip)
    # Non-contamination : le document cible ne doit figurer dans aucune partition.
    if any(config.TARGET_ID in p.book or config.TARGET_ID in str(p.image) for p in pages):
        raise RuntimeError(f"Le document cible {config.TARGET_ID} figure dans le corpus d'entraînement.")
    assignment = split_books(pages, config.SPLIT_FRACTIONS, config.SEED)

    import pandas as pd
    out_stats = config.RESULTS / "corpus"
    out_stats.mkdir(parents=True, exist_ok=True)

    digests, summary = {}, []
    for split in ["train", "val", "test"]:
        split_pages = [p for p in pages if assignment[p.book] == split]
        files = []
        for p in split_pages:
            out = config.CORPUS / split / f"{p.book}__{p.name}.xml"
            write_alto_harmonized(p, out, config.REGION_CLASSES, config.LINE_CLASS)
            files += [out, p.image]
        lines = [l for p in split_pages for l in p.lines]
        write_manifest(lines, config.CORPUS / f"lines_{split}.jsonl")
        digest, sums = split_digest(files, config.DATA)
        (config.CORPUS / f"SHA256SUMS_{split}.txt").write_text("\n".join(sums) + "\n")
        digests[split] = digest
        st = line_stats(lines)
        summary.append({"partition": split, "livres": len({p.book for p in split_pages}),
                        "pages": len(split_pages), **st, "sha256": digest})

    # Description par livre (distribution du corpus)
    per_book = []
    for book in sorted({p.book for p in pages}):
        bp = [p for p in pages if p.book == book]
        ds = bp[0].dataset
        per_book.append({"livre": book, "jeu": ds, "siecle": config.DATASETS[ds]["siecle"],
                         "partition": assignment[book], "pages": len(bp),
                         **line_stats([l for p in bp for l in p.lines])})
    pd.DataFrame(per_book).to_csv(out_stats / "livres.csv", index=False)
    pd.DataFrame(summary).to_csv(out_stats / "partitions.csv", index=False)
    regions = Counter((assignment[p.book], r["type"]) for p in pages for r in p.regions)
    pd.Series(regions).unstack(0).fillna(0).astype(int).to_csv(out_stats / "regions.csv")
    chars = Counter(c for p in pages for l in p.lines for c in l.text)
    pd.DataFrame(sorted(chars.items(), key=lambda kv: -kv[1]), columns=["caractere", "occurrences"]) \
        .assign(code=lambda d: d.caractere.map(lambda c: f"U+{ord(c):04X}")) \
        .to_csv(out_stats / "caracteres.csv", index=False)
    (out_stats / "partition_livres.json").write_text(json.dumps(assignment, indent=1, sort_keys=True))
    (config.CORPUS / "TEST_SCELLE.sha256").write_text(digests["test"] + "\n")

    _figures(per_book, pages)
    log_experiment(config.EXPERIMENTS / "journal.jsonl", "prepare_corpus",
                   {"seed": config.SEED, "fractions": config.SPLIT_FRACTIONS,
                    "datasets": {k: v["commit"] for k, v in config.DATASETS.items()}},
                   {"partitions": summary},
                   ["data/corpus", "results/corpus"], time.time() - t0)
    print(pd.DataFrame(summary).drop(columns="sha256").to_string(index=False))


def _figures(per_book: list[dict], pages) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import unicodedata
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_book).sort_values(["partition", "lignes"])
    colors = {"train": "#4a78b5", "val": "#e0a03a", "test": "#c4473a"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.barh(df.livre + " (" + df.siecle + ")", df.lignes, color=df.partition.map(colors))
    a1.set_xlabel("lignes"); a1.set_title("Lignes par livre (couleur = partition)")
    for k, c in colors.items():
        a1.barh([], [], color=c, label=k)
    a1.legend()
    lengths = [len(unicodedata.normalize("NFD", l.text)) for p in pages for l in p.lines]
    a2.hist(lengths, bins=40, color="#4a78b5")
    a2.set_xlabel("caractères par ligne"); a2.set_title("Longueur des lignes")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "corpus_distribution.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
