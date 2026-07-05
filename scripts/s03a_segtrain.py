"""Étape 3a — Fine-tuning du segmenteur de mise en page (Kraken `ketos segtrain`).

Point de départ : le modèle `blla` fourni avec Kraken. Les lignes gardent sa
classe « default » (détection des lignes de base réutilisée) ; on ajoute les
classes de zones « texte », « illustration », « paratexte » (`--resize union`).
Un premier essai avec `--resize new` (têtes réinitialisées) ne détectait plus
aucune ligne après 5 époques : voir experiments/journal.jsonl.

Usage :
    python -m scripts.s03a_segtrain
"""
from __future__ import annotations

import subprocess
import time
from importlib import resources

import config
from src.repro import fixer_seeds, log_experiment


def main() -> None:
    """Lance `ketos segtrain` et convertit le meilleur point de contrôle."""
    fixer_seeds(config.SEED)
    base = resources.files("kraken").joinpath("blla.mlmodel")
    out_dir = config.MODELS / "kraken" / "segmentation"
    log_path = config.EXPERIMENTS / "logs" / "segtrain.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    st = config.SEGTRAIN
    cmd = ["ketos", "--workers", "0", "-d", st["device"], "-s", str(config.SEED), "segtrain",
           "-f", "alto", "-i", str(base), "--resize", st["resize"],
           "-q", "early", "--lag", str(st["lag"]), "--min-epochs", str(st["epochs_min"]), "-N", str(st["epochs_max"]),
           "--augment", "-o", str(out_dir),
           "-t", str(config.CORPUS / "train.txt"), "-e", str(config.CORPUS / "val.txt")]
    t0 = time.time()
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    ckpts = {int(p.stem.split("_")[1].split("-")[0]): float(p.stem.split("-")[1])
             for p in out_dir.glob("checkpoint_*-*.ckpt")}
    best = sorted(out_dir.glob("best_*.safetensors"), key=lambda p: float(p.stem.split("_")[1]))[-1]
    log_experiment(config.EXPERIMENTS / "journal.jsonl", "segtrain_layout",
                   {"base": "kraken blla.mlmodel", **st, "commande": " ".join(cmd),
                    "classes_zones": sorted(set(config.REGION_CLASSES.values())), "classe_lignes": config.LINE_CLASS},
                   {"val_mean_iu_par_epoque": dict(sorted(ckpts.items())), "meilleur": best.name},
                   [str(best.relative_to(config.ROOT))], time.time() - t0)
    print("Meilleur segmenteur :", best)


if __name__ == "__main__":
    main()
