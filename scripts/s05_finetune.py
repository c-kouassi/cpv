"""Étape 5 — Fine-tuning du modèle de reconnaissance (Kraken, `ketos train`).

Point de départ : le meilleur modèle de la validation sans fine-tuning (étape 2).
Données : ALTO harmonisés de la partition d'apprentissage ; la validation sert à
l'arrêt anticipé et au choix du meilleur point de contrôle. Le test n'intervient pas.

Usage :
    python -m scripts.s05_finetune [--device mps]

Produit :
    models/kraken/htr_ft/best_<précision>.safetensors   meilleur point de contrôle
    models/kraken/charles8_htr_ft.mlmodel               même modèle, format chargé par `src/htr.py`
    results/finetune/courbe.json, figures/finetune_validation.png
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time

import config
from src.repro import fixer_seeds, log_experiment


def learning_curve(log_text: str) -> dict[int, dict]:
    """Extrait la précision de validation par époque du journal de `ketos train`.

    L'affichage de progression de Kraken répète chaque valeur ; on garde la
    dernière valeur lue pour chaque époque.

    Args:
        log_text: Contenu du journal.

    Returns:
        Époque → {"val_accuracy", "val_word_accuracy"}.

    Example:
        >>> learning_curve("stage 0/∞ ... val_accuracy: 0.9 val_word_acc… 0.7")
        {0: {'val_accuracy': 0.9, 'val_word_accuracy': 0.7}}
    """
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", " ".join(log_text.split()))
    out = {}
    # Découpage par segment « stage N/ » (une expression unique avec .*? sur
    # un journal de plusieurs Mo provoque un retour arrière catastrophique).
    for chunk in text.split("stage ")[1:]:
        ep = re.match(r"(\d+)/", chunk)
        acc = re.search(r"val_accuracy:\s*([0-9.]+)", chunk)
        wacc = re.search(r"val_word_acc\S*\s*([0-9.]+)", chunk)
        if ep and acc and wacc:
            out[int(ep.group(1))] = {"val_accuracy": float(acc.group(1)), "val_word_accuracy": float(wacc.group(1))}
    return out


def checkpoint_curve(directory) -> dict[int, float]:
    """Précision de validation par époque, lue dans les noms des points de contrôle.

    `ketos` nomme ses points de contrôle `checkpoint_<époque>-<précision>.ckpt`
    et ne conserve que les plus récents ; hors terminal, il n'écrit pas ses
    métriques dans le journal. Les noms de fichiers sont donc la source fiable.

    Args:
        directory: Dossier de sortie de l'entraînement.

    Returns:
        Époque → précision caractère de validation.

    Example:
        >>> checkpoint_curve(Path("models/kraken/htr_ft"))
        {4: 0.9724, ...}
    """
    out = {}
    for f in directory.glob("checkpoint_*-*.ckpt"):
        ep, acc = f.stem.removeprefix("checkpoint_").split("-")
        out[int(ep)] = float(acc)
    return dict(sorted(out.items()))


def main() -> None:
    """Lance `ketos train`, convertit le meilleur modèle et trace la courbe d'apprentissage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=config.FINETUNE_DEVICE)
    args = parser.parse_args()
    fixer_seeds(config.SEED)
    base_name = json.loads((config.RESULTS / "baseline" / "classement_val.json").read_text())["meilleur"]
    base = config.MODELS / "kraken" / f"{base_name}.mlmodel"
    out_dir = config.MODELS / "kraken" / "htr_ft"
    log_path = config.EXPERIMENTS / "logs" / "finetune_htr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ft = config.FINETUNE
    cmd = ["ketos", "--workers", "0", "-d", args.device, "-s", str(config.SEED), "train",
           "-f", "alto", "-i", str(base), "--resize", ft["resize"],
           "-q", "early", "--lag", str(ft["lag"]), "--min-epochs", str(ft["epochs_min"]), "-N", str(ft["epochs_max"]),
           "-B", str(ft["batch_size"]), "-r", str(ft["lrate"]),
           "--augment" if ft["augment"] else "--no-augment",
           "-o", str(out_dir),
           "-t", str(config.CORPUS / "train.txt"), "-e", str(config.CORPUS / "val.txt")]
    t0 = time.time()
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    duration = time.time() - t0

    best = sorted(out_dir.glob("best_*.safetensors"), key=lambda p: float(p.stem.split("_")[1]))[-1]
    mlmodel = config.MODELS / "kraken" / "charles8_htr_ft.mlmodel"
    subprocess.run(["ketos", "convert", "-o", str(mlmodel), "--weights-format", "coreml", str(best)], check=True)

    curve = learning_curve(log_path.read_text(errors="ignore"))
    (config.RESULTS / "finetune").mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "finetune" / "courbe.json").write_text(json.dumps(curve, indent=1))
    import matplotlib.pyplot as plt
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    ep = sorted(curve)
    plt.figure(figsize=(6, 3.5))
    plt.plot(ep, [1 - curve[e]["val_accuracy"] for e in ep], marker="o", label="CER validation (1 − précision)")
    plt.plot(ep, [1 - curve[e]["val_word_accuracy"] for e in ep], marker="s", label="WER validation")
    plt.xlabel("époque"); plt.ylabel("taux d'erreur"); plt.grid(alpha=.3); plt.legend()
    plt.title(f"Fine-tuning de {base_name}")
    plt.tight_layout(); plt.savefig(config.FIGURES / "finetune_validation.png", dpi=130); plt.close()

    log_experiment(config.EXPERIMENTS / "journal.jsonl", "finetune_htr",
                   {"base": base_name, "device": args.device, **ft, "commande": " ".join(cmd)},
                   {"meilleur_point": best.name, "epoques": len(curve), "courbe": curve},
                   [str(best.relative_to(config.ROOT)), str(mlmodel.relative_to(config.ROOT))], duration)
    print("Meilleur point :", best.name, "— modèle :", mlmodel)


if __name__ == "__main__":
    main()
