"""Reproductibilité : graines aléatoires et journal d'expériences."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import random
import subprocess
from pathlib import Path


def fixer_seeds(seed: int = 42) -> None:
    """Fixe les graines de toutes les sources d'aléatoire utilisées.

    Args:
        seed: Graine commune (Python, NumPy, PyTorch).

    Example:
        >>> fixer_seeds(42)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def log_experiment(journal: Path, name: str, params: dict, metrics: dict,
                   artifacts: list[str] | None = None, duration_s: float | None = None) -> dict:
    """Ajoute une entrée au journal d'expériences (`experiments/journal.jsonl`).

    Args:
        journal: Fichier JSONL du journal.
        name: Nom de l'expérience (ex. « baseline_val », « finetune »).
        params: Hyperparamètres et choix de l'expérience.
        metrics: Résultats mesurés.
        artifacts: Fichiers produits (chemins relatifs au dépôt).
        duration_s: Durée en secondes.

    Returns:
        L'entrée écrite.

    Example:
        >>> log_experiment(Path("experiments/journal.jsonl"), "test", {"seed": 42}, {"cer": 0.1})
    """
    entry = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "experience": name,
        "params": params,
        "metrics": metrics,
        "artifacts": artifacts or [],
        "duree_s": round(duration_s, 1) if duration_s is not None else None,
        "git_commit": _git_commit(journal.parent.parent),
        "python": platform.python_version(),
        "plateforme": platform.platform(),
    }
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
