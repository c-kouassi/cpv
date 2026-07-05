"""Configuration centrale : chemins, corpus, partition, paramètres et seuils.

Toutes les valeurs utilisées par `src/` et `scripts/` sont définies ici, une
seule fois. Les scripts enregistrent la configuration utilisée dans le journal
d'expériences (`experiments/journal.jsonl`).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42

# ── Chemins ──────────────────────────────────────────────────────────────────
DATA = ROOT / "data"
EXTERNAL = DATA / "external"                 # jeux de vérité terrain clonés (non versionnés)
CORPUS = DATA / "corpus"                     # corpus harmonisé et partitionné (généré)
RAW_PDF = DATA / "raw" / "document.pdf"      # PDF Gallica du document cible
IMAGES = DATA / "images"                     # pages du document cible, PNG
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
EXPERIMENTS = ROOT / "experiments"
DATASET_NLP = ROOT / "dataset_nlp"           # livrable pour le Volet 2
SEGMENTATIONS = ROOT / "segmentations"       # PAGE XML du document cible

# ── Corpus d'entraînement (vérité terrain existante, licence libre) ──────────
# Commits figés pour la reproductibilité.
DATASETS = {
    "incunables_15e": {
        "repo": "https://github.com/Gallicorpora/HTR-incunable-15e-siecle",
        "commit": "6e050f9146f584b991fe6288d0590621440b2bf9",
        "dir": "HTR-incunable-15e-siecle",
        "siecle": "XVe",
    },
    "gothiques_16e": {
        "repo": "https://github.com/Gallicorpora/HTR-imprime-gothique-16e-siecle",
        "commit": "35b02f7fed1822ae0dff77366287eb32dc335786",
        "dir": "HTR-imprime-gothique-16e-siecle",
        "siecle": "XVIe",
    },
}
# Livre présent dans les deux jeux : conservé une seule fois (dans le premier).
DUPLICATE_BOOKS = {"bpt6k15260973": "incunables_15e"}

# Regroupement des zones SegmOnto en trois classes de mise en page, et des
# types de lignes en un seul : le segmenteur apprend « où est le texte
# principal, où sont les illustrations, où est le paratexte ».
REGION_CLASSES = {
    "MainZone": "texte", "TitlePageZone": "texte",
    "GraphicZone": "illustration", "DropCapitalZone": "illustration", "StampZone": "illustration",
    "RunningTitleZone": "paratexte", "NumberingZone": "paratexte",
    "QuireMarksZone": "paratexte", "MarginTextZone": "paratexte",
}
LINE_CLASS = "default"   # même classe que le modèle blla : la détection des lignes de base est réutilisée

# Partition PAR LIVRE (deux pages d'un même livre partagent fonte, encre et
# papier : les séparer gonflerait artificiellement les scores). Tirage
# reproductible, stratifié par jeu de données.
SPLIT_FRACTIONS = {"test": 0.15, "val": 0.15}

# ── Document cible ───────────────────────────────────────────────────────────
TARGET_ID = "bpt6k8713867f"
TARGET_TITLE = ("Les lettres envoyées du roy nostre sire à nosseigneurs de parlement, "
                "des comptes & de l'hostel de la ville de Paris (Charles VIII, 1494)")
N_PAGES = 34
PAGE_ROLES = {
    1: "notice Gallica", 2: "conditions d'utilisation Gallica", 3: "plat de reliure",
    4: "garde marbrée", 5: "garde marbrée", 6: "feuillet blanc", 7: "feuillet blanc",
    8: "feuillet blanc", 9: "feuillet blanc", 10: "feuillet blanc", 11: "feuillet blanc (estampille)",
    12: "feuillet blanc (cote Réserve Lh5 1421, estampilles)",
    13: "exemplaire A — page de titre (gravure + 4 lignes)",
    14: "exemplaire A — récit de Rapallo", 15: "exemplaire A — suite du récit",
    16: "exemplaire A — lettre de Turin, processions",
    17: "exemplaire B — page de titre (gravure, très endommagée)",
    18: "exemplaire B — récit de Rapallo", 19: "exemplaire B — suite du récit (endommagée)",
    20: "exemplaire B — lettre de Turin, processions (très endommagée)",
    21: "feuillet blanc", 22: "feuillet blanc", 23: "feuillet blanc", 24: "feuillet blanc",
    25: "feuillet blanc", 26: "feuillet blanc", 27: "feuillet blanc",
    28: "garde marbrée", 29: "garde marbrée", 30: "plat de reliure",
    31: "dos", 32: "tranche", 33: "tranche", 34: "tranche",
}
TARGET_TEXT_PAGES = [13, 14, 15, 16, 17, 18, 19, 20]
# Pages portant le même texte (deux exemplaires) : 13↔17, 14↔18, 15↔19, 16↔20.
TARGET_DUPLICATES = [[13, 17], [14, 18], [15, 19], [16, 20]]
DPI = 300

# ── Prétraitement (valeurs par défaut ; l'ablation choisit la variante) ──────
PREPROCESS = {
    "deskew_max_angle": 5.0,      # degrés
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "denoise_h": 7,
    "background_kernel": 51,
    "sauvola_window": 31,
    "sauvola_k": 0.2,
}
PREPROCESS_VARIANT = "brut"       # remplacé par la meilleure variante de validation

# ── HTR ──────────────────────────────────────────────────────────────────────
KRAKEN_CANDIDATES = {
    "catmus_print_large": "10.5281/zenodo.10592716",
    "gallicorpora_prints": "10.5281/zenodo.7410529",
    "latin_incunabula": "10.5281/zenodo.11113737",
    "mcfrench": "10.5281/zenodo.10874058",
}
# Contamination : un modèle entraîné sur nos données de validation ou de test
# aurait un score gonflé, et l'affiner contaminerait le test. Statut établi
# d'après les fiches Zenodo (voir DATA_SOURCES.md).
KRAKEN_CONTAMINATION = {
    "gallicorpora_prints": "contamine",   # « tous les dépôts du projet Gallicorpora » : nos 2 jeux
    "catmus_print_large": "incertain",    # imprimés XVIe-XXIe, sources non listées (auteur GalliCorpora)
    "latin_incunabula": "incertain",      # affiné à partir de CATMuS-Print
    "mcfrench": "propre",                 # manuscrits 1600-2100 : aucun recouvrement possible
}
KRAKEN_BASE = "mcfrench"   # meilleur CER de validation parmi les modèles « propres » (étape 2)
FINETUNE = {
    "epochs_min": 4,
    "epochs_max": 15,
    "lag": 3,
    "batch_size": 16,
    "lrate": 1e-4,
    "augment": False,   # ablation sur 1 époque : sans 0,963, avec 0,961 ; avec la graine 42 → NaN (journal)
    "resize": "union",
}

# Le fine-tuning de reconnaissance se fait sur CPU : sur le GPU Apple (MPS), la
# perte CTC devient NaN pendant l'entraînement complet (constaté le 16/10, voir
# experiments/journal.jsonl), alors que le CPU converge et est plus rapide ici.
FINETUNE_DEVICE = "cpu"
SEGTRAIN = {"device": "mps", "resize": "union", "epochs_min": 8, "epochs_max": 20, "lag": 4}

# ── Évaluation ───────────────────────────────────────────────────────────────
BOOTSTRAP_N = 1000
ALPHA = 0.05
CER_VALIDATION_THRESHOLD = 0.15     # seuil de validation du sujet
CER_EXCELLENCE_THRESHOLD = 0.08
LINE_CORRECT_CER = 0.0              # McNemar : une ligne est « juste » si CER ≤ ce seuil
CALIBRATION_TARGET_CER = 0.05       # confiance calibrée = P(CER de la ligne ≤ 5 %)
REVIEW_PROBABILITY = 0.5            # needs_review si la confiance calibrée est sous ce seuil
SHORT_LINE_CHARS = 5                # lignes très courtes : toujours à relire
IOU_THRESHOLD = 0.5                 # appariement prédiction / référence
ILLUSTRATION_OVERLAP = 0.5          # ligne écartée si ≥ 50 % de sa surface est dans une illustration (évalué sur la validation, étape 6b : non adopté, CER 2,884 % contre 2,869 %)
