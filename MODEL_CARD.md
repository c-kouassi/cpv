# Model card — `charles8_htr_ft` (reconnaissance) et segmenteur de mise en page

## Modèle de reconnaissance

| | |
|---|---|
| Fichier | `models/kraken/charles8_htr_ft.mlmodel` (poids : `models/kraken/htr_ft/best_0.9737.safetensors`) |
| Architecture | Kraken 7.1.1, réseau VGSL convolutif + LSTM bidirectionnel, décodage CTC, 5,7 M paramètres |
| Point de départ | HTR-United Manu McFrench V3 (DOI 10.5281/zenodo.10874058) |
| Choix du point de départ | meilleur CER de validation sans fine-tuning (8,30 %) parmi les modèles **sans recouvrement documenté** avec le corpus. Gallicorpora+ (6,19 %) a été écarté car entraîné sur nos données (voir `DATA_SOURCES.md`) |
| Données d'entraînement | 6 872 lignes, 134 pages, 10 livres (GalliCorpora incunables XVe s. et gothiques XVIe s.), CC-BY 4.0 |
| Validation (arrêt anticipé) | 1 750 lignes, 3 livres |
| Hyperparamètres | `--resize union`, AdamW, lr 1e-4, lots de 16, sans augmentation, arrêt anticipé (patience 3, min. 4 époques, max. 15), graine 42, CPU |
| Entraînement | 14 époques ; meilleure à l'époque 10 (précision caractère de validation 0,9737) ; ≈ 9 à 10 h de calcul effectif sur CPU Apple (16 h d'horloge, veille de la machine comprise) |
| Conventions de sortie | graphématiques, CREMMALAB (`CONVENTIONS_TRANSCRIPTION.md`) : abréviations non développées, s long transcrit *s*, ⁊, tilde combinant |

### Performances sur le test scellé (3 livres, 1 532 lignes, 54 617 caractères, évalué une seule fois)

| Mesure | Sans fine-tuning (McFrench) | **Affiné** | Seuils du sujet (validation / excellence) |
|---|---|---|---|
| CER, lignes de référence | 10,26 % [9,90 ; 10,62] | **2,76 % [2,59 ; 2,95]** | 15 % / 8 % |
| WER, lignes de référence | 43,47 % [42,19 ; 44,73] | **17,34 % [16,38 ; 18,36]** | 25 % / 15 % |
| Lignes exactes | 108 | **665** | |
| McNemar (lignes exactes) | | b = 24, c = 581, p ≈ 9·10⁻¹⁴⁰ | |
| CER de bout en bout (redressement + segmentation + HTR) | | **3,94 % [3,42 ; 4,56]**, WER 18,71 % | |

Intervalles : bootstrap sur les lignes, N = 1000, 95 %. Empreinte du test : `data/corpus/TEST_SCELLE.sha256`.

Par livre de test : 2,49 % (bpt6k1057722q, XVe), 2,90 % (bpt6k1520377h, XVIe), 3,31 % (bpt6k8713143m, XVe).
Par longueur de ligne : < 10 caractères 20,3 % (46 lignes) ; 10-29 : 5,8 % ; 30-49 : 2,5 % ; ≥ 50 : 5,6 %.

Erreurs résiduelles dominantes : espaces (332 espaces manquantes, 241 en trop sur l'ensemble des erreurs),
trait de coupure lu comme virgule (49), points omis (33), confusions c/t, l/s, l/t, lettre suscrite ᷤ omise (25),
tilde sur *e* omis (14). Les erreurs d'espaces viennent en grande partie de conventions de segmentation en mots
(élisions séparées dans la vérité terrain : « l eschaper »).

### Confiances et `needs_review`

Confiance brute = moyenne des probabilités des caractères décodés. Calibration : régression isotone ajustée sur la
validation, cible P(CER de la ligne ≤ 5 %) (`models/calibration/isotone_val.json`).

| | ECE (test) |
|---|---|
| confiance brute | 0,203 |
| confiance calibrée | **0,051** |

Règle : `needs_review` si confiance calibrée < 0,5, ligne de moins de 5 caractères, ou polygone hors image.
Sur le test : **10,8 %** des lignes signalées ; CER des lignes signalées 9,0 %, des lignes non signalées 2,2 %.

## Segmenteur de mise en page

| | |
|---|---|
| Fichier | `models/kraken/segmentation/best_0.3594.safetensors` |
| Point de départ | modèle `blla` de Kraken 7.1.1 |
| Classes | lignes : `default` (réutilisée du modèle d'origine) ; zones : `texte`, `illustration`, `paratexte` (ajoutées, `--resize union`) |
| Entraînement | 134 pages ALTO (zones SegmOnto regroupées), augmentation, GPU Apple (MPS), 8 époques, meilleure à l'époque 2 (val mean IU 0,359) |

| Validation (40 pages) | blla par défaut | **blla affiné** |
|---|---|---|
| CER de bout en bout (même reconnaisseur) | 33,3 % | **8,1 %** |
| IoU moyenne des lignes | 0,742 | **0,824** |
| Lignes avec IoU > 0,5 | 84,7 % | **97,3 %** |
| F1 de détection des lignes | 0,920 | **0,977** |
| IoU zones texte / illustration / paratexte | 0,905 / 0 / 0 | **0,919 / 0,406 / 0** |

Test scellé (35 pages) : IoU moyenne des lignes **0,836**, 98 % des lignes > 0,5, F1 0,922 ;
IoU zones texte 0,853, illustration 0,822, paratexte 0.

## Limites

- **Paratexte non détecté** (titres courants, signatures, foliotation) : 51 zones seulement à l'apprentissage.
- **Lignes courtes** : CER de 20 % sous 10 caractères.
- **Domaine** : incunables et gothiques français ; aucune page du document cible (Charles VIII, 1494) n'a de
  vérité terrain, son CER n'est donc pas mesuré (voir `scripts/s08_annotations.py`). Sur ce document très
  endommagé, la confiance chute : 72 % des lignes sont signalées `needs_review` (sans le filtre des illustrations).
- **Biais du corpus** : 13 livres sur 16 sont des incunables du XVe s. ; 3 gothiques du XVIe s. ; un seul atelier
  par livre ; pas de manuscrits ; textes majoritairement littéraires et religieux. Le test contient moins
  d'abréviations que l'apprentissage (32 % des lignes contre 53 %) : le CER de test est probablement optimiste
  pour un texte très abrégé.
- **IAA non mesuré** : nécessite deux annotateurs humains (`scripts/s08_annotations.py`).
- Le fine-tuning sur GPU Apple (MPS) a divergé (perte NaN), de même qu'avec augmentation et la graine 42 ;
  le modèle final est entraîné sur CPU, sans augmentation (journal : `experiments/journal.jsonl`).

## Usage prévu

Transcription graphématique d'imprimés français des XVe-XVIe siècles en caractères gothiques, avec relecture
humaine des lignes `needs_review`. Ne convient pas aux manuscrits ni aux imprimés en romain sans évaluation préalable.
