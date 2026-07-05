# Conventions de transcription

Ces règles valent pour **toute** transcription du projet : corpus d'entraînement harmonisé,
sorties du modèle, annotations humaines (IAA, document cible). Une règle modifiée ici doit l'être
aussi dans `src/corpus.py` (`HARMONIZE`).

## 1. Niveau de transcription

**Graphématique**, selon les recommandations CREMMALAB (<https://cremmalab.hypotheses.org>),
suivies par le jeu GalliCorpora « incunables du XVe siècle », qui fournit les trois quarts des
données d'entraînement :

- on transcrit les **graphèmes** imprimés, pas leurs variantes de forme (allographes) ;
- on **ne développe pas** les abréviations : on garde le signe ;
- on **ne modernise pas** l'orthographe, la ponctuation ni la segmentation en mots ;
- on **ne corrige pas** les coquilles de l'imprimeur.

## 2. Règles

| Cas | Règle | Exemple imprimé → transcription |
|---|---|---|
| s long `ſ` | transcrit `s` (allographe de *s*) | `noſtre` → `nostre` |
| r rond `ꝛ` | transcrit `r` | |
| u / v, i / j | tels qu'imprimés, sans régularisation | `enuoye`, `iour` |
| *et* tironien | `⁊` (U+204A) | |
| esperluette | `&` si imprimée telle quelle | |
| tilde ou macron d'abréviation | lettre + tilde combinant U+0303, quelle que soit la forme du trait | `cõpter`, `ũg`, `q̃` |
| lettre suscrite | lettre + lettre combinante suscrite | `ͥ` (U+0365), `ᷤ` (U+1DE4) |
| *par*, *pro*, *que*, *con*, *-us* | `ꝑ`, `ꝓ`, `ꝗ`/`q̃`, `ꝯ`, `ꝰ` | `ꝑ mer` |
| pied-de-mouche | `¶` (les variantes `⁋` sont ramenées à `¶`) | |
| coupure en fin de ligne | `-` si un trait est imprimé (les variantes `¬`, `=` sont ramenées à `-`), rien sinon | `fu-` / `rent` |
| capitales | telles qu'imprimées | |
| ponctuation | telle qu'imprimée (`.` `:` `/` `;`) | |
| espaces | une espace entre deux mots séparés à l'impression ; mots soudés laissés soudés | `amonseigneur` |
| lettre illisible (tache, déchirure) | `?` par lettre | `fu?ent` |
| passage détruit | `⟦…⟧` | |
| ligne qui n'est pas du texte (gravure, ornement) | pas de transcription (ligne supprimée) | |
| encodage | Unicode NFC à l'écriture ; les comparaisons se font en NFC | |

## 3. Harmonisation des deux jeux d'entraînement

| Différence | Jeu « incunables 15e » | Jeu « gothiques 16e » | Choix du projet |
|---|---|---|---|
| s long | `s` | `ſ` (1 867 occurrences) | `s` |
| coupure de fin de ligne | `-` | `¬` (258) | `-` |
| pied-de-mouche | `¶` et `⁋` | `¶` | `¶` |

Sans cette harmonisation, le modèle apprendrait deux sorties différentes pour un même glyphe selon
le livre, et le CER compterait comme erreur une simple différence de convention.

## 4. Forme normalisée (pour le Volet 2)

Le data contract fournit en plus, pour chaque ligne, un champ `text_normalized` (`src/normalize_text.py`) :
abréviations à valeur fixe développées (`⁊` → *et*, `ꝑ` → *par*, `ꝰ` → *us*…), tilde nasal
développé (`cõpter` → *compter*, `ũg` → *ung*, `ũn` → *un*), pieds-de-mouche retirés. Ce n'est
pas une normalisation orthographique (elle relève du Volet 2) et la transcription de référence
reste le champ `text`.

## 5. Procédure d'annotation humaine (IAA, document cible)

1. Transcrire à partir de **l'image seule**, sans voir la sortie du modèle ni la transcription de
   l'autre annotateur ni la vérité terrain.
2. Un fichier par ligne, une seule ligne de texte, en UTF-8.
3. En cas de doute sur une lettre, `?` plutôt qu'une conjecture.
4. Noter dans `annotations/<campagne>/remarques.md` les cas qui ont posé problème : ils servent à
   préciser ces conventions.
