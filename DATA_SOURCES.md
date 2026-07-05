# Sources des données et des modèles

## Corpus d'entraînement, de validation et de test

| Jeu | Dépôt (commit figé) | Contenu | Licence |
|---|---|---|---|
| GalliCorpora — Données HTR incunables du 15e siècle | <https://github.com/Gallicorpora/HTR-incunable-15e-siecle> (`6e050f9146f584b991fe6288d0590621440b2bf9`) | 13 incunables français (1400-1500), 149 pages, 7 608 lignes, ALTO v4 + images Gallica | CC-BY 4.0 (fiche HTR-United ; le dépôt GitHub indique CC0-1.0 : nous retenons la plus contraignante) |
| GalliCorpora — Données imprimés gothiques du 16e siècle | <https://github.com/Gallicorpora/HTR-imprime-gothique-16e-siecle> (`35b02f7fed1822ae0dff77366287eb32dc335786`) | 4 imprimés gothiques (1500-1599), 80 pages, 2 971 lignes, ALTO v4 + images Gallica | CC-BY 4.0 (idem) |

Attribution : Simon Gabay, Ariane Pinche (direction), Noé Leroy (transcription), Kelly Christensen —
projet GalliCorpora. Images : gallica.bnf.fr / Bibliothèque nationale de France.

Traitements appliqués (`scripts/s01_prepare_corpus.py`) :

- versions `.chocomufin.xml` (caractères harmonisés par les auteurs avec ChocoMufin) ;
- harmonisation complémentaire entre les deux jeux (`CONVENTIONS_TRANSCRIPTION.md` § 3) ;
- livre `bpt6k15260973`, présent dans les deux jeux, conservé une seule fois ;
- zones SegmOnto regroupées en trois classes (texte, illustration, paratexte) ;
- partition par livre, graine 42 ; empreintes SHA-256 dans `data/corpus/SHA256SUMS_*.txt`,
  empreinte du test scellé dans `data/corpus/TEST_SCELLE.sha256`.

Les données brutes ne sont pas versionnées (`data/external/`) : le script les clone au commit indiqué.

## Document cible (transcription, non utilisé pour l'entraînement)

*Les lettres envoyées du roy nostre sire à nosseigneurs de parlement, des comptes & de l'hostel de
la ville de Paris* (Charles VIII, septembre 1494), BnF, Réserve Lh⁵ 1421 (Lb²⁸ \*),
<https://gallica.bnf.fr/ark:/12148/bpt6k8713867f>.

Conditions Gallica : réutilisation non commerciale libre et gratuite avec mention
« Source gallica.bnf.fr / Bibliothèque nationale de France ». Le PDF n'est pas redistribué dans le
dépôt (`data/raw/`, non versionné).

Vérification de non-contamination : l'identifiant `bpt6k8713867f` n'apparaît dans aucun des deux jeux
d'entraînement (vérifié dans `scripts/s01_prepare_corpus.py`).

## Modèles pré-entraînés

| Modèle | DOI | Données d'origine | Licence |
|---|---|---|---|
| CATMuS-Print (Large, 2024-01-30) | 10.5281/zenodo.10592716 | imprimés XVIe-XXIe s., multilingue (S. Gabay, T. Clérice) | CC-BY 4.0 |
| Gallicorpora+ ancient prints | 10.5281/zenodo.7410529 | imprimés anciens (GalliCorpora) | voir Zenodo |
| Latin Incunabula and Early Prints | 10.5281/zenodo.11113737 | incunables latins | voir Zenodo |
| HTR-United-Manu_McFrench | 10.5281/zenodo.10874058 | manuscrits français XVIIe-XXIe s. (A. Chagué, T. Clérice) | CC-BY 4.0 |
| Kraken `blla` (segmentation par défaut) | fourni avec kraken 7.1.1 | — | Apache 2.0 (kraken) |

### Recouvrement avec notre corpus (contamination)

Un modèle déjà entraîné sur nos pages de validation ou de test aurait un score gonflé, et
l'affiner contaminerait le test scellé. Statut établi d'après les descriptions Zenodo
(`config.KRAKEN_CONTAMINATION`) :

| Modèle | Données d'entraînement déclarées | Statut |
|---|---|---|
| Gallicorpora+ | « tous les dépôts du projet Gallicorpora », donc nos deux jeux | **contaminé** : écarté |
| CATMuS-Print Large | imprimés XVIe-XXIe s. de plusieurs projets, liste non publiée | incertain |
| Latin Incunabula | incunables de Reichenau, affiné à partir de CATMuS-Print | incertain |
| McFrench | manuscrits 1600-2100 | sans recouvrement possible : **retenu comme base** |

Gallicorpora+ obtenait le meilleur CER de validation sans fine-tuning (6,19 %), ce qui illustre le
risque : sans cette vérification, on aurait retenu le modèle qui avait déjà vu le test.

Tous sont compatibles avec un usage de recherche non commerciale. Ils sont téléchargés par
`kraken get <DOI>` (`src/htr.py`, `fetch_model`).

## Données personnelles

Aucune : documents du XVe et du XVIe siècle, domaine public.
