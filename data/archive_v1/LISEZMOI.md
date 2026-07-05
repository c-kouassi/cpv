# Sorties Kraken de la v1 (archive)

`kraken_v1_pages_014-020.txt` : texte produit par la v1 du projet pour les pages 14 à 20
(la page 13 n'avait pas été transcrite), extrait de l'ancien `results/transcriptions/texte_brut.txt`.

Limites connues, qui rendent ce texte impropre à toute mesure de qualité HTR :
- segmenteur par boîtes (`kraken segment` sans `-bl`) utilisé avec un modèle entraîné sur des lignes de base ;
- modèle HTR-United-Manu_McFrench, entraîné sur des manuscrits du XVIIe au XXIe siècle ;
- images binarisées puis enregistrées en JPG ;
- limites de page perdues (les pages ont été concaténées).

Il ne sert qu'à faire tourner la partie NLP (notebook 05) en attendant les nouvelles transcriptions.
Les pages 1 et 2 de l'ancien fichier ont été retirées : la page 1 avait été retapée à la main
(la sortie Kraken réelle était illisible) et la page 2 est la notice Gallica.
