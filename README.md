# Pipeline_Elections_Legislatives
 
 _Projet DESU de Data Sciences (Aix-Marseille Université) visant à étudier les résultats des élections législatives anticipées de 2024. Développé par Alexandre Lainé sous la supervision de Laurent Perrinet._

## MAJ 2024/09/09

**Question _scientifique_ :** En se basant sur les résultats des deux tours d'une élection sur un certain nombre de bureaux de vote, est-il possible de prédire les résultats du deuxème tour sur un autre ensemble de bureaux de vote uniquement à partir des résultats du premier tour ?

### Règles du jeu :
Tout d'abord, les [règles du jeu](https://www.service-public.fr/particuliers/vosdroits/F1943) sont les suivantes. Le mode de scrutin est un scrutin majoritaire à 2 tours:
- Pour être élu au 1er tour, un candidat doit obtenir : Plus de 50 % des suffrages exprimésEnsemble des bulletins de vote déposés dans l'urne, déduction faite des votes blancs (absence de bulletin de vote ou bulletin de vote sans indication) et des votes nuls (bulletins de vote déchirés ou annotés), et un nombre de voix au moins égal à 25 % du nombre des électeurs inscrits. Si aucun candidat n'est élu dès le 1er tour, un 2d tour est organisé une semaine plus tard.
- Seuls certains candidats peuvent se présenter au 2d tour :  Les 2 candidats qui sont arrivés en tête + Les candidats suivants, à condition d'avoir obtenu un nombre de voix au moins égal à 12,5 % du nombre des électeurs inscrits. Au 2d tour, le candidat qui obtient le plus grand nombre de voix est élu. En cas d'égalité, le plus âgé des candidats est élu.

### Organisation :

- main : Les notebooks ainsi que les fichiers Python composant la pipeline d'analyse et de visualisation des données et des résultats :
    - install_requirements.py : installation des librairies présentent dans les fichiers .py. **Attention**, les librairies présentent dans les notebooks et notamment _PyTorch_ devront être installées manuellement.
    - Preprocessing : mise en forme des données pour le reste de la pipeline.
    - Visualisation : exploration par la représentation graphique du jeu de donné.
    - Transfert de Voix : Le nerf de la guerre ! Zone d'apprentissage de la matrice permettant de transformer les résultats du premier tour en ceux du second tour. Avec plusieurs tests différents :
        a) Modèle classique développé lors des élections présidentielles de 2022.
        b) Version avec un blocage des poids de la matrice lorsque la nuance n'a pas obtenue plus de 12.5% des voix lors du premier tour.
        c,d,e) Focalisation sur les bureaux de vote avec des face-à-face spécifiqeus lors du second tour.
        f) Regroupement des nuances sous la forme de grandes familles.
- rapport : Une version LaTeX et pdf du rapport à rendre/rendu pour l'obtention du DESU.
- results : Les résultats des différents modèles utilisés.
- datasets : Les jeux de données de cette pipeline qui ont été trouvée sur [le site gouvernemental](https://www.data.gouv.fr/).
- fig : Les figures enregistrées par la pipeline au format pdf.