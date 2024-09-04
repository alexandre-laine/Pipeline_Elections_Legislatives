# Pipeline_Elections_Legislatives
 
 _Projet DESU de Data Sciences (Aix-Marseille Université) visant à étudier les résultats des élections législatives anticipées de 2024. Développé par Alexandre Lainé sous la supervision de Laurent Perrinet._

## MAJ 2024/09/04

Question "scientifique" : En se basant sur les résultats des deux tours d'une élection sur un certain nombre de bureaux de vote, est-il possible de prédire les résultats du deuxème tour sur un autre ensemble de bureaux de vote uniquement à partir des résultats du premier tour ?

Questionnement : S'il existe un règle permettant de lier les résultats du premier tour à ceux du second tour de façon général, est-ce qu'il est possible de soutenir qu'un vote est un choix personnel et libre ?

Mise en place de la pipeline d'analyse avec :
- Préprocessing : Mise en forme des données.
- Visualisation : Première exploration des données et test avec matplotlib/seaborn/plotly.
- Modèlisation Transfert :  Reproduction plutôt fidèle du modèle et de la méthode utilisée pour prédire les reports de vois lors du second tour des élections présidentielles.
- Autoencodeur : Test de l'autoencodeur parce que je n'ai jamais utilisé ni fait ce type de réseau et je pense qu'il peut peut être nous mettre en avant des éléments sur lesquels je dois m'attarder !

# modèle de transfert de vote

Tout d'abord, les [règles du jeu](https://www.service-public.fr/particuliers/vosdroits/F1943) sont les suivantes:

Le mode de scrutin est un scrutin majoritaire à 2 tours:
- Pour être élu au 1er tour, un candidat doit obtenir : Plus de 50 % des suffrages exprimésEnsemble des bulletins de vote déposés dans l'urne, déduction faite des votes blancs (absence de bulletin de vote ou bulletin de vote sans indication) et des votes nuls (bulletins de vote déchirés ou annotés)    Et un nombre de voix au moins égal à 25 % du nombre des électeurs inscrits. Si aucun candidat n'est élu dès le 1er tour, un 2d tour est organisé une semaine plus tard.
- Seuls certains candidats peuvent se présenter au 2d tour :  Les 2 candidats qui sont arrivés en tête + Les candidats suivants, à condition d'avoir obtenu un nombre de voix au moins égal à 12,5 % du nombre des électeurs inscrits. Au 2d tour, le candidat qui obtient le plus grand nombre de voix est élu. En cas d'égalité, le plus âgé des candidats est élu.

C'est donc un peu différent de https://laurentperrinet.github.io/sciblog/posts/2022-06-08-transfert-de-voix.html avec le modèle

```
class TransfertVoix(torch.nn.Module):
    def __init__(self, N_1er, N_2eme):
        super(TransfertVoix, self).__init__()
        self.lin = torch.nn.Linear(N_2eme, N_1er, bias=False)

    def forward(self, p_1):
        M = torch.softmax(self.lin.weight, axis=1)
        p_2_pred = torch.matmul(p_1, M)
        return p_2_pred
```
où `N_2eme` correspondait au nombre de choix au deuxieme tour. Pour chaque bureau de vote, il y a au moins deux (on peut éliminer les bureaux de vote qui passe au premier tour, ça n'apprend rien) et ceux qui sont au dessus d'un score au moins égal à 12,5 %. Et il y a les désistements et alliances (ce qui nous permettra de savoir si qui se serait pasé sans ces desistements (= une cata ?).

Il faut donc trouver un modèle pour lequel on clampe `p_2_pred` à zero pour les candidats qui ne sont pas au second tour... reste à savoir comment faire...
