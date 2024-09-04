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