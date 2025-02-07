"""
auteur:Alexandre
date:2024/09/01
"""

import os
import params
import numpy as np
import pandas as pd
import joblib as jb
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def load_data(
    file_name
):

    """
    ### Fonction de chargement des données
    """
    
    path_to_load = os.path.join(params.datasets_raw_path, file_name)

    if os.path.exists(path_to_load):

        data = pd.read_csv(
            path_to_load, 
            sep=";", 
            decimal=",",
            low_memory=False
        )

    else :

        print(f"The file '{file_name}' does not seem to exist in the folder '{params.datasets_raw_path}'")

    return data


def prepare(
    df,
    name,
    encodeur=None
):
    
    """
    ### Fonction de préprocessing n°1
    
    _Préparation du dataset afin d'obtenir un tableau présentant pour chaque bureau de vote le résultats de chaque nuance politique._ \n
    
    **input :** \n
        - df : pandas DataFrame d'origine.
        - name : nom pour l'enregistrement du fichier de sortie.
        - encodeur : si diff de None utilisé pour créer l'identifiant du bureau de vote.

    **return :** \n
        - df : pandas Dataframe après cette phase de préprocessing
        - encodeur : encodeur utilisé pour créer l'identifiant du bureau de vote
    """

    process_dataframe = os.path.join(params.datasets_pp_path,f"df-{name}.csv")

    if os.path.exists(process_dataframe):

        print("Loading the already preprocessed dataset !")
        df = pd.read_csv(process_dataframe)

        return df, None
    
    else:
        
        print("We need to preprocessed this dataset !")
        # Drop les colonnes vides
        df_1 = df.drop(columns=df.keys()[df.isna().sum() >= df.shape[0]])

        # Drop des colonnes inutile à l'analyse
        to_drop = []
        for key in df_1.keys():
            if key[0] == "%":
                to_drop.append(key)
            elif key[:3] == "Nom":
                to_drop.append(key)
            elif key[:3] == "Elu":
                to_drop.append(key)
            elif key[:6] == "Prénom":
                to_drop.append(key)
            elif key[:6] == "Numéro":
                to_drop.append(key)
        df_2 = df_1.drop(columns=to_drop)

        # Compile les colonnes répétées
        df_3 = pd.concat(
                [df_2[df_2.keys()[11::3]].melt().rename(columns={"value":"Nuance"})["Nuance"],
                df_2[df_2.keys()[12::3]].melt().rename(columns={"value":"Sex"})["Sex"],
                df_2[df_2.keys()[13::3]].melt().rename(columns={"value":"Voix"})["Voix"]],
                axis=1, names=["Nuance","Sex","Voix"]
            )

        # Répète les colonnes non-répétées
        to_concat = []
        for i in df_2.keys():
            if i[:6] == "Nuance":
                to_concat.append(df_2[df_2.keys()[:11]])
        df_4 = pd.concat(to_concat).reset_index(drop=True)

        # Création d'un code d'identification pour chaque bureau de vote
        if encodeur == None:
            bv_encodeur = LabelEncoder().fit(df_4["Code commune"] + df_4["Code département"] + df_4["Code BV"])
            df_code_id_bv = pd.DataFrame(
                bv_encodeur.transform(df_4["Code commune"] + df_4["Code département"] + df_4["Code BV"]),
                columns=["Code_id_bv"]
                )
        else:
            bv_encodeur = encodeur
            df_code_id_bv = pd.DataFrame(
                bv_encodeur.transform(df_4["Code commune"] + df_4["Code département"] + df_4["Code BV"]),
                columns=["Code_id_bv"]
                )

        # Réunification des trois parties du dataset de base
        df_5 = pd.concat(
            [df_3,df_4,df_code_id_bv],
            axis=1
            )
        Nuances = df_5["Nuance"].unique()

        # Compilation des résultats de toutes les nuances politiques pour chaque bureau de vote
        df_6 = pd.DataFrame([], columns=np.concatenate([Nuances,["Code_id_bv"]], axis=0))

        for nuance in tqdm(Nuances, desc="Processing dataset "):

            df_ = df_5[df_5["Nuance"] == nuance][["Code_id_bv","Voix"]]

            results = []

            for id_bv in df_5["Code_id_bv"].unique():

                results.append(df_[df_["Code_id_bv"] == id_bv]["Voix"].sum().item())
            
            df_nuance = pd.DataFrame(np.array(results), columns=[nuance])
            
            df_6[nuance] = df_nuance

        df_6["Code_id_bv"] = df_5["Code_id_bv"].unique()

        df_7 = pd.concat([df_6,df_5.iloc[:df_6.shape[0]][df_5.keys()[11]]], axis=1)

        df_8 = pd.DataFrame(df_7.to_numpy(), columns=df_7.keys().fillna("autre"))

        # Enregistrement du dataset après preprocessing
        df_8.to_csv(process_dataframe, index=False)

        return df_8, bv_encodeur

def clear(
    df_1,
    df_2,
    name1,
    name2
):
    """
    ### Fonction de préprocessing n°2
    
    _Comparaison des DataFrames du 1er et 2nd tour afin d'enlever les bureaux de votes n'ayant pas participer au second tour._ \n
    """

    final_dataframe1 = os.path.join(params.datasets_pp_path,f"DF_{name1}.csv")
    final_dataframe2 = os.path.join(params.datasets_pp_path,f"DF_{name2}.csv")

    if os.path.exists(final_dataframe1) and os.path.exists(final_dataframe2):

        print("The cleaning phase has already been completed !")
        df_1ff = pd.read_csv(final_dataframe1)
        df_2ff = pd.read_csv(final_dataframe2)
    
    else:

        print("We need to clean up this dataset !")
        # Récupération des bv commun entre le 1er et le 2nd tour (on enlève ceux qui ont élu dès le premier tour !)
        print(f"\nIl y a {(1 - df_2.shape[0]/df_1.shape[0]) *100:.3f}% de bv disparues entre les 2 tours.")
        comm_bv = np.intersect1d(df_1["Code_id_bv"],df_2["Code_id_bv"])

        def find_common(bv):
            return df_1[df_1["Code_id_bv"] == bv].index[0]

        # to_analyse = [df_1[df_1["Code_id_bv"] == bv].index[0] for bv in tqdm(comm_bv)] # Version lente
        to_analyse = jb.Parallel(n_jobs=-1)(jb.delayed(find_common)(bv) for bv in tqdm(comm_bv, desc="Bureaux de vote")) # Version rapide

        # Vérification qu'on a bien les mêmes bv sur les mêmes lignes
        print((df_1.iloc[to_analyse]["Code_id_bv"].to_numpy() == df_2["Code_id_bv"].to_numpy()).sum() / df_2.shape[0] * 100,"% de correspondance entre les bv du 1er et 2nd tour !")

        # On enlève les éléments inutiles
        df_1f = df_1.iloc[to_analyse].drop(columns="Code_id_bv").reset_index(drop=True).drop(columns=df_1.keys()[df_1.sum() == 0][0])
        df_2f = df_2.drop(columns="Code_id_bv").reset_index(drop=True).drop(columns=df_2.keys()[df_2.sum() == 0][0])

        # On retire aussi les bureauw de votes ne présentant aucun inscrit (et oui il y en a !)
        print(f"\nIl y a {df_1f[df_1f.sum(axis=1) == 0].index.shape[0]} bureaux de votes ne pésentant aucun votants au premier tours")
        print(f"Il y a {df_2f[df_2f.sum(axis=1) == 0].index.shape[0]} bureaux de votes ne pésentant aucun votants au second tours")
        print(f"""\n
        1er tour (bv id) : {df_1f[df_1f.sum(axis=1) == 0].index}
        2nd tour (bv id) : {df_2f[df_2f.sum(axis=1) == 0].index}
        """)

        to_drop = np.concatenate(
            [
                df_1f[df_1f["Exprimés"] == 0].index.to_numpy(),
                df_2f[df_2f["Exprimés"] == 0].index.to_numpy()
            ]
        )
        
        df_1ff = df_1f.drop(index=to_drop)
        df_2ff = df_2f.drop(index=to_drop)

        # Sauvegrade du Dataset utilisable
        df_1ff.to_csv(final_dataframe1, index=False)
        df_2ff.to_csv(final_dataframe2, index=False)

    return df_1ff, df_2ff

def prepare_supervised(
        df
):
    """
    ### Fonction de préparation pour l'apprentissage supervisée
    
    _Faible mise en forme des données du second tour mettant en avant les différents paramètres utilisable pour chaque bureau de vote._
    """

    df_1 = df.drop(columns=df.keys()[df.isna().sum() >= df.shape[0]])

    # Drop des colonnes inutile à l'analyse
    to_drop = []
    for key in df_1.keys():
        if key[0] == "%":
            to_drop.append(key)
        elif key[:3] == "Nom":
            to_drop.append(key)
        # elif key[:3] == "Elu":
        #     to_drop.append(key)
        elif key[:6] == "Prénom":
            to_drop.append(key)
        elif key[:6] == "Numéro":
            to_drop.append(key)
        elif key[:7] == "Libellé":
            to_drop.append(key)
        elif key[:4] == "Sexe":
            to_drop.append(key)
        # elif key[:6] == "Nuance":
        #     to_drop.append(key)
        elif key[:4] == "Voix":
            to_drop.append(key)

    df_2 = df_1.drop(columns=to_drop)

    df_3 = df_2.drop(index=df_2[df_2["Exprimés"] == 0].index)

    df_4 = df_3[df_3.keys()[:9]].reset_index(drop=True)

    Nuances = df_3[df_3.keys()[9::2]].melt()["value"]
    Elected = df_3[df_3.keys()[10::2]].melt()["value"]

    Gagnants = Nuances[Elected[Elected == "élu"].index].reset_index(drop=True)

    return df_4, Gagnants