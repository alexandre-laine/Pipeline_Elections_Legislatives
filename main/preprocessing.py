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
    
    path_to_load = os.path.join(params.datasets_path, file_name)

    if os.path.exists(path_to_load):

        data = pd.read_csv(
            path_to_load, 
            sep=";", 
            decimal=",",
            low_memory=False
        )

    else :

        print(f"The file '{file_name}' does not seem to exist in the folder '{params.datasets_path}'")

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

    process_dataframe = os.path.join(params.datasets_path,f"df-{name}.csv")

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
        df_7 = pd.DataFrame(df_6.to_numpy(), columns=df_6.keys().fillna("autre"))

        # Normalisation
        df_8 = df_7[df_7.keys()[:-1]].to_numpy().T / (np.sum(df_7[df_7.keys()[:-1]].to_numpy(), axis=1) + 1)
        df_n = pd.DataFrame(df_8.T, columns=df_7.keys()[:-1]).fillna(0)

        # Dataframe final
        T_df = pd.concat([df_n, df_7["Code_id_bv"]], axis=1)

        # Enregistrement du dataset après preprocessing
        T_df.to_csv(process_dataframe, index=False)

        return T_df, bv_encodeur

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

    final_dataframe1 = os.path.join(params.datasets_path,f"DF_{name1}.csv")
    final_dataframe2 = os.path.join(params.datasets_path,f"DF_{name2}.csv")

    if os.path.exists(final_dataframe1) and os.path.exists(final_dataframe2):

        print("The cleaning phase has already been completed !")
        df_1f = pd.read_csv(final_dataframe1)
        df_2f = pd.read_csv(final_dataframe2)
    
    else:

        print("We need to clean up this dataset !")
        # Récupération des bv commun entre le 1er et le 2nd tour (on enlève ceux qui ont élu dès le premier tour !)
        print(f"Il y a {(1 - df_2.shape[0]/df_1.shape[0]) *100:.3f}% de bv disparues entre les 2 tours.")
        comm_bv = np.intersect1d(df_1["Code_id_bv"],df_2["Code_id_bv"])

        def find_common(bv):
            return df_1[df_1["Code_id_bv"] == bv].index[0]

        # to_analyse = [df_1[df_1["Code_id_bv"] == bv].index[0] for bv in tqdm(comm_bv)] # Version lente
        to_analyse = jb.Parallel(n_jobs=-1)(jb.delayed(find_common)(bv) for bv in tqdm(comm_bv, desc="Bureaux de vote")) # Version rapide

        # Vérification qu'on a bien les mêmes bv sur les mêmes lignes
        print((df_1.iloc[to_analyse]["Code_id_bv"].to_numpy() == df_2["Code_id_bv"].to_numpy()).sum() / df_2.shape[0] * 100,"% de correspondance entre les bv du 1er et 2nd tour !")

        # On enlève les éléments inutiles
        df_1f = df_1.iloc[to_analyse].drop(columns="Code_id_bv").reset_index(drop=True)
        df_2f = df_2.drop(columns="Code_id_bv").reset_index(drop=True)

        # Sauvegrade du Dataset utilisable
        df_1f.to_csv(final_dataframe1, index=False)
        df_2f.to_csv(final_dataframe2, index=False)

    return df_1f, df_2f

def clarifiation(
        df
):
    """
    ### Fonction de préparation pour l'autoencodeur
    
    _Faible mise en forme des données pour l'utilisation de l'auto-encodeur_
    """

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
        elif key[:7] == "Libellé":
            to_drop.append(key)

    df_2 = df_1.drop(columns=to_drop)

    df_3 = df_2.fillna(0)

    df_4 = df_3.replace({
        "MASCULIN":0,
        "FEMININ":1
        })
    
    # Encodage des Labels qualitatifs
    encodeur = []

    listing = params.Nuances_politiques
    listing.append(0)
    encodeur_nuances = LabelEncoder().fit(listing)
    encodeur.append(["Nuance",encodeur_nuances])

    df_5 = df_4
    for k in df_4.keys():
        if k[:6] == "Nuance":
            df_5[k] = encodeur_nuances.transform(df_4[k])

    for k in ["Code département", "Code commune", "Code BV"]:
        encodeur_spe = LabelEncoder().fit(df_4[k])
        encodeur.append([k,encodeur_spe])
        df_5[k] = encodeur_spe.transform(df_4[k])
    
    return df_5, encodeur