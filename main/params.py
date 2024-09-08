"""
auteur:Alexandre
date:2024/09/01
"""

import os
from watermark import watermark

#--------- Path
cwd = os.path.dirname(os.getcwd())
datasets_path = os.path.join(cwd, "datasets")
figures_path = os.path.join(cwd, "fig")

#--------- Noms fichiers
dataset_1er_tour = "1ertour_legislative"
dataset_2nd_tour = "2ndtour_legislative"

#--------- Informations générales
Nuances_politiques = ['UG', 'ECO', 'ENS', 'RN', 'EXG', 'REC', 'LR', 'UDI', 'DVD', 'DSV', 'UXD', 'DIV', 
                      'REG', 'HOR', 'DVG', 'DVC', 'EXD', 'SOC', 'RDG', 'COM', 'FI', 'autre', 'VEC']
Grandes_Nuances = ["UG","ENS","RN","LR","DVD","UXD"]

import matplotlib.cm as cm

#--------- Graphical parameters
cm_jet = cm.jet

def WaTer():
    print(watermark())