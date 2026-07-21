"""
auteur:Alexandre
date:2024/09/01
"""

from watermark import watermark
import matplotlib.cm as cm
from pathlib import Path
import torch

#--------- Path
cwd = Path.cwd().resolve().parent
datasets_raw_path = cwd / "datasets_raw"
datasets_pp_path = cwd / "datasets_pp"
figures_path = cwd / "fig"
results_path = cwd / "results"

#--------- Support de calcul
if torch.cuda.is_available():
    device = "cuda"
elif torch.xpu.is_available():
    device = "xpu"
elif torch.mps.is_available():
    device = "mps"
else:
    device="cpu"

#--------- Noms fichiers
dataset_1er_tour = "1ertour_legislative"
dataset_2nd_tour = "2ndtour_legislative"

#--------- Informations générales
Nuances_politiques = ['UG', 'ECO', 'ENS', 'RN', 'EXG', 'REC', 'LR', 'UDI', 'DVD', 'DSV', 'UXD', 'DIV', 
                      'REG', 'HOR', 'DVG', 'DVC', 'EXD', 'SOC', 'RDG', 'COM', 'FI', 'autre', 'VEC']
Grandes_Nuances = ["UG","ENS","RN","LR","DVD","UXD"]
conversion_nuance_dico = {
    0 : 1,
    1 : 12,
    2 : 0,
    3 : 2,
    4 : None,
    5 : None,
    6 : 3,
    7 : 13,
    8 : 5,
    9 : 15,
    10 : 4,
    11 : 10,
    12 : 8,
    13 : 6,
    14 : 11,
    15 : 7,
    16 : 9,
    17 : 14,
    18 : None,
    19 : None,
    20 : 16,
    21 : None,
    22 : 17
}

#--------- Graphical parameters
cm_jet = cm.jet

def WaTer():
    print(watermark())