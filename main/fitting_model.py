"""
auteur:Alexandre
date:2024/09/06
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# class TransfertVoix(torch.nn.Module):
#     def __init__(self, N_1er, N_2eme):#, device=None):
#         super(TransfertVoix, self).__init__()
        
#         self.lin = torch.nn.Linear(N_2eme, N_1er, bias=False)

#     def forward(self, p_1):
        
#         M = torch.softmax(self.lin.weight, axis=1)
#         p_2_pred = torch.matmul(p_1, M)
        
#         return p_2_pred
    
class TransfertVoix(torch.nn.Module):
    def __init__(self, N_1er, N_2eme):#, device=None):
        super(TransfertVoix, self).__init__()
        
        self.lin = torch.nn.Linear(N_2eme, N_1er, bias=False)

    def forward(self, p_1):
        
        p_1_calmped = p_1 * (p_1 > 0.125)

        M = torch.softmax(self.lin.weight, axis=1)
        p_2_pred = torch.matmul(p_1_calmped, M)
        
        return p_2_pred
    
def pdf_loss(p_pred, p, weight):

    ind_nonzero = (p==0) + (p_pred==0)
    p[ind_nonzero] = 1.
    p_pred[ind_nonzero] = 1.
    kl_div = p * (p.log() - p_pred.log())
    loss_train = (kl_div * weight[:, None]).sum()
    
    return loss_train

def fit_data(
    df_1f,
    df_2f,
    
    lr=1e-5,
    batch_size=2**5,
    num_epochs=500,
    
    seed=2024,
    device=None
        
):
    
    # Paramètrage des calculs
    if device == "cpu":
        print(f"Il semble y avoir {os.cpu_count()} coeurs dans cet ordi !")
    elif device == "cuda":
        if torch.cuda.is_available() == True:
            print("GPU ! GPU ! GPU !")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda": torch.cuda.empty_cache()

    # Preparation des données
    pre_X_train, pre_X_test, pre_Y_train, pre_Y_test = train_test_split(
        df_1f, df_2f, 
        test_size=.2, 
        random_state=seed, 
        shuffle=True
    )
    X_train, X_test = pre_X_train[pre_X_train.keys()[:-2]], pre_X_test[pre_X_test.keys()[:-2]]
    Y_train, Y_test = pre_Y_train[pre_Y_train.keys()[:-2]], pre_Y_test[pre_Y_test.keys()[:-2]]

    # X_Votants_train, X_Votants_test = pre_X_train[pre_X_train.keys()[-1:]], pre_X_test[pre_X_test.keys()[-1:]]
    # Y_Votants_train, Y_Votants_test = pre_Y_train[pre_Y_train.keys()[-1:]], pre_Y_test[pre_Y_test.keys()[-1:]]

    loader = DataLoader(
                    TensorDataset(
                        torch.Tensor(X_train.to_numpy()).to(device), 
                        torch.Tensor(Y_train.to_numpy()).to(device)
                    ), batch_size=batch_size, shuffle=True
                )
    
    # Préparation du modèle
    N_1er, N_2eme = X_train.shape[-1], Y_train.shape[-1]
    trans = TransfertVoix(N_1er, N_2eme)
    trans = trans.to(device)

    # Préparation de l'apprentissage
    optimizer = torch.optim.Adam(
        trans.parameters(), 
        lr=lr
    )

    losses_train = []
    losses_test = []

    for epoch in tqdm(range(int(num_epochs)), desc=f"Computing on {device}"):
        
        for p_1, p_2 in loader:

            p_1, p_2 = p_1.to(device), p_2.to(device)
            sum_1, sum_2 = p_1.sum(axis=1)[:,None], p_2.sum(axis=1)[:,None]
            
            # Sécurisation anti 0
            if np.where(sum_1 == 0)[0].size > 0:
                sum_1[np.where(sum_1 == 0)[0]] = 1
            if np.where(sum_2 == 0)[0].size > 0:
                sum_2[np.where(sum_2 == 0)[0]] = 1

            p_1t = p_1 / sum_1
            p_2t = p_2 / sum_2
            
            p_2_pred = trans(p_1t)
            
            weight = sum_2 / sum_2.sum()

            loss_train = pdf_loss(p_2_pred, p_2t, weight)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        losses_train.append(loss_train.item())

        with torch.no_grad():
            
            x_test = torch.Tensor(X_test.to_numpy()).to(device)
            sum_test_1 = x_test.sum(axis=-1)[:,None].to(device)
            sum_test_1[np.where(sum_test_1 == 0)] = 1

            y_test = torch.Tensor(Y_test.to_numpy()).to(device)
            sum_test_2 = y_test.sum(axis=-1)[:,None].to(device)
            sum_test_2[np.where(sum_test_2 == 0)] = 1

            Y_pred = trans(x_test / sum_test_1)
            
            weight = sum_test_2 / sum_test_2.sum()

            losse_test = pdf_loss(Y_pred, y_test / sum_test_2, weight)
            losses_test.append(losse_test.item())

    return trans, np.array(losses_train), np.array(losses_test), Y_pred.cpu().detach().numpy(), (y_test / sum_test_2).cpu().detach().numpy()