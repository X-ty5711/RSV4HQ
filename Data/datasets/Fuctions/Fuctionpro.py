# Fuctionpro.py
## Data Augmentation and Input

import numpy as np
import torch
import random

def load_array(data_arrays, batch_size, is_train=True):  #@save

    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def fun_calDiff(data,y,step):
    if len(data) != len(y):
        raise ValueError("data and y must have the same length")

    indices = [i * step + int(step/2) for i in range(round(len(data) / step))]
    data = [data[i] for i in indices]
    y = [y[i] for i in indices]

    U_avi,U_diff,Y_diff = [],[],[]

    for i in range(len(data) - 1):
        if y[i+1] == y[i]:
            raise ValueError("y[i+1] - y[i] is zero, cannot divide by zero")
        U_avi.append((data[i+1] + data[i])/2)
        U_diff.append((data[i+1] - data[i]) / (y[i+1] - y[i]))
        Y_diff.append(y[i+1] - y[i])

    return U_avi,U_diff, Y_diff

def splitData0(data0,step):
    U0 = data0[0]
    H0 = data0[1]
    B0 = data0[int(len(data0)-1)]
    Fr = (U0/np.sqrt(9.81*H0))
    X = B0+2*H0
    R=B0*H0/(B0+2*H0)
    Re = U0*R/1e-6
    dsize = int((len(data0)-2)/2.0)
    Ux0 = data0[2:dsize+2]
    Y = data0[dsize+2:2*dsize+2]
    # half_dsize = int(dsize/2.0)
    # inValueU = Ux0
    # [:half_dsize]
    # inValueY = Y[:half_dsize]
    
## Data Grade Enhance————
    aver = sum(Ux0)/len(Ux0)
    Uamx_Uaver= max(Ux0)/aver
    U_avi,U_diff,Y_diff = fun_calDiff(Ux0,Y,step)
##——————————————————————————————————————
    inValueU = U_avi/np.average(U_avi)
    pin_np1 = np.array(U_avi).reshape(1,-1)
    pin_np2 = np.array(U_diff).reshape(1,-1)
    pin_np3 = np.array(Y_diff).reshape(1,-1)
    pin_np = np.vstack((pin_np1, pin_np2, pin_np3))
    target_np = np.array([U0,H0])
    Z = np.array([U0,H0,B0,Fr,Re,X,R,step,aver,Uamx_Uaver])
    return pin_np, target_np ,Z

def genFeaturesLabels(data_set,step=1):
    features,labels,z=[],[],[]
    for data0 in data_set:
        pin_np, target_np ,zi=splitData0(data0,step)
        if zi[9]>1.1:
            continue
        if zi[1]>10:
            continue
        if zi[3]>0.99:
            continue
        features.append(pin_np)
        labels.append(target_np)
        z.append(zi)
    features =  torch.from_numpy(np.array(features))
    labels =  torch.from_numpy(np.array(labels))
    z = torch.from_numpy(np.array(z))
    return features, labels ,z

def datainput(data_set,batch_size=20,step=1):
    data_size=len(data_set)
    random.seed(1)

    train_size = int(data_size*0.8)
    train_cases = random.sample(list(np.arange(0,data_size)), train_size)
    train_cases.sort()
    test_cases = []
    for i in range(data_size):
        if i not in train_cases:
            test_cases.append(i)
            
    train_iter = load_array((genFeaturesLabels(data_set[train_cases],step)), batch_size)
    test_iter = load_array((genFeaturesLabels(data_set[test_cases],step)), batch_size)
    return train_iter,test_iter
