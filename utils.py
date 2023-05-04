import torch
import numpy as np
import scipy.io as sio

def load_data(dtype):    
    data = sio.loadmat('XRMBf2KALDI_window7_single1.mat')
    x1_orig = data['X1']    
    
    data = sio.loadmat('XRMBf2KALDI_window7_single2.mat')
    x2_orig = data['X2']    
    trainLabel = data['trainLabel']
    
    train = 27600
    tune = 9200
    
    indices = (data['trainID'] == 11).flatten()
    x1_11 = x1_orig[indices]
    x2_11 = x2_orig[indices]
    y_11 = trainLabel[indices]
    
    x1 = x1_11[:train]
    x2 = x2_11[:train]
    y = y_11[:train]
    
    xv1 = x1_11[train:train+tune]
    xv2 = x2_11[train:train+tune]
    yv = y_11[train:train+tune]
    
    xt1 = x1_11[train+tune:]
    xt2 = x2_11[train+tune:]
    yt = y_11[train+tune:]
    
    x1 = torch.tensor(x1, dtype=dtype)
    x2 = torch.tensor(x2, dtype=dtype)
    
    xv1 = torch.tensor(xv1, dtype=dtype)
    xv2 = torch.tensor(xv2, dtype=dtype)
    
    xt1 = torch.tensor(xt1, dtype=dtype)
    xt2 = torch.tensor(xt2, dtype=dtype)
    
    y = np.asarray(y, dtype='int32')
    yv = np.asarray(yv, dtype='int32')
    yt = np.asarray(yt, dtype='int32')

    return x1, x2, y, xv1, xv2, yv, xt1, xt2, yt