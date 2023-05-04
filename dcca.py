import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler

import time

class DNN(nn.Module):
    def __init__(self, layers):
        
        super(DNN, self).__init__()
        modules = []
        
        for i in range(len(layers) - 1):
            if i == len(layers) - 2:
                modules.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layers[i], affine=False),
                    nn.Linear(layers[i], layers[i+1])
                ))
            else:
                modules.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layers[i]),
                    nn.ReLU(),
                    nn.Linear(layers[i], layers[i+1])
                ))
                
        self.dnn = nn.ModuleList(modules)
        
    def forward(self, x):
        for layer in self.dnn:
            x = layer(x)
        return x

class DCCA(nn.Module):
    def __init__(self, nn1_layers, nn2_layers, mode):
        super(DCCA, self).__init__()
        if mode == 'l-bfgs':
            self.nn1 = DNN(nn1_layers).float()
            self.nn2 = DNN(nn2_layers).float()
        else:
            self.nn1 = DNN(nn1_layers).double()
            self.nn2 = DNN(nn2_layers).double()
        
    def forward(self, x1, x2):
        return self.nn1(x1), self.nn2(x2)
    
class TorchWrapper:
    def __init__(self, model, loss, opt, proj_dim, batch_size, device, mode):
        self.model = model
        self.loss = loss
        self.opt = opt
        self.proj_dim = proj_dim
        self.batch_size = batch_size
        self.device = device
        self.mode = mode
        self.f = 'model.dat'
        
    def get_losses_projs(self, x1, x2, calc_losses=False):
        with torch.no_grad():
            self.model.eval()  
            
            losses = []
            outputs1 = []
            outputs2 = []
            
            if self.mode == 'l-bfgs':
                idx_x1 = x1[:, :]
                idx_x2 = x2[:, :]
                o1, o2 = self.model(idx_x1, idx_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                if calc_losses:
                    idx_x2_loss = self.loss(o1, o2, self.proj_dim, self.device, torch.float32)
                    losses.append(idx_x2_loss.item())
            else:
                batch_idxs = list(BatchSampler(SequentialSampler(
                    range(x1.size(0))), batch_size=self.batch_size, drop_last=False))
                for batch_idx in batch_idxs:
                    batch_x1 = x1[batch_idx, :]
                    batch_x2 = x2[batch_idx, :]
                    o1, o2 = self.model(batch_x1, batch_x2)
                    outputs1.append(o1)
                    outputs2.append(o2)
                    if calc_losses:
                        b_loss = self.loss(o1, o2, self.proj_dim, self.device, torch.float64)
                        losses.append(b_loss.item())
            
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        
        return losses, outputs
    
    def get_proj(self, x1, x2):
        losses, outputs = self.get_losses_projs(x1, x2)
        return outputs
    
    def get_loss(self, x1, x2):
        losses, _ = self.get_losses_projs(x1, x2, calc_losses=True)
        return np.mean(losses)
    
    def get_cca_proj(self, x1, x2, cca):
        outputs = self.get_proj(x1, x2)
        outputs = cca.transform(outputs[0], outputs[1])
        return outputs
    
    def fit(self, x1, x2, xv1, xv2, epochs):
        
        best_val_loss = 99999
        train_losses = []
        elapsed_time = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            self.model.train()
            
            if self.mode == 'l-bfgs':
                idx_x1 = x1[:,:]
                idx_x2 = x2[:,:]
                
                def loss_closure():
                    self.opt.zero_grad()
                    o1, o2 = self.model(idx_x1, idx_x2)
                    idx_loss = self.loss(o1, o2, self.proj_dim, self.device, torch.float32)
                    idx_loss.backward()
                    return idx_loss
                
                self.opt.step(loss_closure)
                self.opt.zero_grad()
                o1, o2 = self.model(idx_x1, idx_x2)
                idx_loss = self.loss(o1, o2, self.proj_dim, self.device, torch.float32)
                train_losses.append(idx_loss.item())
                
            else:
                batch_idxs = list(BatchSampler(
                    RandomSampler(range(x1.size(0))),
                    batch_size=self.batch_size,
                    drop_last=False
                ))
                
                for batch in batch_idxs:
                    self.opt.zero_grad()
                    b_x1 = x1[batch,:]
                    b_x2 = x2[batch,:]
                    o1, o2 = self.model(b_x1, b_x2)
                    b_loss = self.loss(o1, o2, self.proj_dim, self.device, torch.float64)
                    train_losses.append(b_loss.item())
                    b_loss.backward()
                    self.opt.step()
                    
            train_loss = np.mean(train_losses)
            elapsed_time += time.time() - epoch_start
            
            with torch.no_grad():
                self.model.eval()
                val_loss = self.get_loss(xv1, xv2)
                
                print(f'[Epoch {epoch+1}: t = {elapsed_time:.2f} | train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f}]')
                if val_loss < best_val_loss:
                    print(f'\tNew best model: val_loss = {val_loss:.4f} < {best_val_loss:.4f}')
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.f)
                    
    def load_best(self):
        state = torch.load(self.f)
        self.model.load_state_dict(state)
        
    def load_specific(self, loc):
        state = torch.load(loc)
        self.model.load_state_dict(state)