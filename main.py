'''
This code is inspired by the following projects:
https://github.com/Michaelvll/DeepCCA
https://github.com/robinthibaut/deep-cca
https://github.com/VahidooX/DeepCCA
'''

import torch
import torch.nn as nn
from utils import load_data
from sklearn import svm
from sklearn.metrics import accuracy_score
from loss import cca_loss
from dcca import DCCA, TorchWrapper

from sklearn.cross_decomposition import CCA

mode = 'sgd'

device = torch.device('cuda')
print('device', torch.cuda.current_device())
print(f'get_device_name: {torch.cuda.get_device_name(0)}')

print('Loading data from 2 files')

dtype = torch.float64

if mode == 'l-bfgs':
    print("Using L-BFGS")
    dtype = torch.float32
else:
    if mode != 'sgd':
        print('Invalid mode: defaulting to SGD')
    else:
        print("Using SGD")

x1, x2, y, xv1, xv2, yv, xt1, xt2, yt = load_data(dtype)
print('Setting up environment')

lr = 1e-3
epochs = 10
batch_size = 200
reg_par = 1e-5
proj_dim = 112

nn1_layers = [x1.shape[1], 1800, 1800, proj_dim]
nn2_layers = [x2.shape[1], 1200, 1200, proj_dim]

if mode == 'l-bfgs':
    model = DCCA(nn1_layers, nn2_layers, mode).float()
else:
    model = DCCA(nn1_layers, nn2_layers, mode).double()

model = nn.DataParallel(model)
model.to(device)

loss = cca_loss

if mode == 'l-bfgs':
    opt = torch.optim.LBFGS(model.parameters(), lr)
else:
    opt = torch.optim.SGD(model.parameters(), lr, weight_decay=reg_par)

tw = TorchWrapper(model, loss, opt, proj_dim, batch_size, device, mode)

x1.to(device)
x2.to(device)

xv1.to(device)
xv2.to(device)

xt1.to(device)
xt2.to(device)

print('Training model\n')
tw.fit(x1, x2, xv1, xv2, epochs)
tw.load_best()
    
outputs = tw.get_proj(x1, x2)

print(f'Final validation loss: {tw.get_loss(xv1, xv2):.4f}')
print(f'Final test loss: {tw.get_loss(xt1, xt2):.4f}')
print('\nFitting CCA for post-processing')

l_cca = CCA(n_components=proj_dim)
l_cca.fit(outputs[0], outputs[1])

outputs = tw.get_cca_proj(torch.cat([x1, xv1, xt1], dim=0), torch.cat(
    [x2, xv2, xt2], dim=0), l_cca)

proj_x1 = outputs[0][:x1.size(0)]
proj_x2 = outputs[1][:x2.size(0)]

proj_xv1 = outputs[0][x1.size(0) : x1.size(0)+xv1.size(0)]
proj_xv2 = outputs[1][x2.size(0) : x2.size(0)+xv2.size(0)]

proj_xt1 = outputs[0][-xt1.size(0):]
proj_xt2 = outputs[1][-xt1.size(0):]

print('Fitting classifier')

clf = svm.LinearSVC(C=0.01, dual=False)
clf.fit(proj_x1, y.ravel())


print('\nProgram complete\nModel Details')
pred = clf.predict(proj_xv1)
print(f'\tValidation accuracy: {accuracy_score(yv, pred)*100:.2f}%')
pred = clf.predict(proj_xt1)
print(f'\tTest accuracy: {accuracy_score(yt, pred)*100:.2f}%')