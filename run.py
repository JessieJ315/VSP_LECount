# import torch

# !pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
# !pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install torch_geometric_signed_directed
# !pip install git+https://github.com/pyg-team/pytorch_geometric.git

import scipy.special
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from math import ceil
from torch_geometric.utils.convert import to_networkx
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from more_itertools import one
from math import comb

import partial_order_func as pofun
import vsp_func as vspfun
from le-gnn import data_generation, Net

log_factorial = lambda x : scipy.special.gammaln(1+x)

max_n = 50
num_features = 16
batch_size = 10

train_dataset = data_generation(max_n, num_features, 100)
val_dataset = data_generation(max_n, num_features, 20)
test_dataset = data_generation(max_n, num_features, 20)

train_loader = DenseDataLoader(train_dataset, batch_size=batch_size)
val_loader = DenseDataLoader(val_dataset, batch_size=batch_size)
test_loader = DenseDataLoader(test_dataset, batch_size=batch_size)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data.x, data.adj,None)
        loss = F.mse_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()
    mse_all = 0

    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, None)[0]
        mse = F.mse_loss(pred,data.y.view(-1))
        mse_all += data.y.size(0) * float(mse)

    return mse_all / len(loader.dataset)


best_val_mse = test_mse = 10000
train_loss_all = []
val_loss_all = []
# times = []
for epoch in range(1, 1501):
    # start = time.time()
    train_loss = train(epoch)
    val_mse = test(val_loader)
    train_loss_all.append(train_loss)
    val_loss_all.append(val_mse)
    if val_mse < best_val_mse:
        test_mse = test(test_loader)
        best_val_mse = val_mse
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_mse:.4f}, Test Acc: {test_mse:.4f}')
    # times.append(time.time() - start)
# print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
