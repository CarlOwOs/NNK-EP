import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

import torch_geometric.nn as pyg_nn
from torch_geometric.utils import dropout_adj

from utils.nnk import nnk_prunning

class XXSGCN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(XXSGCN, self).__init__()

        self.conv1 = pyg_nn.GCNConv(in_channels, 6)
        self.conv2 = pyg_nn.GCNConv(6, 12)
        self.conv3 = pyg_nn.GCNConv(12, 24)
        self.conv4 = pyg_nn.GCNConv(24, 48)

        self.dropout = nn.Dropout(0.5)

        self.mlp = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.ReLU(),
            nn.Linear(4, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        #x = self.dropout(x)
        #edge_index, edge_attr = nnk_prunning(x.cpu(), edge_index.cpu(), None, self.training)
        #edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=0.5)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        #x = self.dropout(x)
        #edge_index, edge_attr = nnk_prunning(x.cpu(), edge_index.cpu(), None, self.training)
        #edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=0.5)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        #x = self.dropout(x)
        #edge_index, edge_attr = nnk_prunning(x.cpu(), edge_index.cpu(), None, self.training)
        #edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=0.5)

        x = self.conv4(x, edge_index)
        x = F.relu(x)

        #x = self.dropout(x)
        #edge_index, edge_attr = nnk_prunning(x.cpu(), edge_index.cpu(), None, self.training)
        #edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=0.5)

        x = pyg_nn.global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
"""
class XXSGCN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(XXSGCN, self).__init__()

        self.conv1 = pyg_nn.GCNConv(in_channels, 32)
        self.conv2 = pyg_nn.GCNConv(32, 32)
        self.conv3 = pyg_nn.GCNConv(32, 32)

        self.dropout = nn.Dropout(0.5)

        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index)
        #x = self.dropout(x)
        x = F.relu(x)

        edge_index, edge_attr = nnk_prunning(x.cpu(), edge_index.cpu(), None, self.training)

        x = self.conv2(x, edge_index)
        #x = self.dropout(x)
        x = F.relu(x)

        edge_index, edge_attr = nnk_prunning(x.cpu(), edge_index.cpu(), None, self.training)

        x = self.conv3(x, edge_index)
        #x = self.dropout(x)
        x = F.relu(x)

        edge_index, edge_attr = nnk_prunning(x.cpu(), edge_index.cpu(), None, self.training)

        x = self.mlp(x)

        return x
"""
