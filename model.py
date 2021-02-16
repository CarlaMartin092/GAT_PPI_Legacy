import torch
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv, GATConv
from sklearn.metrics import f1_score
from torch import nn, optim

class GATModel(nn.Module):

    def __init__(self, graph, n_heads, n_layers, input_size, hidden_size, output_size, nonlinearity, dropout = 0.6):
        super().__init__()

        self.n_layers = n_layers
        self.g = graph
        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * hidden_size if i > 0 else input_size
            out_hidden = hidden_size if i < n_layers - 1 else output_size
            out_channels = n_heads

            self.convs.append(GATConv(in_hidden, out_hidden, num_heads=n_heads, attn_drop=0))
            self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))
        
        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = nonlinearity
    
    def forward(self, feat):
        h = feat
        h = self.dropout0(h)

        for i in range(self.n_layers):
            conv = self.convs[i](self.g, h)
            linear = self.linear[i](h).view(conv.shape)

            h = conv + linear

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h, negative_slope=0.2)
                h = self.dropout(h)

        h = h.mean(1)

        return h
