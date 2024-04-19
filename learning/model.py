import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN
from torch_geometric.nn import global_add_pool

class GraphGIN(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_layers, dropout, **kwargs):
        super(GraphGIN, self).__init__()

        self.mpnn = GIN(hidden_channels=hidden_channels, 
                        out_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout, **kwargs)

        self.dropout = dropout
        self.head = nn.Linear(hidden_channels, out_channels) 


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.mpnn(x, edge_index, batch=batch)
        x = global_add_pool(x, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.head(x)

        return F.log_softmax(x, dim=-1)

        