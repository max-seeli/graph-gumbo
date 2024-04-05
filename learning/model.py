import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool

class GraphGAT(nn.Module):
    """
    GraphGAT model for graph level prediction
    """
    def __init__(self, num_features, num_layers, num_classes):
        super(GraphGAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_features, 128))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(128, 128))
        self.seq = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr=edge_attr))
        x = global_add_pool(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.seq(x)
        return F.log_softmax(x, dim=1)


    
