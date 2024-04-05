import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool

class GraphSAGE(nn.Module):
    """
    GraphSAGE model for graph level prediction
    """
    def __init__(self, num_features, num_layers, num_classes):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, 128))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(128, 128))
        self.lin = nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)
        return F.log_softmax(self.lin(x), dim=-1)
