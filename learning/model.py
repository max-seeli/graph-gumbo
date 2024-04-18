import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, GINConv, BatchNorm, LayerNorm, JumpingKnowledge
from torch_geometric.nn import global_add_pool

class EmbResGCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn=True, use_res=True):
        super(EmbResGCNBlock, self).__init__()

        self.conv = GINConv(MLP([in_channels, out_channels]), train_eps=True)
        
        if use_bn:
            self.norm = BatchNorm(out_channels)
        else:
            self.norm = LayerNorm(out_channels)

        if use_res and in_channels != out_channels:
            raise ValueError('Residual connection is only available for in_channels == out_channels')
        self.res = nn.Identity()
        self.use_res = use_res

    def forward(self, x, edge_index):
        x_res = self.res(x)
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        if self.use_res:
            x = x + x_res
        return x
        

class EmbResGCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout):
        super(EmbResGCN, self).__init__()

        self.conv1 = EmbResGCNBlock(in_channels, hidden_channels, use_bn=True, use_res=False)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(EmbResGCNBlock(hidden_channels, hidden_channels, use_bn=True, use_res=True))
        
        self.jump = JumpingKnowledge(mode='cat')
        self.conv_out = EmbResGCNBlock(hidden_channels * (num_layers), hidden_channels, use_bn=True, use_res=False)

        self.head = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            xs += [x]
        
        x = self.jump(xs)
        x = self.conv_out(x, edge_index)
        
        x = global_add_pool(x, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.head(x)

        return F.log_softmax(x, dim=-1)

        