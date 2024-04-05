"""
Train a model on the NCI dataset from TU Dortmund.
Comparing the performance of the regular dataset as well as a transformed dataset (product graphs).
"""
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from model import GraphSAGE

path = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = TUDataset(path, name='NCI1')
loader = DataLoader(dataset, batch_size=32, shuffle=True)


model = GraphSAGE(dataset.num_features, 3, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = F.nll_loss


def train():
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test():
    model.eval()
    ys, preds = [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        ys.append(data.y)
        preds.append(out.argmax(dim=1))
    y, pred = torch.cat(ys), torch.cat(preds)
    acc = pred.eq(y).sum().item() / y.size(0)
    f1 = f1_score(y.cpu(), pred.cpu(), average='micro')
    return acc, f1


for epoch in range(1, 101):
    acc, f1 = test()
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}')


