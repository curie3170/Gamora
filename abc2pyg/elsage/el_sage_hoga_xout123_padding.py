import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import MLP,SAGEConv, global_mean_pool, BatchNorm
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch_geometric.data import DataLoader

import time

import wandb
def initialize_wandb(args):
    if args.wandb:
        wandb.init(
            project="el_sage",
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_layers": args.num_layers,
                "hidden_dim": args.hidden_dim,
                "highest_order": args.highest_order,
                "dropout": args.dropout
                
            }
        )
    else:
        wandb.init(mode="disabled")


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, max_num_nodes, num_layers =4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.max_num_nodes = max_num_nodes
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

        self.mlp1 = MLP([max_num_nodes, hidden_dim, 1],
                       norm=None, dropout=0.5) #instead of global_mean_pool
        self.bn = BatchNorm(hidden_dim)
        self.mlp2 = MLP([hidden_dim, hidden_dim, out_dim],
                       norm=None, dropout=0.5)
    
    def forward(self, data, gamora_output):
        x = torch.cat((data.x_ori, gamora_output[0], gamora_output[1], gamora_output[2]), 1)
        #x = data.x_ori
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.adj_t)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout) #torch.Size([58666, hidden_dim])
        '''
        x = global_mean_pool(x, data.batch) # torch.Size([batch, hidden_dim])
        x = self.fc(x) # torch.Size([batch, hidden_dim])
        '''
        x = x.reshape([-1,x.shape[1],self.max_num_nodes])
        x = self.mlp1(x).squeeze(2) #[32, 75]
        x = self.bn(x)
        x = F.relu(x)
        x = self.mlp2(x)
        return x #torch.Size([batch, 16])

def train(hoga_model, model, loader, optimizer, device, dataset):
    start_time = time.time()
    hoga_model.eval()
    model.train()
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss() #sigmoid + BCE
    for data in loader:
        data = data.to(device)
        out1, out2, out3, _ = hoga_model(data.x)
        optimizer.zero_grad()
        out = model(data,[out1, out2, out3])
        loss = criterion(out, data.y.reshape(-1, dataset.num_classes))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("--- Train time: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    train_acc, train_acc_all_bits = test(hoga_model, model, loader, device, dataset)
    print("--- Test time: %s seconds ---" % (time.time() - start_time))
    return total_loss / len(loader), train_acc, train_acc_all_bits

@torch.no_grad()
def test(hoga_model, model, loader, device, dataset):
    hoga_model.eval()
    model.eval()
    correct = 0
    correct_all = 0
    total = 0
    total_all = 0
    for data in loader:
        data = data.to(device)
        out1, out2, out3, _ = hoga_model(data.x)
        out = model(data, [out1, out2, out3])
        out = torch.sigmoid(out)
        pred = (out > 0.5).float()
        correct += (pred == data.y.reshape(-1, dataset.num_classes)).sum().item()
        correct_all+= torch.eq(pred, data.y.reshape(-1, dataset.num_classes)).all(dim=1).sum().item()
        total += data.y.reshape(-1, dataset.num_classes).numel()
        total_all += len(torch.eq(pred, data.y.reshape(-1, dataset.num_classes)).all(dim=1))
    return correct / total, correct_all / total_all

# @torch.no_grad()
# def test(hoga_model, model, loader, device, dataset):
#     hoga_model.eval()
#     model.eval()
#     correct = 0
#     correct_all = 0
#     total = 0
#     total_all = 0
#     for data in loader:
#         data = data.to(device)
#         out1, out2, out3, _ = hoga_model(data.x)
#         out = model(data, [out1, out2, out3])
#         out = torch.sigmoid(out)
#         pred = (out > 0.5).float()
#         pred = pred[:, :-1]
#         print(pred.shape)
#         y = data.y.reshape(-1, dataset.num_classes)[:, :-1]
#         correct += (pred == y).sum().item()
#         correct_all+= torch.eq(pred, y).all(dim=1).sum().item()
#         total += y.numel()
#         total_all += len(torch.eq(pred, y).all(dim=1))
#     return correct / total, correct_all / total_all
