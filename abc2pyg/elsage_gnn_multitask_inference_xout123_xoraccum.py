#Feat_Reduce to match # of nodes to '4'
import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm

from dataset_prep import PygNodePropPredDataset, Evaluator, EdgeListDataset
from dataset_prep.dataset_gl_pyg import LogicGateDataset
from dataset_prep.dataset_xor_pyg import XORDataset
from dataset_prep.dataset_xor_accum_pyg import XORAccumDataset

from logger import Logger
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import copy
from elsage.el_sage_baseline_xout123_xoraccum import GraphSAGE
from elsage.el_sage_baseline_xout123_xoraccum import train as train_el
from elsage.el_sage_baseline_xout123_xoraccum import test as test_el
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import wandb

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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

#If # of node_feat is not 4
class Feat_Reduce(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(Feat_Reduce, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels, bias=False)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels, bias=False)
        self.dropout = dropout
    def forward(self, data):
        x = self.conv1(data.x, data.adj_t)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout) 
        x = self.conv2(x, data.adj_t)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout) 
        x = torch.sigmoid(x)
        return x
        
class SAGE_MULT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE_MULT, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # two linear layer for predictions
        self.linear = torch.nn.ModuleList()
        self.linear.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        
        self.bn0 = BatchNorm1d(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.linear:
            lin.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # print(x[0])
        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = self.linear[1](x) # for xor
        x2 = self.linear[2](x) # for maj
        x3 = self.linear[3](x) # for roots
        # print(self.linear[0].weight)
        # print(x1[0])
        return x, x1.log_softmax(dim=-1), x2.log_softmax(dim=-1), x3.log_softmax(dim=-1)
    
    def forward_nosampler(self, x, adj_t, device):
        # tensor placement
        x.to(device)
        adj_t.to(device)
        
        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # print(x[0])
        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = self.linear[1](x) # for xor
        x2 = self.linear[2](x) # for maj
        x3 = self.linear[3](x) # for roots
        # print(self.linear[0].weight)
        # print(x1[0])
        return x1, x2, x3

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                xs.append(x)

                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
            #print(x_all.size())
            
        x_all = self.linear[0](x_all)
        x_all = F.relu(x_all)
        x_all = self.bn0(x_all)
        x1 = self.linear[1](x_all) # for xor
        x2 = self.linear[2](x_all) # for maj
        x3 = self.linear[3](x_all) # for roots
        pbar.close()

        return x1, x2, x3  
     
    
def main():
    parser = argparse.ArgumentParser(description='mult16')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--datatype', type=str, default='xor_accum', choices=['xor_accum'])
    parser.add_argument('--device', type=int, default=0)
    #args for gamora
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default='SAGE_mult8')
    
    #args for elsage
    parser.add_argument('--root', type=str, default='/home/curie/masGen/DataGen/dataset8', help='Root directory of dataset')
    parser.add_argument('--highest_order', type=int, default=8, help='Highest order for the EdgeListDataset')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    #parser.add_argument('--num_layers', type=int, default=4) # x + gamora_output
    #parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=75, help='Hidden dimension size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()
    initialize_wandb(args)
    set_seed(args.seed)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ### evaluation dataset loading
    dataset = XORAccumDataset(root = args.root, highest_order = args.highest_order)
    ### evaluation dataset loading
    if args.datatype == 'xor_accum':
        dataset = XORAccumDataset(root = args.root, highest_order = args.highest_order)
    data = dataset[0]
    print(data)
    data = T.ToSparseTensor()(data)
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    data = data.to(device)
    
    featreduce_model = Feat_Reduce(dataset[0].num_node_features, args.hidden_dim, 4, args.dropout).to(device)
    
    gamora_model = SAGE_MULT(4, args.hidden_channels,
                     3, args.num_layers,
                     args.dropout).to(device)

    gamora_model.load_state_dict(torch.load(args.model_path))
    
    elsage_model = GraphSAGE(in_dim=13,#dataset[0].num_node_features, #9 for gamora_output
                 hidden_dim=args.hidden_dim, 
                 out_dim=dataset.num_classes,
                 num_layers=args.num_layers,
                 dropout=args.dropout
                 ).to(device)

    #optimizer = torch.optim.Adam(elsage_model.parameters(), args.learning_rate)#, weight_decay=5e-4)
    optimizer = torch.optim.Adam(list(featreduce_model.parameters()) + list(elsage_model.parameters()), args.learning_rate)
    for epoch in range(1, args.epochs + 1):
        loss, train_acc, train_all_bits = train_el(featreduce_model, gamora_model, elsage_model, train_loader, optimizer, device, dataset)
        if epoch % 1 == 0:
            val_acc, val_acc_all_bits = test_el(featreduce_model, gamora_model, elsage_model, val_loader, device, dataset)
            test_acc, test_acc_all_bits = test_el(featreduce_model, gamora_model, elsage_model, test_loader, device, dataset)
            wandb.log({"Epoch": epoch, "Loss": loss, "Train_acc": train_acc, "Train_acc_all_bits": train_all_bits, 
                       "Val_acc":val_acc, "Test_acc": test_acc, "Val_acc_all_bits":val_acc_all_bits, "Test_acc_all_bits": test_acc_all_bits})
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Train acc all bits: {train_all_bits:.4f}, Val acc all bits: {val_acc_all_bits:.4f}, Test acc all bits: {test_acc_all_bits:.4f}')
            
    
if __name__ == "__main__":
    main()
