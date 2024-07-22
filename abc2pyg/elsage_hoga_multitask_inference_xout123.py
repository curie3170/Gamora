import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm

from dataset_prep import PygNodePropPredDataset, Evaluator
from dataset_prep.dataset_el_pyg import EdgeListDataset
from dataset_prep.dataset_gl_pyg import LogicGateDataset
from dataset_prep.dataset_xor_pyg import XORDataset

from logger import Logger
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import time
import copy
from elsage.el_sage_hoga_xout123 import GraphSAGE
from elsage.el_sage_hoga_xout123 import train as train_el
from elsage.el_sage_hoga_xout123 import test as test_el
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from hoga_model import HOGA
from hoga_utils import *
import wandb
from torch.utils.data import Dataset

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class ProcessedSparseDataset(Dataset):
    def __init__(self, raw_dataset, args):
        self.data = []
        for data in raw_dataset:
            processed_data = preprocess(data, args)
            sparse_data = T.ToSparseTensor()(processed_data)
            self.data.append(sparse_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
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
        

def main():
    parser = argparse.ArgumentParser(description='elsage_hoga')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--datatype', type=str, default='aig', choices=['aig', 'logic', 'xor'])
    #args for HOGA
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model_path', type=str, default='models/hoga_mult8_mult.pt')
    parser.add_argument('--directed', action='store_true')
    parser.add_argument('--hoga_hidden_channels', type=int, default=256)
    parser.add_argument('--hoga_num_layers', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    
    #args for elsage
    parser.add_argument('--num_layers', type=int, default=4)
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
    if args.datatype == 'aig':
        dataset = EdgeListDataset(root = args.root, highest_order = args.highest_order)
    elif  args.datatype == 'logic':
        dataset = LogicGateDataset(root = args.root, highest_order = args.highest_order)
    elif  args.datatype == 'xor':
        dataset = XORDataset(root = args.root, highest_order = args.highest_order)
    processed_dataset = ProcessedSparseDataset(dataset, args) 

    train_dataset, test_dataset = train_test_split(processed_dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    data = dataset[0]
    data = T.ToSparseTensor()(data)
    data = data.to(device)
    
    hoga_model = HOGA(data.num_features, args.hoga_hidden_channels, 3, args.hoga_num_layers,
            args.dropout, num_hops=args.num_hops+1, heads=args.heads).to(device)
    hoga_model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    
    elsage_model = GraphSAGE(in_dim=13, #dataset[0].num_node_features,#13, #9 for gamora_output
                 hidden_dim=args.hidden_dim, 
                 out_dim=dataset.num_classes,
                 num_layers=args.num_layers,
                 dropout=args.dropout
                 ).to(device)
    optimizer = torch.optim.Adam(elsage_model.parameters(), args.lr)#, weight_decay=5e-4)
    for epoch in range(1, args.epochs + 1):
        loss, train_acc, train_all_bits = train_el(hoga_model, elsage_model, train_loader, optimizer, device, dataset)
        if epoch % 1 == 0:
            val_acc, val_acc_all_bits = test_el(hoga_model, elsage_model, val_loader, device, dataset)
            test_acc, test_acc_all_bits = test_el(hoga_model, elsage_model, test_loader, device, dataset)
            wandb.log({"Epoch": epoch, "Loss": loss, "Train_acc": train_acc, "Train_acc_all_bits": train_all_bits, 
                       "Val_acc":val_acc, "Test_acc": test_acc, "Val_acc_all_bits":val_acc_all_bits, "Test_acc_all_bits": test_acc_all_bits})
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Train acc all bits: {train_all_bits:.4f}, Val acc all bits: {val_acc_all_bits:.4f}, Test acc all bits: {test_acc_all_bits:.4f}')
            
    
if __name__ == "__main__":
    main()
