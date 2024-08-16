import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphSAGE, MLP,SAGEConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_prep.dataset_xor_binary_pyg import XORBinaryDataset
import wandb
import time

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
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers =4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

        self.fc = Linear(hidden_dim, hidden_dim)
        self.mlp1 = MLP([8, hidden_dim, 1], 
                       norm=None, dropout=0.5) #instead of global_mean_pool
        self.bn = BatchNorm(hidden_dim)
        #self.fc = Linear(hidden_dim, hidden_dim)
        self.mlp2 = MLP([hidden_dim, hidden_dim, out_dim],
                       norm=None, dropout=0.5)

    
    def forward(self, data, gamora_output, po, po_batch):
        #x = data.x
        #x = torch.cat((gamora_output[0], gamora_output[1], gamora_output[2]), 1)
        x = torch.cat((data.x, gamora_output[0], gamora_output[1], gamora_output[2]), 1)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.adj_t)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout) #torch.Size([58666, hidden_dim])
        # x = global_mean_pool(x[po], po_batch) # torch.Size([batch * # of POs,, hidden_dim])
        # x = self.fc(x) # torch.Size([batch, hidden_dim])
        x = x[po]
        x = x.reshape([-1 ,x.shape[1], 8]) #(batch, hidden_dim, max_nodes)
        x = self.mlp1(x).squeeze(2) #(batch, hidden_dim, 1) -> (32, 75)
        x = self.bn(x)
        x = F.relu(x)
        x = self.mlp2(x) # (32, 8)
        x = F.relu(x)

        return x #torch.Size([batch, 16])

def train(gamora_model, model, loader, optimizer, device, dataset):
    start_time = time.time()
    gamora_model.eval()
    model.train()
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss() #sigmoid + BCE
    for data in loader:
        po = torch.zeros(data.po.shape)
        po_batch = torch.zeros(data.po.shape)
        for i in range(0, len(data.po), 8):
            po[i:i+8] = data.po[i:i+8]+ int(394 * (i // 8))
        for i in range(0, len(data.po), 8):
            po_batch[i:i+8] = (i // 8)
        data = data.to(device)
        out1, out2, out3 = gamora_model.forward_nosampler(data.x, data.adj_t, device)
        optimizer.zero_grad()
        out = model(data,[out1, out2, out3], po.int(), po_batch.long().to(device))
        loss = criterion(out, data.y.reshape(-1, dataset.num_classes))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("--- Train time: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    train_acc, train_acc_all_bits, train_acc_each_bit = test(gamora_model, model, loader, device, dataset)
    print(train_acc_each_bit)
    print("--- Test time: %s seconds ---" % (time.time() - start_time))
    return total_loss / len(loader), train_acc, train_acc_all_bits, train_acc_each_bit

@torch.no_grad()
def test(gamora_model, model, loader, device, dataset):
    gamora_model.eval()
    model.eval()
    correct = 0
    correct_all = 0
    total = 0
    total_all = 0
    correct_bit = [0]*dataset.num_classes
    for data in loader:
        po = torch.zeros(data.po.shape)
        po_batch = torch.zeros(data.po.shape)
        for i in range(0, len(data.po), 8):
            po[i:i+8] = data.po[i:i+8]+ int(394 * (i // 8))
        for i in range(0, len(data.po), 8):
            po_batch[i:i+8] = (i // 8)
        data = data.to(device)
        out1, out2, out3 = gamora_model.forward_nosampler(data.x, data.adj_t, device)
        out = model(data, [out1, out2, out3], po.int(), po_batch.long().to(device))
        out = torch.sigmoid(out)
        pred = (out > 0.5).float()
        correct += (pred == data.y.reshape(-1, dataset.num_classes)).sum().item()
        correct_all+= torch.eq(pred, data.y.reshape(-1, dataset.num_classes)).all(dim=1).sum().item()
        total += data.y.reshape(-1, dataset.num_classes).numel()
        total_all += len(torch.eq(pred, data.y.reshape(-1, dataset.num_classes)).all(dim=1))
        for i in range(dataset.num_classes):
            correct_bit[i]+=(pred[:,i] == data.y.reshape(-1, dataset.num_classes)[:,i]).sum().item()
    return correct / total, correct_all / total_all, [x / total_all for x in correct_bit]

def main(args):
    initialize_wandb(args)
    root = args.root
    lr = args.learning_rate
    epochs = args.epochs
    hidden_dim = args.hidden_dim
    
    dataset = XORBinaryDataset(root=root, highest_order=args.highest_order, transform=T.ToSparseTensor())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split dataset into training, validation, and test sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = GraphSAGE(in_dim=dataset[0].num_node_features, 
                 hidden_dim=hidden_dim, 
                 out_dim=dataset.num_classes,
                 num_layers=args.num_layers,
                 dropout=args.dropout
                 ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)#, weight_decay=5e-4)
    
    for epoch in range(1, epochs + 1):
        loss, train_acc, train_acc_all_bits, train_acc_each_bit = train(model, train_loader, optimizer, device, dataset)
        if epoch % 1 == 0:
            val_acc, val_acc_all_bits, val_acc_each_bit = test(model, val_loader, device, dataset)
            test_acc, test_acc_all_bits, test_acc_each_bit = test(model, test_loader, device, dataset)
            wandb.log({"Epoch": epoch, "Loss": loss, "Train_acc": train_acc, "Train_acc_all_bits": train_acc_all_bits, 
                       "Val_acc":val_acc, "Test_acc": test_acc, "Val_acc_all_bits":val_acc_all_bits, "Test_acc_all_bits": test_acc_all_bits})
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Train acc all bits: {train_acc_all_bits:.4f}, Val acc all bits: {val_acc_all_bits:.4f}, Test acc all bits: {test_acc_all_bits:.4f}')
            
    
if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='ELGraphSAGE Training')
    parser.add_argument('--root', type=str, default='/home/curie/ELGraphSAGE/dataset/edgelist', help='Root directory of dataset')
    parser.add_argument('--highest_order', type=int, default=8, help='Highest order for the EdgeListDataset')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=75, help='Hidden dimension size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()
    main(args)
    
    #python el_sage.py --root /home/curie/GraphSAGE/dataset/edgelist --highest_order 16 --learning_rate 0.0001 --epochs 500 --hidden_dim 75 --batch_size 64
    #root = '/home/curie/masGen/DataGen/dataset'
    #root = '/home/curie/GraphSAGE/dataset/edgelist'
