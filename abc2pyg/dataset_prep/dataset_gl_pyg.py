import os
import os.path as osp
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from typing import List
from torch_geometric.loader import DataLoader
#from loader.dataloader import DataLoader
from tqdm import tqdm

class LogicGateDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, highest_order = 16):
        self.highest_order = highest_order
        super(LogicGateDataset, self).__init__(root, transform, pre_transform, highest_order)
        self.data, self.slices = torch.load(self.processed_paths[0])
    '''
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
            - highest_order: highest_order of Mas.eqn
    '''
    @property
    def num_classes(self):
        return self.highest_order
        
    @property
    def raw_file_names(self):
        sorted(os.listdir(self.root_folders)) #processed folder

    @property
    def processed_file_names(self):
        return ['data_logic.pt']
    
    def process(self):
        data_list = []

        root_folders = sorted(os.listdir(self.root))
        for data_name in tqdm(root_folders):
            if 'processed' in data_name:
                pass
            else:
                folder = os.path.join(self.root, data_name)
                bprimtive_path = os.path.join(folder, 'bprimtive')
                mas_gl_path = os.path.join(folder, 'Mas'+str(self.highest_order)+'.gl')

                # Load labels
                with open(bprimtive_path, 'r') as f:
                    labels = [int(x) for x in f.read().strip().split()]
                    y = torch.zeros(self.highest_order)
                    y[torch.tensor(labels) - 1] = 1  # 1-hot encoding
                # Load graph edges and node features
                edge_index = []
                node_feat = {}
                with open(mas_gl_path, 'r') as f:
                    f.readline()
                    for line in f:
                        in_node1, in_node2, out_node, edge_type = line.strip().split()
                        in_node1, in_node2, out_node = int(in_node1), int(in_node2), int(out_node) 

                        if edge_type == '00': #Pi
                            edge_index.append([in_node1, out_node])
                            node_feat[out_node] = [99, 99, 99, 99]
                        elif edge_type == '11': #Po
                            edge_index.append([in_node1, out_node])
                            node_feat[out_node] = [-99, -99, -99, -99]
                        elif edge_type == '01': #AND     
                            edge_index.append([in_node1, out_node])
                            edge_index.append([in_node2, out_node])
                            node_feat[out_node] = [0, 1, 0, 1]#[0, 99, 0, 99]
                        elif edge_type == '10': #XOR     
                            edge_index.append([in_node1, out_node])
                            edge_index.append([in_node2, out_node])
                            node_feat[out_node] = [1, 0, 1, 0]#[99, 0, 99, 0]

                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                # Convert x_dict to a tensor
                num_nodes = edge_index.max().item() + 1
                x = torch.full((num_nodes, 4), 50).float()  # Initialize with 50
                
                for node, features in node_feat.items():
                    x[node-1] = torch.tensor(features).float()
                
                # # Check for any uninitialized values
                #if torch.isnan(x).any():
                # if 50 in x:
                #     raise ValueError("Uninitialized node features found in x tensor")
                
                # Create Data object
                #num_nodes = edge_index.max().item() + 1
                adj_t = SparseTensor.from_edge_index(edge_index)
                data = Data(edge_index=edge_index, y=y, x=x, adj_t=adj_t)
                data = data if self.pre_transform is None else self.pre_transform(data)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    dataset = LogicGateDataset(root = '/home/curie/masGen/DataGen/dataset16', highest_order = 16) #, transform=T.ToSparseTensor())
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])
    # dataloader = DataLoader(dataset, batch_size=32)
    
    # from sklearn.model_selection import train_test_split
    # train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    # train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
    
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader= DataLoader(val_dataset, batch_size=32, shuffle=False)
    # test_loader= DataLoader(test_dataset, batch_size=32, shuffle=False)
    # for batch in train_loader:
    #     print(batch)
