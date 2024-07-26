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

class LogicGatePaddingDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, highest_order = 16):
        self.highest_order = highest_order
        super(LogicGatePaddingDataset, self).__init__(root, transform, pre_transform, highest_order)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(self.data is not None)
    '''
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
            - highest_order: highest_order of Mas.eqn
    '''
    @property
    def num_classes(self):
        # predicting # of highest_order -1
        #return self.highest_order -1
        return self.highest_order  
        
    @property
    def raw_file_names(self):
        sorted(os.listdir(self.root_folders)) #processed folder

    @property
    def processed_file_names(self):
        return ['data_logic_padding.pt']
    
    def process(self):
        if not os.path.exists(self.processed_paths[0]):
            self.max_num_nodes = self.find_max_num_nodes()
        padded_data_list = []
        root_folders = sorted(os.listdir(self.root))
        print(f'Padding data')
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
                    #predict # of highest order -1
                    # y = torch.zeros(self.highest_order-1)
                    # if len(labels) > 1:
                    #      y[torch.tensor(labels[:-1]) - 1] = 1  # 1-hot encoding
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
                pad_size = self.max_num_nodes - num_nodes
                # Pad node features
                x_padded = torch.cat([x, torch.full((pad_size, x.size(1)), 50.0)], dim=0)
                # Pad edge indices
                edge_index_padded = edge_index
                edge_index_padded = torch.cat([edge_index_padded, torch.tensor([[num_nodes + i, num_nodes + i] for i in range(pad_size)], dtype=torch.long).t().contiguous()], dim=1)
                # Pad adjacency matrix
                adj_t = SparseTensor.from_edge_index(edge_index)
                adj_t_padded = adj_t.to_dense()
                adj_t_padded = torch.nn.functional.pad(adj_t_padded, (0, pad_size, 0, pad_size), value=0)
                adj_t_padded = SparseTensor.from_dense(adj_t_padded)
                padded_data = Data(x=x_padded, edge_index=edge_index_padded, y=y, adj_t=adj_t_padded)
                padded_data_list.append(padded_data)

        data, slices = self.collate(padded_data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f'Saved padded data in {self.processed_paths[0]}')
    
    def find_max_num_nodes(self):
        root_folders = sorted(os.listdir(self.root))
        max_num_nodes = 0
        print(f'Calculating max_num_nodes')
        for data_name in tqdm(root_folders):
            if 'processed' in data_name:
                pass
            else:
                folder = os.path.join(self.root, data_name)
                mas_gl_path = os.path.join(folder, 'Mas'+str(self.highest_order)+'.gl')
                edge_index = []
                with open(mas_gl_path, 'r') as f:
                    f.readline()
                    for line in f:
                        in1, in2, out, _ = line.strip().split()
                        edge_index.append([int(in1), int(out)])
                        edge_index.append([int(in2), int(out)])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                # Convert x_dict to a tensor
                num_nodes = edge_index.max().item() + 1
                if num_nodes > max_num_nodes:
                    max_num_nodes = num_nodes
        self.max_num_nodes = max_num_nodes
        print(f'Finished! max_num_nodes: {max_num_nodes}')
        return max_num_nodes
    
if __name__ == '__main__':
    dataset = LogicGatePaddingDataset(root = '/home/curie/masGen/DataGen/dataset8', highest_order = 8) #, transform=T.ToSparseTensor()
    print(dataset.find_max_num_nodes())
    print(dataset[0])
    print(dataset[1])
    '''
    dataloader = DataLoader(dataset, dataset, batch_size=32)
    
    from sklearn.model_selection import train_test_split
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(dataset, train_dataset, batch_size=32, shuffle=True)
    val_loader= DataLoader(dataset, val_dataset, batch_size=32, shuffle=False)
    test_loader= DataLoader(dataset, test_dataset, batch_size=32, shuffle=False)
    for batch in train_loader:
        print(batch)
    '''
    dataloader = DataLoader(dataset, batch_size=32)
    
    from sklearn.model_selection import train_test_split
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader= DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader= DataLoader(test_dataset, batch_size=32, shuffle=False)
    for batch in train_loader:
        print(batch)
