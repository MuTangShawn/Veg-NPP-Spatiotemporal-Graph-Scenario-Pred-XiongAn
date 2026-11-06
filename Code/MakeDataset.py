
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

import tqdm

import CONFIG

class CustomGraphDataset(Dataset):
    def __init__(self, graph_list, repeat_ratio=1, transform=None, pre_transform=None):
        super(CustomGraphDataset, self).__init__(graph_list, transform, pre_transform)
        self.graph_list = graph_list
        self.repeat_ratio = repeat_ratio

    def len(self):
        return len(self.graph_list) * self.repeat_ratio

    def get(self, idx):
        real_index = idx // self.repeat_ratio
        graph_data = self.graph_list[real_index]

        node_features = graph_data['node_features']
        edge_index = graph_data['edge_index']
        edge_attr = graph_data['edge_attr']
        future_fac = graph_data['future_fac']
        ground_truth = graph_data['ground_truth']

        graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            future_fac=torch.tensor(future_fac, dtype=torch.float),
            y=torch.tensor(ground_truth, dtype=torch.float),
        )
        return graph_data

def calc_max_deg(graph_dataset):
    max_degree = 0
    for data in tqdm.tqdm(graph_dataset):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes,
                   dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)

    for data in tqdm.tqdm(graph_dataset):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes,
                   dtype=torch.long)
        deg += torch.bincount(d, minlength=max_degree + 1)

    return deg

def make_dataset(train_data_list, val_data_list, test_data_list):
    train_dataset = CustomGraphDataset(graph_list=train_data_list, repeat_ratio=CONFIG.TRAIN_DATASET_REPEAT_RATIO)
    val_dataset = CustomGraphDataset(graph_list=val_data_list, repeat_ratio=CONFIG.VAL_DATASET_REPEAT_RATIO)
    test_dataset = CustomGraphDataset(graph_list=test_data_list, repeat_ratio=CONFIG.TEST_DATASET_REPEAT_RATIO)

    CONFIG.DEGREE = calc_max_deg(train_dataset)
    print(CONFIG.DEGREE)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.VAL_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.TEST_BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader