
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, to_undirected
from torch_geometric.nn import TransformerConv, GATv2Conv, PNAConv

from CustomLayers.Sinusoidal_Positional_Encoding import PositionalEncoding
from CustomLayers.Attention_Aggregator import AttentionPooling
import CustomLayers.NBEATS_Module

import DrawLog
import CustomLayers

import CONFIG
import PATH

torch.manual_seed(CONFIG.INITIAL_SEED)
torch.cuda.manual_seed(CONFIG.INITIAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TemporalSpatialGateFusion(nn.Module):
    def __init__(self, hidden_dim, one_minus_alpha=True):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.one_minus_alpha = one_minus_alpha

    def forward(self, temporal_feat, spatial_feat):

        alpha = self.gate(torch.cat([temporal_feat, spatial_feat], dim=-1))

        if self.one_minus_alpha:
            fusion = alpha * temporal_feat + (1.0 - alpha) * spatial_feat
        else:
            fusion = temporal_feat + alpha * spatial_feat

        return fusion, alpha

# Name: 1-GNN, 2-SeqNN

class CustumModule_GAT_LSTM(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_GAT_LSTM, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = nn.LSTM(CHANNEL * 4, CHANNEL * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bb_gnn_1 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = nn.LSTM(CHANNEL * 4, CHANNEL * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bb_gnn_2 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = nn.LSTM(CHANNEL * 4, CHANNEL * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bb_gnn_3 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)

        x_seq, _ = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq, _ = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq, _ = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_seq = x_seq[:, -1, :] # (num_nodes, time, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_PNA_LSTM(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_PNA_LSTM, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)

        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = nn.LSTM(CHANNEL * 4, CHANNEL * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bb_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = nn.LSTM(CHANNEL * 4, CHANNEL * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bb_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = nn.LSTM(CHANNEL * 4, CHANNEL * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bb_gnn_3 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)

        x_seq, _ = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq, _ = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq, _ = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_seq = x_seq[:, -1, :] # (num_nodes, time, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_XFMR_LSTM(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_XFMR_LSTM, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = nn.LSTM(CHANNEL * 4, CHANNEL * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bb_gnn_1 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = nn.LSTM(CHANNEL * 4, CHANNEL * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bb_gnn_2 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = nn.LSTM(CHANNEL * 4, CHANNEL * 2, num_layers=1, batch_first=True, bidirectional=True)
        self.bb_gnn_3 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)

        x_seq, _ = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq, _ = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq, _ = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_seq = x_seq[:, -1, :] # (num_nodes, time, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_GAT_XFMR(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_GAT_XFMR, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)

        
        self.pos_emb_1 = PositionalEncoding(CHANNEL * 4)
        xfmr_en_layer_1 = nn.TransformerEncoderLayer(
            d_model=CHANNEL * 4, 
            nhead=4, 
            dim_feedforward=CHANNEL * 4, 
            dropout=0.0,
            activation="gelu",
            batch_first=True
        )
        self.bb_seqnn_1 = nn.TransformerEncoder(xfmr_en_layer_1, num_layers=1)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_gnn_1 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        xfmr_en_layer_2 = nn.TransformerEncoderLayer(
            d_model=CHANNEL * 4, 
            nhead=4, 
            dim_feedforward=CHANNEL * 4, 
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.bb_seqnn_2 = nn.TransformerEncoder(xfmr_en_layer_2, num_layers=1)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_gnn_2 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        xfmr_en_layer_3 = nn.TransformerEncoderLayer(
            d_model=CHANNEL * 4, 
            nhead=4, 
            dim_feedforward=CHANNEL * 4, 
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.bb_seqnn_3 = nn.TransformerEncoder(xfmr_en_layer_3, num_layers=1)
        self.aggr_1 = AttentionPooling(CHANNEL * 4)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_gnn_3 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.pos_emb_1(x)
        x_seq = self.bb_seqnn_1(x_seq) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_seq = self.aggr_1(x_seq) # (num_nodes, time, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_PNA_XFMR(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_PNA_XFMR, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)

        
        self.pos_emb_1 = PositionalEncoding(CHANNEL * 4)
        xfmr_en_layer_1 = nn.TransformerEncoderLayer(
            d_model=CHANNEL * 4, 
            nhead=4, 
            dim_feedforward=CHANNEL * 4, 
            dropout=0.0,
            activation="gelu",
            batch_first=True
        )
        self.bb_seqnn_1 = nn.TransformerEncoder(xfmr_en_layer_1, num_layers=1)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        xfmr_en_layer_2 = nn.TransformerEncoderLayer(
            d_model=CHANNEL * 4, 
            nhead=4, 
            dim_feedforward=CHANNEL * 4, 
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.bb_seqnn_2 = nn.TransformerEncoder(xfmr_en_layer_2, num_layers=1)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        xfmr_en_layer_3 = nn.TransformerEncoderLayer(
            d_model=CHANNEL * 4, 
            nhead=4, 
            dim_feedforward=CHANNEL * 4, 
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.bb_seqnn_3 = nn.TransformerEncoder(xfmr_en_layer_3, num_layers=1)
        self.aggr_1 = AttentionPooling(CHANNEL * 4)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_gnn_3 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.pos_emb_1(x)
        x_seq = self.bb_seqnn_1(x_seq) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_seq = self.aggr_1(x_seq) # (num_nodes, time, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_XFMR_XFMR(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_XFMR_XFMR, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.pos_emb_1 = PositionalEncoding(CHANNEL * 4)
        xfmr_en_layer_1 = nn.TransformerEncoderLayer(
            d_model=CHANNEL * 4, 
            nhead=4, 
            dim_feedforward=CHANNEL * 4, 
            dropout=0.0,
            activation="gelu",
            batch_first=True
        )
        self.bb_seqnn_1 = nn.TransformerEncoder(xfmr_en_layer_1, num_layers=1)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_gnn_1 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        xfmr_en_layer_2 = nn.TransformerEncoderLayer(
            d_model=CHANNEL * 4, 
            nhead=4, 
            dim_feedforward=CHANNEL * 4, 
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.bb_seqnn_2 = nn.TransformerEncoder(xfmr_en_layer_2, num_layers=1)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_gnn_2 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        xfmr_en_layer_3 = nn.TransformerEncoderLayer(
            d_model=CHANNEL * 4, 
            nhead=4, 
            dim_feedforward=CHANNEL * 4, 
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.bb_seqnn_3 = nn.TransformerEncoder(xfmr_en_layer_3, num_layers=1)
        self.aggr_1 = AttentionPooling(CHANNEL * 4)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_gnn_3 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.pos_emb_1(x)
        x_seq = self.bb_seqnn_1(x_seq) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_seq = self.aggr_1(x_seq) # (num_nodes, time, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_GAT_NBEATS(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_GAT_NBEATS, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.TIME_STEP,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_1 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                 out_length=CONFIG.TIME_STEP,
                                                                 hidden_features=CHANNEL * 4,
                                                                 theta_features=CHANNEL * 2,)
        self.bb_gnn_2 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.OUT_LENGTH,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_3 = GATv2Conv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, add_self_loops=False)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, 1, 4C)
        x_seq = torch.squeeze(x_seq, dim=1) # (num_nodes, 1, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_PNA_NBEATS(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_PNA_NBEATS, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.TIME_STEP,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                 out_length=CONFIG.TIME_STEP,
                                                                 hidden_features=CHANNEL * 4,
                                                                 theta_features=CHANNEL * 2,)
        self.bb_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.OUT_LENGTH,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_3 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, 1, 4C)
        x_seq = torch.squeeze(x_seq, dim=1) # (num_nodes, 1, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_XFMR_NBEATS(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_XFMR_NBEATS, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.TIME_STEP,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_1 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                 out_length=CONFIG.TIME_STEP,
                                                                 hidden_features=CHANNEL * 4,
                                                                 theta_features=CHANNEL * 2,)
        self.bb_gnn_2 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.OUT_LENGTH,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_3 = TransformerConv(CHANNEL * 4, CHANNEL * 4, heads=4, edge_dim=CHANNEL * 4, concat=False, beta=True)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, 1, 4C)
        x_seq = torch.squeeze(x_seq, dim=1) # (num_nodes, 1, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_PNA_NBEATS_Abla_woGNN(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_PNA_NBEATS_Abla_woGNN, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.TIME_STEP,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                 out_length=CONFIG.TIME_STEP,
                                                                 hidden_features=CHANNEL * 4,
                                                                 theta_features=CHANNEL * 2,)
        self.bb_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.OUT_LENGTH,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_3 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        #x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        #x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = self.br_node_lin_2(future_fac)  # (num_nodes, 2C) -> (num_nodes, 4C)
        #future_fac = self.br_node_norm_1(future_fac)
        #future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        #future_fac = self.br_graph_norm_1(future_fac)
        #future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, 1, 4C)
        x_seq = torch.squeeze(x_seq, dim=1) # (num_nodes, 1, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        #x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x[:, -1, :]) # 2 * (num_nodes, 4C) -> (num_nodes, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_PNA_NBEATS_Abla_woEAGU(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_PNA_NBEATS_Abla_woEAGU, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.TIME_STEP,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                 out_length=CONFIG.TIME_STEP,
                                                                 hidden_features=CHANNEL * 4,
                                                                 theta_features=CHANNEL * 2,)
        self.bb_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.OUT_LENGTH,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_3 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, 1, 4C)
        x_seq = torch.squeeze(x_seq, dim=1) # (num_nodes, 1, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_PNA_NBEATS_Abla_woFGI(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_PNA_NBEATS_Abla_woFGI, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.TIME_STEP,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                 out_length=CONFIG.TIME_STEP,
                                                                 hidden_features=CHANNEL * 4,
                                                                 theta_features=CHANNEL * 2,)
        self.bb_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.OUT_LENGTH,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_3 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_1(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x, _ = self.bb_gate_2(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, 1, 4C)
        x_seq = torch.squeeze(x_seq, dim=1) # (num_nodes, 1, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        #x_seq = x_seq * fac_scaler_1
        x, _ = self.bb_gate_3(x_seq, x_spa) # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += x_skip
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

class CustumModule_PNA_NBEATS_Abla_woGF(nn.Module):
    def __init__(self, output_channels=CONFIG.DECOMP_DIM, node_feature_dim=CONFIG.DECOMP_DIM, edge_feature_dim=CONFIG.EDGE_ATTR_DIM, future_fac_dim=CONFIG.FUT_FAC_DIM):
        super(CustumModule_PNA_NBEATS_Abla_woGF, self).__init__()
        
        CHANNEL = 32

        self.br_node_lin_1 = nn.Linear(future_fac_dim, CHANNEL * 2)
        self.br_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.br_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.br_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.br_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.br_node_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.br_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_node_lin_1 = nn.Linear(node_feature_dim, CHANNEL * 2)
        self.bb_node_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.bb_node_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.edge_lin_1 = nn.Linear(edge_feature_dim, CHANNEL * 2)
        self.edge_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 4)
        self.edge_norm_1 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_3 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_2 = nn.LayerNorm(CHANNEL * 4)
        self.edge_lin_4 = nn.Linear(CHANNEL * 4, CHANNEL * 4)
        self.edge_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.bb_seqnn_1 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.TIME_STEP,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_1 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_1 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_1 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_2 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                 out_length=CONFIG.TIME_STEP,
                                                                 hidden_features=CHANNEL * 4,
                                                                 theta_features=CHANNEL * 2,)
        self.bb_gnn_2 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_2 = TemporalSpatialGateFusion(CHANNEL * 4)
        self.bb_graph_norm_2 = nn.LayerNorm(CHANNEL * 4)
        
        self.bb_seqnn_3 = CustomLayers.NBEATS_Module.NBEATS_PerCH(in_length=CONFIG.TIME_STEP,
                                                                  out_length=CONFIG.OUT_LENGTH,
                                                                  hidden_features=CHANNEL * 4,
                                                                  theta_features=CHANNEL * 2,)
        self.bb_gnn_3 = PNAConv(CHANNEL * 4, CHANNEL * 4,
                                aggregators=['mean', 'max', 'min', 'std',],
                                scalers=['identity', 'amplification',],
                                deg=torch.tensor(CONFIG.DEGREE),
                                edge_dim=CHANNEL * 4,)
        self.bb_gate_3 = TemporalSpatialGateFusion(CHANNEL * 4, one_minus_alpha=False)
        self.bb_graph_norm_3 = nn.LayerNorm(CHANNEL * 4)

        self.out_lin_1 = nn.Linear(CHANNEL * 4, CHANNEL * 2)
        self.out_lin_2 = nn.Linear(CHANNEL * 2, CHANNEL * 1)
        self.out_lin_3 = nn.Linear(CHANNEL * 1, output_channels)
        
    def forward(self, x, edge_index, edge_attr, future_fac, batch):

        x = F.tanh(self.bb_node_lin_1(x))  # (num_nodes, time, K) -> (num_nodes, time, 2C)
        x = F.gelu(self.bb_node_lin_2(x))  # (num_nodes, time, 2C) -> (num_nodes, time, 4C)
        x = self.bb_node_norm_1(x)
        x_skip = x[:, -12] # (num_nodes, 4C)

        edge_attr_1 = F.tanh(self.edge_lin_1(edge_attr))  # (num_edges, edge_feature_dim) -> (num_edges, 2C)
        edge_attr_1 = F.gelu(self.edge_lin_2(edge_attr_1))  # (num_edges, 2C) -> (num_edges, 4C)
        edge_attr_1 = self.edge_norm_1(edge_attr_1)
        edge_attr_2 = F.gelu(self.edge_lin_3(edge_attr_1))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_2 = self.edge_norm_2(edge_attr_2)
        edge_attr_3 = F.gelu(self.edge_lin_4(edge_attr_2))  # (num_edges, 4C) -> (num_edges, 4C)
        edge_attr_3 = self.edge_norm_3(edge_attr_3)
 
        x_seq = self.bb_seqnn_1(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_1(x_spa[t], edge_index, edge_attr_1), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x = x_seq + x_spa # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_1(x)

        x_seq = self.bb_seqnn_2(x) # (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x_spa = x.transpose(0, 1) # (num_nodes, time, 4C) -> (time, num_nodes, 4C)
        edge_attr_avg = torch.mean(torch.stack([edge_attr_1, edge_attr_2]), dim=0)
        x_spa = F.gelu(torch.stack(list(map(lambda t: self.bb_gnn_2(x_spa[t], edge_index, edge_attr_avg), range(x_spa.shape[0]))))) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_spa = x_spa.transpose(0, 1) # (time, num_nodes, 4C) -> (num_nodes, time, 4C)
        x = x_seq + x_spa # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_2(x)

        future_fac = F.gelu(self.br_node_lin_1(future_fac))  # (num_nodes, D_fac) -> (num_nodes, 2C)
        future_fac = F.gelu(self.br_node_lin_2(future_fac))  # (num_nodes, 2C) -> (num_nodes, 4C)
        future_fac = self.br_node_norm_1(future_fac)
        future_fac = F.gelu(self.br_gnn_1(future_fac, edge_index, edge_attr_1)) # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_1(future_fac)
        future_fac = self.br_gnn_2(future_fac, edge_index, edge_attr_avg) # (num_nodes, 4C) -> (num_nodes, 4C)
        fac_scaler_1 = F.sigmoid(future_fac)
        future_fac = F.gelu(future_fac)
        future_fac = self.br_graph_norm_2(future_fac)
        future_fac = F.gelu(self.br_node_lin_3(future_fac))  # (num_nodes, 4C) -> (num_nodes, 4C)
        future_fac = self.br_graph_norm_3(future_fac)

        x_seq = self.bb_seqnn_3(x) # (num_nodes, time, 4C) -> (num_nodes, 1, 4C)
        x_seq = torch.squeeze(x_seq, dim=1) # (num_nodes, 1, 4C) -> (num_nodes, 4C)
        edge_attr_avg2 = torch.mean(torch.stack([edge_attr_1, edge_attr_2, edge_attr_3]), dim=0)
        x_spa = F.gelu(self.bb_gnn_3(x_seq, edge_index, edge_attr_avg2)) # (num_nodes, 4C) -> (num_nodes, 4C)
        x_seq = x_seq * fac_scaler_1
        x = x_seq + x_spa # 2 * (num_nodes, time, 4C) -> (num_nodes, time, 4C)
        x = self.bb_graph_norm_3(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x += (x_skip + future_fac)
        x = F.gelu(self.out_lin_1(x))  # (num_nodes, 4C) -> (num_nodes, 2C)
        x = self.out_lin_2(x) # (num_nodes, 2C) -> (num_nodes, 1C)
        x = self.out_lin_3(x) # (num_nodes, 1C) -> (num_nodes, K)

        x = torch.cat((F.sigmoid(x[..., :1]), F.tanh(x[..., 1:])), dim=-1)

        return x

def get_model():
    #model = CustumModule_GAT_LSTM()
    #model = CustumModule_PNA_LSTM()
    #model = CustumModule_XFMR_LSTM()
    #model = CustumModule_GAT_XFMR()
    #model = CustumModule_PNA_XFMR()
    #model = CustumModule_XFMR_XFMR()
    #model = CustumModule_GAT_NBEATS()
    #model = CustumModule_PNA_NBEATS()
    model = CustumModule_XFMR_NBEATS()
    #model = CustumModule_PNA_NBEATS_Abla_woGNN()
    #model = CustumModule_PNA_NBEATS_Abla_woEAGU()
    #model = CustumModule_PNA_NBEATS_Abla_woFGI()
    #model = CustumModule_PNA_NBEATS_Abla_woGF()

    if(os.path.exists(PATH.MODEL_WEIGHT_PATH)):
        model.load_state_dict(torch.load(PATH.MODEL_WEIGHT_PATH))
        model.eval()
        print("Model Weight Loaded!")
    else:
        print("Model Weight Doesn't Exist!")
        DrawLog.init_log()
    return model

def save_model(model):
    torch.save(model.state_dict(), PATH.MODEL_WEIGHT_PATH)

    
if(__name__ == "__main__"):
    model = get_model()
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")

    print