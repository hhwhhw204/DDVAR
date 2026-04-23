import os
import math
import sys
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch_geometric.nn import SAGPooling
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, GINConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
sys.path.append("/home/hyz/workspace/hhw/SecondWork/MainExperiments/Disgenet/train/code/model")

# GCN parameters
GCN_FEATURE_DIM = 480
GCN_OUTPUT_DIM = 128
DSSP_EMBEDDING = 24
ASSEQ_EMBEDDING = 48
PSSM_DIM = 20 + 20



class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, head):
        super(GAT, self).__init__()
        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=head, concat=True, dropout=0.25)
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim*head)
        self.gat2 = GATv2Conv(hidden_dim*head, out_dim, heads=head, concat=False, dropout=0.25)
        self.bn2 = nn.BatchNorm1d(out_dim)
    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.bn1.reset_parameters()
        self.gat2.reset_parameters()
        self.bn2.reset_parameters()            
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.relu(x)  
        x = self.bn1(x)
        x = self.gat2(x, edge_index)
        x = self.relu(x) 
        x = self.bn2(x)
        return x
    


class Attention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=0.5)
        self.layernorm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        x_out,attn_map = self.attn(x, x, x) 
        x_out = self.layernorm(x + x_out)
        return x_out, attn_map
    

def RotaryPositionEmbedding(x):
    """
    对输入 x 应用 RoPE 位置编码
    :param x: 输入张量，形状为 (batch_size, seq_len, dim)
    :return: 应用 RoPE 的张量，形状为 (batch_size, seq_len, dim)
    """
    device = x.device  # Get the device of the input tensor
    seq_len = x.shape[1]
    dim = x.shape[2]
    
    assert dim % 2 == 0, "Embedding dimension must be even."

    half_dim = dim // 2
    freq = 10000 ** (-torch.arange(0, half_dim, 2).float() / half_dim).to(device)  # Ensure freq is on the same device
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # Ensure position is on the same device
    angle = position * freq
    angle = angle.repeat(1, 2)

    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x_rotated = torch.cat([ 
        x1 * torch.cos(angle) - x2 * torch.sin(angle),
        x1 * torch.sin(angle) + x2 * torch.cos(angle)
    ], dim=-1)
    return x_rotated



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb_dssp_ref = nn.Embedding(9, DSSP_EMBEDDING)
        self.emb_dssp_alt = nn.Embedding(9, DSSP_EMBEDDING)
        self.emb_asseq_ref = nn.Embedding(21, ASSEQ_EMBEDDING)
        self.emb_asseq_alt = nn.Embedding(21, ASSEQ_EMBEDDING)

        self.gat1 = GAT(GCN_FEATURE_DIM, GCN_OUTPUT_DIM, GCN_OUTPUT_DIM, head=4)
        self.gat2 = GAT(GCN_FEATURE_DIM, GCN_OUTPUT_DIM, GCN_OUTPUT_DIM, head=4)        
        
        self.attn1 = Attention(DSSP_EMBEDDING + ASSEQ_EMBEDDING + PSSM_DIM, num_heads=4)
        self.attn1_2 = Attention(DSSP_EMBEDDING + ASSEQ_EMBEDDING + PSSM_DIM, num_heads=4)
        self.attn2 = Attention(DSSP_EMBEDDING + ASSEQ_EMBEDDING + PSSM_DIM, num_heads=4)
        self.attn2_2 = Attention(DSSP_EMBEDDING + ASSEQ_EMBEDDING + PSSM_DIM, num_heads=4)

        self.dropout = nn.Dropout(0.25)
        # prop_dim = DSSP_EMBEDDING + ASSEQ_EMBEDDING + PSSM_DIM
        # self.fc1 = nn.Linear(2 * GCN_OUTPUT_DIM + 4 * prop_dim, 32)
        self.fc1 = nn.Linear((GCN_OUTPUT_DIM + DSSP_EMBEDDING + ASSEQ_EMBEDDING + PSSM_DIM)*4, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU()


    def forward(self, ref_data, alt_data):
        ref_esm_repr, ref_edge_index, _, ref_dssp, ref_asseq, ref_pssm = ref_data.esm_repr, ref_data.edge_index, ref_data.edge_attr, ref_data.dssp, ref_data.asseq, ref_data.pssm
        alt_esm_repr, alt_edge_index, _, alt_dssp, alt_asseq, alt_pssm = alt_data.esm_repr, alt_data.edge_index, alt_data.edge_attr, alt_data.dssp, alt_data.asseq, alt_data.pssm

        # GAT
        ref_graph = self.gat1(ref_esm_repr, ref_edge_index)
        alt_graph = self.gat1(alt_esm_repr, alt_edge_index)

        batch_size = ref_data.num_graphs 
        graph_repr_ref = ref_graph.view(batch_size, -1, GCN_OUTPUT_DIM)
        graph_repr_alt = alt_graph.view(batch_size, -1, GCN_OUTPUT_DIM)

        ref_mean = torch.mean(graph_repr_ref, dim=1)
        alt_mean = torch.mean(graph_repr_alt, dim=1)
        diff = alt_mean - ref_mean   # mut - wt
        abs_diff = torch.abs(diff)  # |mut - wt|
        graph_repr = torch.cat([ref_mean, alt_mean, diff, abs_diff], dim=1)

        # 其他特征
        ref_dssp = self.emb_dssp_ref(ref_dssp).view(-1, DSSP_EMBEDDING)
        alt_dssp = self.emb_dssp_alt(alt_dssp).view(-1, DSSP_EMBEDDING)
        ref_asseq = self.emb_asseq_ref(ref_asseq).view(-1, ASSEQ_EMBEDDING)
        alt_asseq = self.emb_asseq_alt(alt_asseq).view(-1, ASSEQ_EMBEDDING)

        # 旋转位置编码
        ref_dssp = RotaryPositionEmbedding(ref_dssp.view(-1,200,DSSP_EMBEDDING))
        alt_dssp = RotaryPositionEmbedding(alt_dssp.view(-1,200,DSSP_EMBEDDING))
        ref_asseq = RotaryPositionEmbedding(ref_asseq.view(-1,200,ASSEQ_EMBEDDING)) 
        alt_asseq = RotaryPositionEmbedding(alt_asseq.view(-1,200,ASSEQ_EMBEDDING))
        ref_pssm = RotaryPositionEmbedding(ref_pssm.view(-1,200,PSSM_DIM))
        alt_pssm = RotaryPositionEmbedding(alt_pssm.view(-1,200,PSSM_DIM))
        ref_properties = torch.cat((ref_dssp, ref_asseq, ref_pssm), dim=2)
        alt_properties = torch.cat((alt_dssp, alt_asseq, alt_pssm), dim=2)
        
        # attn
        ref_properties,_ = self.attn1_2(self.attn1(ref_properties)[0])
        alt_properties,_ = self.attn1_2(self.attn1(alt_properties)[0])

        ref_prop = torch.mean(ref_properties, dim=1)
        alt_prop = torch.mean(alt_properties, dim=1)
        diff = alt_prop - ref_prop  # mut - wt
        abs_diff = torch.abs(diff)  # |mut - wt|
        prop = torch.cat([ref_prop, alt_prop, diff, abs_diff], dim=1)

        # concat
        x = torch.cat([graph_repr, prop],dim=1)    
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x.squeeze(1), 0
