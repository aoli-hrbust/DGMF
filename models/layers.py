
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math
from torch_geometric.utils import softmax
from torch_scatter import scatter
import numpy as np

import copy

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

class Linerlayer(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(Linerlayer, self).__init__()
        self.weight = glorot_init(inputdim, outputdim)
    def forward(self, x, sparse=False):
        if sparse:
            x = torch.sparse.mm(x, self.weight)
        else:
            x = torch.mm(x, self.weight)
        return x

class DenseGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseGCNLayer, self).__init__()
        self.linear = Linerlayer(in_features, out_features)

    def forward(self, x, adj):
        # 计算 D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=-1)  # [N]
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

        out = torch.matmul(norm_adj, x)  # A @ X
        out = self.linear(out)
        return out

class DenseGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DenseGCN, self).__init__()
        self.conv1 = DenseGCNLayer(input_dim, hidden_dim)
        self.conv2 = DenseGCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj):  # adj is dense adjacency matrix, e.g., from A_pruned
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

class ViewsAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_views,
                attn_drop, 
                residual):
        super(ViewsAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_views = num_views
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_views, input_dim))  # 位置embedding信息[16, 128]
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))  # [128, 128]; W*Q
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input; [143, 16]: 143个节点，每个节点16个位置信息
        position_inputs = torch.arange(0,self.num_views).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)  # 重复143个节点; 每个节点有16个时间步
        views_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F]; 每个节点在各个时刻对应到的128维向量
        # views_inputs = inputs

        ##################################################################################################
        # 2: Query, Key based multi-head self attention. [143, 16, 128]
        q = torch.tensordot(views_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]; 第一个矩阵第2个维度，乘以，第二个矩阵的第0个维度
        k = torch.tensordot(views_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(views_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # q = views_inputs  # [N, T, F]; 第一个矩阵第2个维度，乘以，第二个矩阵的第0个维度
        # k = views_inputs  # [N, T, F]
        # v = views_inputs  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)  # 每个head的维度
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (self.num_views ** 0.5)  # Q*K


        # views_inputs_ = torch.cat(torch.split(views_inputs, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        # outputs = views_inputs_

        # 4: Masked (causal) softmax to compute attention weights. 目的是将之前没有出现的时间步，设置为0;
        diag_val = torch.ones_like(outputs[0])  # [16,16]的全1向量
        tril = torch.tril(diag_val)  # 下三角阵
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]  重复N次（2288）; [2288, 16, 16]
        padding = torch.ones_like(masks) * (-2**32+1)  # 负无穷
        outputs = torch.where(masks==0, padding, outputs)  # outputs中mask为0的地方，填充padding中负无穷的数值
        outputs = F.softmax(outputs, dim=2)  # output:[2288, 16, 16]
        self.attn_wts_all = outputs # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)  # dropout
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]  # (K*Q)*V; ouput-经过归一化后的attention系数[2288, 16, 16]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + views_inputs
        # ###################################################################################################
        #
        # outputs = views_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.0, activate_last=False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or activate_last:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ResidualGraphBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0, residual_alpha=0.2):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        self.residual_alpha = residual_alpha
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, h, h0, row, col, norm_weight):
        h_lin = self.linear(h)
        msg = h_lin[col] * norm_weight.unsqueeze(-1)
        agg = torch.zeros_like(h_lin)
        agg.index_add_(0, row, msg)
        agg = self.norm(agg)
        agg = F.relu(agg, inplace=True)
        agg = F.dropout(agg, p=self.dropout, training=self.training)
        return (1.0 - self.residual_alpha) * agg + self.residual_alpha * h0