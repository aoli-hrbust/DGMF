import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.modules.loss import BCEWithLogitsLoss
import math
from torch_geometric.data import Data
from .layers import *


def tricky_divide(x, y):
    return (x.t() / y).t()


def tricky_multiply(x, y):
    return (x.t() * y).t()


def add_noise(mat, stdev=0.001):
    dims = mat.shape[1]
    noise = stdev + torch.randn([dims], device=mat.device) * stdev * 1e-1
    noise = torch.diag(noise)
    noise = torch.unsqueeze(noise, dim=0)
    noise = torch.tile(noise, [mat.shape[0], 1, 1])
    return mat + noise


def reconstruction_loss(x_recon, x_true):
    x_recon = torch.cat(x_recon, dim=1)
    x_true = torch.cat(x_true, dim=1)
    return F.mse_loss(x_recon, x_true)


def all_loss(x_recon, x_true, logit_sp, logit_sh, labels, label_index):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss_rec = reconstruction_loss(x_recon, x_true)
    loss_sp = torch.mean(criterion(logit_sp[label_index], labels[label_index]))
    loss_sh = torch.mean(criterion(logit_sh[label_index], labels[label_index]))
    return loss_rec, loss_sp, loss_sh


class GMM(nn.Module):
    def __init__(self, n_components, dim):
        super().__init__()
        self.n_components = n_components
        self.dim = dim
        self.phi = nn.Parameter(torch.ones(n_components) / n_components)
        self.mu = nn.Parameter(torch.randn(n_components, dim))
        self.sigma = nn.Parameter(torch.stack([torch.eye(dim) for _ in range(n_components)]))

    def compute_energy(self, z):
        # z: [B, D]
        device = z.device
        dtype = z.dtype
        eps_eye = 1e-6 * torch.eye(self.dim, device=device, dtype=dtype)

        z = z.unsqueeze(1)           # [B, 1, D]
        mu = self.mu.unsqueeze(0)    # [1, K, D]
        sigma = self.sigma           # [K, D, D]

        sigma_stable = sigma + eps_eye.unsqueeze(0)
        inv_sigma = torch.linalg.inv(sigma_stable)                        # [K, D, D]
        det_sigma = torch.clamp(torch.linalg.det(sigma_stable), min=1e-6)  # [K]

        diff = z - mu               # [B, K, D]
        diff_vec = diff.unsqueeze(-1)  # [B, K, D, 1]

        maha = torch.matmul(inv_sigma.unsqueeze(0), diff_vec).squeeze(-1)  # [B, K, D]
        maha = torch.sum(maha * diff, dim=-1)                               # [B, K]

        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=dtype))
        log_prob = -0.5 * maha - 0.5 * self.dim * log_2pi
        log_prob -= 0.5 * torch.log(det_sigma.unsqueeze(0))

        log_pi = F.log_softmax(self.phi, dim=0)
        log_prob += log_pi.unsqueeze(0)

        energy_total = -torch.logsumexp(log_prob, dim=1)  # [B]
        penalty = torch.sum(1.0 / (torch.diagonal(sigma_stable, dim1=-2, dim2=-1) + 1e-12))

        return energy_total, penalty


class GMMEstimator(nn.Module):
    def __init__(self, args, input_dims, n_views, tau=1.0, use_gmm=True):
        super().__init__()
        self.args = args
        self.input_dims = input_dims
        self.n_views = n_views
        self.tau = tau
        self.use_gmm = use_gmm
        self.dataset = args.dataset
        # self.alpha_a = float(getattr(args, 'alpha_a', 0.0))
        self.alpha_a = args.alpha_a

        self.view_head_config = list(map(int, args.view_head_config.split(",")))
        self.view_layer_config = list(map(int, args.view_layer_config.split(",")))
        self.v_drop = args.v_drop

        self.hidden_last = self.view_layer_config[-1]

        self.cached_pruned_graphs = None

        if self.use_gmm:
            self.structural_attn = nn.ModuleList()
            for v in range(self.n_views):
                structural_attention_layers = nn.Sequential()
                gcn_layer = GCN(input_dims[v], input_dims[v], 128)
                structural_attention_layers.append(gcn_layer)
                self.structural_attn.append(structural_attention_layers)

            # View Attention
            input_dim_ = 128
            self.views_attn = nn.Sequential()
            for i in range(len(self.view_layer_config)):
                layer = ViewsAttentionLayer(
                    input_dim=input_dim_,
                    n_heads=self.view_head_config[i],
                    num_views=self.n_views,
                    attn_drop=self.v_drop,
                    residual=self.args.residual
                )
                self.views_attn.add_module(f"views_layer_{i}", layer)
                input_dim_ = self.view_layer_config[i]

            # GMM for each view
            self.view_gmms = nn.ModuleList([
                GMM(int(self.args.K), self.hidden_last) for _ in range(self.n_views)
            ])

        else:
            self.simple_gcns = nn.ModuleList([
                GCN(input_dims[v], input_dims[v], self.hidden_last) for v in range(self.n_views)
            ])

    def _graph_to_dense_adj(self, graph):
        """
        graph: PyG Data
        return:
            adj_dense: [N, N]
        """
        x = graph.x
        device = x.device
        dtype = x.dtype
        N = x.size(0)

        adj = torch.zeros((N, N), device=device, dtype=dtype)

        edge_index = graph.edge_index
        if hasattr(graph, 'edge_weight') and graph.edge_weight is not None:
            edge_weight = graph.edge_weight.to(device=device, dtype=dtype)
        else:
            edge_weight = torch.ones(edge_index.size(1), device=device, dtype=dtype)

        adj[edge_index[0], edge_index[1]] = edge_weight
        return adj

    def _dense_adj_to_graph(self, x, adj_dense):
        """
        x: [N, F]
        adj_dense: [N, N]
        return:
            PyG Data
        """
        edge_index = adj_dense.nonzero(as_tuple=False).t().contiguous()
        edge_weight = adj_dense[edge_index[0], edge_index[1]]
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    def _delete_edges_across_views(self, graphs):
        if self.alpha_a <= 0:
            return graphs

        if self.cached_pruned_graphs is not None:
            return self.cached_pruned_graphs

        dense_adjs = [self._graph_to_dense_adj(g) for g in graphs]  # list of [N, N]

        a = torch.stack(dense_adjs, dim=0).sum(dim=0)  # [N, N]

        pruned_graphs = []
        threshold = self.alpha_a * len(dense_adjs)

        for graph, adj_i in zip(graphs, dense_adjs):
            adj_new = adj_i.clone()
            adj_new[a < threshold] = 0.0

            if self.dataset not in ['BBCnews']:
                N = adj_new.size(0)
                eye = torch.eye(N, device=adj_new.device, dtype=adj_new.dtype)
                adj_new = adj_new + eye

            new_graph = self._dense_adj_to_graph(graph.x, adj_new)
            pruned_graphs.append(new_graph)

        self.cached_pruned_graphs = pruned_graphs
        return pruned_graphs

    def forward(self, graphs):
        if self.use_gmm:
            graphs = self._delete_edges_across_views(graphs)
            structural_out = []
            for t in range(self.n_views):
                x = self.structural_attn[t][0](graphs[t])
                structural_out.append(x)

            structural_outputs = torch.stack(structural_out, dim=1)  # [B, V, 128]
            views_out = self.views_attn(structural_outputs)          # [B, V, D]
            views_out = views_out.permute(1, 0, 2)                   # [V, B, D]
            energy_penalty = [gmm.compute_energy(z) for gmm, z in zip(self.view_gmms, views_out)]
            energy_views = [e[0] for e in energy_penalty]   # list of [B]
            penalties = [e[1] for e in energy_penalty]      # list of scalar

            energies = torch.stack(energy_views, dim=1)     # [B, V]
            total_energies = torch.sum(energies)
            total_penalty = sum(penalties)

            # ell_n^(v) = log p_n^(v) = -E_n^(v)
            log_densities = -energies                       # [B, V]
            weights = F.softmax(log_densities / self.tau, dim=1)  # [B, V]

            return views_out, weights, total_energies, total_penalty, energies

        else:
            simple_out = []
            for t in range(self.n_views):
                x = self.simple_gcns[t](graphs[t])   # [B, D]
                simple_out.append(x)

            views_out = torch.stack(simple_out, dim=0)  # [V, B, D]

            B = views_out.shape[1]
            V = views_out.shape[0]
            device = views_out.device
            dtype = views_out.dtype

            weights = torch.full(
                (B, V),
                fill_value=1.0 / V,
                device=device,
                dtype=dtype
            )
            total_energies = torch.tensor(0.0, device=device, dtype=dtype)
            total_penalty = torch.tensor(0.0, device=device, dtype=dtype)
            energies = torch.zeros(B, V, device=device, dtype=dtype)

            return views_out, weights, total_energies, total_penalty, energies


class SHSPModule(nn.Module):
    def __init__(self, n_views, hidden_dim, input_dims, num_classes, use_shsp=True):
        super().__init__()
        self.n_views = n_views
        self.hidden_dim = hidden_dim
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.use_shsp = use_shsp

        self.fus_encoder1 = Linerlayer(hidden_dim, hidden_dim)

        # shared / specific
        self.shared_encoder = Linerlayer(hidden_dim, hidden_dim // 2)
        self.specific_encoder = Linerlayer(hidden_dim, hidden_dim // 2)

        self.share_decoder = Linerlayer(hidden_dim // 2, hidden_dim)

        self.view_decoders = nn.ModuleList([
            Linerlayer(hidden_dim, d) for d in input_dims
        ])

        self.specific_classifiers = Linerlayer(hidden_dim // 2, num_classes)
        self.share_classifiers = Linerlayer(hidden_dim // 2, num_classes)

    def forward(self, views_out, weights):
        # views_out: [V, B, D]
        # weights:   [B, V]

        views_out_bvd = views_out.permute(1, 0, 2)  # [B, V, D]

        # reliability-aware fusion
        z_fus = weights.unsqueeze(-1) * views_out_bvd   # [B, V, D]
        z_fus = torch.sum(z_fus, dim=1)                 # [B, D]
        z_fus = self.fus_encoder1(z_fus)                # [B, D]

        if not self.use_shsp:
            z_sh_cls = self.shared_encoder(z_fus)       # [B, D/2]
            logit_sh = self.share_classifiers(z_sh_cls)

            x_hat = [dec(z_fus) for dec in self.view_decoders]

            z_sp = [torch.zeros_like(z_sh_cls) for _ in range(self.n_views)]
            z_sp_cls = torch.zeros_like(z_sh_cls)
            logit_sp = torch.zeros_like(logit_sh)

            return x_hat, z_sp_cls, z_sh_cls, logit_sp, logit_sh, z_sp

        z_sp = [self.specific_encoder(views_out[v]) for v in range(self.n_views)]  # V * [B, D/2]
        z_sh = self.shared_encoder(z_fus)                                           # [B, D/2]

        z_sp_de = [z + z_sh for z in z_sp]                  # V * [B, D/2]
        z_sp_de = [self.share_decoder(z) for z in z_sp_de] # V * [B, D]
        x_hat = [dec(z) for dec, z in zip(self.view_decoders, z_sp_de)]

        z_sp_cls = torch.sum(torch.stack(z_sp, dim=1), dim=1)   # [B, D/2]
        z_sh_cls = z_sh                                          # [B, D/2]

        logit_sp = self.specific_classifiers(z_sp_cls)           # [B, C]
        logit_sh = self.share_classifiers(z_sh_cls)              # [B, C]

        return x_hat, z_sp_cls, z_sh_cls, logit_sp, logit_sh, z_sp


class DGMF(nn.Module):
    def __init__(self, args, input_dims, hidden_dim, num_classes):
        super(DGMF, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.input_dims = input_dims
        self.n_views = len(input_dims)
        self.dropout = args.dropout
        self.l1 = args.l1
        self.l2 = args.l2
        self.tau = float(getattr(args, 'tau', 1.0))

        self.use_gmm = bool(getattr(args, 'use_gmm', True))
        self.use_shsp = bool(getattr(args, 'use_shsp', True))

        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.view_head_config = list(map(int, args.view_head_config.split(",")))
        self.view_layer_config = list(map(int, args.view_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.v_drop = args.v_drop

        hidden_last = self.view_layer_config[-1]

        self.gmm_estimator = GMMEstimator(
            args=args,
            input_dims=self.input_dims,
            n_views=self.n_views,
            tau=self.tau,
            use_gmm=self.use_gmm
        )

        self.shsp_module = SHSPModule(
            n_views=self.n_views,
            hidden_dim=hidden_last,
            input_dims=self.input_dims,
            num_classes=self.num_classes,
            use_shsp=self.use_shsp
        )

        self.fus_encoder = Linerlayer(hidden_last * self.n_views, hidden_last)
        self.cat_gmm_view_encoders = Linerlayer(sum(self.input_dims), hidden_dim[0])

    def forward(self, graphs):
        views_out, weights, total_energies, total_penalty, energies = self.gmm_estimator(graphs)

        x_hat, z_sp_cls, z_sh_cls, logit_sp, logit_sh, z_sp = self.shsp_module(
            views_out, weights
        )

        return x_hat, z_sp_cls, z_sh_cls, logit_sp, logit_sh, weights, total_energies, total_penalty, z_sp

    def feature_extractor(self, x, x_r):
        euclidean_dist = torch.norm(x - x_r, dim=1, keepdim=True) / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

        n1 = F.normalize(x, p=2, dim=1)
        n2 = F.normalize(x_r, p=2, dim=1)
        cosine_sim = torch.sum(n1 * n2, dim=1, keepdim=True)

        return torch.cat([euclidean_dist, cosine_sim], dim=1)




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


class Decomposition(nn.Module):
    def __init__(self, inputdim_list, outputdim):
        super(Decomposition, self).__init__()
        self.W = nn.ModuleList()
        for i in range(len(inputdim_list)):
            self.W.append(Linerlayer(inputdim_list[i], outputdim))

    def forward(self, feature_list):
        de_feature_list = []
        for i in range(len(feature_list)):
            x = self.W[i](feature_list[i], sparse=True)
            de_feature_list.append(x)
        return de_feature_list


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)