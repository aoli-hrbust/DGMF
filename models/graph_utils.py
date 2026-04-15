from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass
class CandidateGraph:
    row: torch.Tensor
    col: torch.Tensor
    base_weight: torch.Tensor
    num_nodes: int



def _symmetrize_edges(row: np.ndarray, col: np.ndarray, num_nodes: int):
    edge_set = set()
    for r, c in zip(row.tolist(), col.tolist()):
        if r == c:
            continue
        edge_set.add((int(r), int(c)))
        edge_set.add((int(c), int(r)))
    edge_set = sorted(edge_set)
    if len(edge_set) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    row = np.array([e[0] for e in edge_set], dtype=np.int64)
    col = np.array([e[1] for e in edge_set], dtype=np.int64)
    row = np.clip(row, 0, num_nodes - 1)
    col = np.clip(col, 0, num_nodes - 1)
    return row, col



def knn_graph_from_numpy(features: np.ndarray, k: int = 10, metric: str = 'cosine'):
    n = features.shape[0]
    k = int(min(max(k, 1), max(n - 1, 1)))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(features)
    indices = nbrs.kneighbors(return_distance=False)
    row = np.repeat(np.arange(n), k)
    col = indices[:, 1:].reshape(-1)
    row, col = _symmetrize_edges(row, col, n)
    return row, col



def build_candidate_graphs(
    features: List[torch.Tensor],
    k: int,
    metric: str = 'cosine',
    device: Optional[torch.device] = None,
) -> List[CandidateGraph]:
    graphs = []
    for feat in features:
        feat_np = feat.detach().cpu().numpy().astype(np.float32)
        row, col = knn_graph_from_numpy(feat_np, k=k, metric=metric)
        if device is None:
            device = feat.device
        row_t = torch.from_numpy(row).long().to(device)
        col_t = torch.from_numpy(col).long().to(device)
        base_weight = torch.ones(row_t.size(0), device=device)
        graphs.append(CandidateGraph(row=row_t, col=col_t, base_weight=base_weight, num_nodes=feat.shape[0]))
    return graphs



def refresh_candidate_graphs(
    clean_features: List[torch.Tensor],
    k: int,
    metric: str = 'cosine',
) -> List[CandidateGraph]:
    return build_candidate_graphs(clean_features, k=k, metric=metric, device=clean_features[0].device)



def normalize_edge_weight(
    num_nodes: int,
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = row.device
    if add_self_loops:
        self_loop = torch.arange(num_nodes, device=device, dtype=torch.long)
        row = torch.cat([row, self_loop], dim=0)
        col = torch.cat([col, self_loop], dim=0)
        weight = torch.cat([weight, torch.ones(num_nodes, device=device, dtype=weight.dtype)], dim=0)

    deg = torch.zeros(num_nodes, device=device, dtype=weight.dtype)
    deg.index_add_(0, row, weight)
    deg_inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
    norm_weight = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    return row, col, norm_weight



def labeled_edge_targets(row: torch.Tensor, col: torch.Tensor, labels: torch.Tensor, labeled_mask: torch.Tensor):
    valid = labeled_mask[row] & labeled_mask[col]
    if valid.sum().item() == 0:
        return valid, None
    same = (labels[row[valid]] == labels[col[valid]]).float()
    return valid, same
