from typing import Optional

import torch
import torch.nn.functional as F


def incidence_from_scores(scores: torch.Tensor, topk: int, include_self: bool = True) -> torch.Tensor:
    """Build one compact hyperedge per center vertex from score top-k neighbors."""
    if scores.dim() != 3:
        raise ValueError(f"scores must be [B, N, N], got {tuple(scores.shape)}")
    bsz, n_nodes, _ = scores.shape
    k = min(max(int(topk), 1), max(n_nodes - 1, 1))
    scores = scores.clone()
    eye = torch.eye(n_nodes, device=scores.device, dtype=torch.bool).unsqueeze(0)
    scores = scores.masked_fill(eye, -torch.finfo(scores.dtype).max)
    nn_idx = scores.topk(k=k, dim=-1).indices
    h = scores.new_zeros(bsz, n_nodes, n_nodes)
    edge_ids = torch.arange(n_nodes, device=scores.device).view(1, n_nodes, 1).expand(bsz, n_nodes, k)
    h[torch.arange(bsz, device=scores.device).view(-1, 1, 1), nn_idx, edge_ids] = 1.0
    if include_self:
        h[torch.arange(bsz, device=scores.device).view(-1, 1), torch.arange(n_nodes, device=scores.device), torch.arange(n_nodes, device=scores.device)] = 1.0
    return h


def incidence_from_correlation(
    corr: torch.Tensor,
    topk: int,
    lambda_sparse: float = 0.7,
    mu_group: float = 0.0,
    group_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sparse/FGHM-inspired hyperedge construction from absolute correlations."""
    scores = corr.abs()
    if group_labels is not None and mu_group > 0:
        labels = group_labels.to(scores.device)
        group_scores = torch.zeros_like(scores)
        for group_id in labels.unique(sorted=True):
            mask = labels == group_id
            if mask.any():
                group_mean = scores[:, :, mask].mean(dim=-1, keepdim=True)
                group_scores[:, :, mask] = group_mean
        scores = lambda_sparse * scores + mu_group * group_scores
    return incidence_from_scores(scores, topk=topk, include_self=True)


def pairwise_negative_distance(x: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, x, p=2.0)
    return -dist


def incidence_from_features(x: torch.Tensor, topk: int) -> torch.Tensor:
    return incidence_from_scores(pairwise_negative_distance(x), topk=topk, include_self=True)


def population_incidence(train_x: torch.Tensor, query_x: Optional[torch.Tensor], topk: int) -> torch.Tensor:
    """Build train-only or train+query population incidence without query labels."""
    if query_x is None:
        return incidence_from_features(train_x.unsqueeze(0), topk=topk).squeeze(0)

    all_x = torch.cat([train_x, query_x], dim=0)
    n_train = train_x.shape[0]
    n_total = all_x.shape[0]
    k = min(max(int(topk), 1), max(n_train - 1, 1))
    h = all_x.new_zeros(n_total, n_total)

    train_scores = -torch.cdist(train_x, train_x, p=2.0)
    train_scores.fill_diagonal_(-torch.finfo(train_scores.dtype).max)
    train_nn = train_scores.topk(k=k, dim=-1).indices
    for center in range(n_train):
        h[center, center] = 1.0
        h[train_nn[center], center] = 1.0

    query_scores = -torch.cdist(query_x, train_x, p=2.0)
    query_nn = query_scores.topk(k=min(k, n_train), dim=-1).indices
    for offset in range(query_x.shape[0]):
        edge = n_train + offset
        h[edge, edge] = 1.0
        h[query_nn[offset], edge] = 1.0
    return h


def safe_prob(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1).clamp_min(1e-8)
