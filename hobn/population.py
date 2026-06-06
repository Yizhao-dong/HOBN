from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypergraph import population_incidence
from .layers import AdaptiveHypergraphConv


def upper_triangle_features(corr: np.ndarray) -> np.ndarray:
    n = corr.shape[-1]
    iu = np.triu_indices(n, k=1)
    return corr[:, iu[0], iu[1]]


@dataclass
class FCSelector:
    selected: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.selected].astype(np.float32, copy=False)


def fit_fc_selector(fc_train: np.ndarray, y_train: np.ndarray, dim: int) -> FCSelector:
    dim = min(dim, fc_train.shape[1])
    y_train = y_train.astype(np.int64)
    class0 = fc_train[y_train == 0]
    class1 = fc_train[y_train == 1]
    if len(class0) == 0 or len(class1) == 0:
        score = np.nanvar(fc_train, axis=0)
    else:
        mean0 = np.nanmean(class0, axis=0)
        mean1 = np.nanmean(class1, axis=0)
        var0 = np.nanvar(class0, axis=0)
        var1 = np.nanvar(class1, axis=0)
        denom = np.sqrt(var0 / max(len(class0), 1) + var1 / max(len(class1), 1) + 1e-6)
        score = np.abs(mean1 - mean0) / denom
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    selected = np.argpartition(score, -dim)[-dim:]
    selected = np.sort(selected)
    return FCSelector(selected=selected)


class PopulationBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        try:
            from dhg.nn import HGNNPConv

            self.backend = "dhg"
            self.hgc1 = HGNNPConv(in_dim, hidden_dim, drop_rate=dropout, is_last=False)
            self.hgc2 = HGNNPConv(hidden_dim, hidden_dim, drop_rate=dropout, is_last=False)
        except Exception:
            self.backend = "native"
            self.hgc1 = AdaptiveHypergraphConv(in_dim, hidden_dim, dropout)
            self.hgc2 = AdaptiveHypergraphConv(hidden_dim, hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.backend == "dhg":
            if x.dim() != 2 or h.dim() != 2:
                raise ValueError("DHG population branch expects x=[N,F] and h=[N,E].")
            from dhg import Hypergraph

            residual = self.input_proj(x)
            hg = Hypergraph(num_v=x.shape[0], e_list=incidence_to_e_list(h), device=x.device)
            x = self.hgc1(x, hg)
            x = self.hgc2(x, hg)
            x = x + residual
            return self.classifier(x)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        residual = self.input_proj(x)
        x = self.hgc1(x, h)
        x = self.hgc2(x, h)
        x = x + residual
        return self.classifier(x.squeeze(0))


def incidence_to_e_list(h: torch.Tensor) -> List[List[int]]:
    h_cpu = h.detach().to("cpu")
    edges: List[List[int]] = []
    for edge_id in range(h_cpu.shape[1]):
        vertices = torch.nonzero(h_cpu[:, edge_id] > 0, as_tuple=False).view(-1).tolist()
        if vertices:
            edges.append(vertices)
    if not edges:
        raise ValueError("Population incidence matrix has no hyperedges.")
    return edges


class PopulationModel(nn.Module):
    def __init__(self, branch_count: int, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList(
            [PopulationBranch(in_dim, hidden_dim, num_classes, dropout=dropout) for _ in range(branch_count)]
        )

    def forward(self, features: List[torch.Tensor], incidences: List[torch.Tensor]) -> List[torch.Tensor]:
        return [branch(x, h) for branch, x, h in zip(self.branches, features, incidences)]


def population_loss(logits: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    branch_ce = torch.stack([F.cross_entropy(out, labels) for out in logits]).mean()
    fused_ce = F.cross_entropy(torch.stack(logits, dim=0).mean(dim=0), labels)
    return branch_ce + fused_ce


def make_population_features(
    embeddings_train: List[np.ndarray],
    embeddings_test: List[np.ndarray],
    corr_train_branches: List[np.ndarray],
    corr_test_branches: List[np.ndarray],
    y_train: np.ndarray,
    fc_dim: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    train_features: List[np.ndarray] = []
    test_features: List[np.ndarray] = []
    for emb_tr, emb_te, corr_tr, corr_te in zip(embeddings_train, embeddings_test, corr_train_branches, corr_test_branches):
        fc_train = upper_triangle_features(corr_tr)
        fc_test = upper_triangle_features(corr_te)
        selector = fit_fc_selector(fc_train, y_train, fc_dim)
        fc_train_sel = selector.transform(fc_train)
        fc_test_sel = selector.transform(fc_test)
        train_features.append(np.concatenate([emb_tr.astype(np.float32), fc_train_sel], axis=1))
        test_features.append(np.concatenate([emb_te.astype(np.float32), fc_test_sel], axis=1))
    return train_features, test_features


def make_population_graphs(
    train_features: List[np.ndarray],
    test_features: List[np.ndarray],
    topk: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    train_x, train_h, all_x, all_h = [], [], [], []
    for feat_tr, feat_te in zip(train_features, test_features):
        tr = torch.from_numpy(feat_tr).float().to(device)
        te = torch.from_numpy(feat_te).float().to(device)
        h_tr = population_incidence(tr, None, topk=topk).to(device)
        h_all = population_incidence(tr, te, topk=topk).to(device)
        train_x.append(tr)
        train_h.append(h_tr)
        all_x.append(torch.cat([tr, te], dim=0))
        all_h.append(h_all)
    return train_x, train_h, all_x, all_h
