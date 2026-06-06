from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .atlas import AtlasSpec, slice_atlas_tensor
from .hypergraph import incidence_from_correlation, incidence_from_features, safe_prob
from .layers import AdaptiveHypergraphConv, TransEncoder


class BranchEncoder(nn.Module):
    def __init__(
        self,
        roi_count: int,
        time_points: int,
        num_classes: int,
        trans_hidden: int = 512,
        trans_heads: int = 8,
        trans_layers: int = 1,
        trans_dropout: float = 0.1,
        hyper_hidden: int = 512,
        hyper_out: int = 32,
        embedding_dim: int = 64,
        classifier_hidden: int = 64,
        spatial_topk: int = 8,
        temporal_topk: int = 3,
        lambda_sparse: float = 0.7,
        mu_group: float = 0.0,
        group_labels: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.roi_count = roi_count
        self.spatial_topk = spatial_topk
        self.temporal_topk = temporal_topk
        self.lambda_sparse = lambda_sparse
        self.mu_group = mu_group
        if group_labels is not None:
            self.register_buffer("group_labels", group_labels.clone().long())
        else:
            self.group_labels = None

        self.spatial_in = nn.Linear(roi_count, hyper_hidden)
        self.temporal_encoder = TransEncoder(time_points, trans_hidden, trans_heads, trans_layers, trans_dropout)
        self.temporal_in = nn.Linear(trans_hidden, hyper_hidden)

        self.spatial_hgc1 = AdaptiveHypergraphConv(hyper_hidden, hyper_hidden, trans_dropout)
        self.spatial_hgc2 = AdaptiveHypergraphConv(hyper_hidden, hyper_out, trans_dropout)
        self.temporal_hgc1 = AdaptiveHypergraphConv(hyper_hidden, hyper_hidden, trans_dropout)
        self.temporal_hgc2 = AdaptiveHypergraphConv(hyper_hidden, hyper_out, trans_dropout)
        self.node_fuse = nn.Sequential(
            nn.Linear(hyper_out * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, classifier_hidden),
            nn.GELU(),
            nn.Dropout(trans_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, timeseries: torch.Tensor, corr: torch.Tensor):
        h_spa = incidence_from_correlation(
            corr,
            topk=self.spatial_topk,
            lambda_sparse=self.lambda_sparse,
            mu_group=self.mu_group,
            group_labels=self.group_labels,
        )
        x_spa = self.spatial_in(corr)
        x_spa = self.spatial_hgc1(x_spa, h_spa)
        x_spa = self.spatial_hgc2(x_spa, h_spa)

        x_tem0 = self.temporal_encoder(timeseries)
        h_tem = incidence_from_features(x_tem0.detach(), topk=self.temporal_topk)
        x_tem = self.temporal_in(x_tem0)
        x_tem = self.temporal_hgc1(x_tem, h_tem)
        x_tem = self.temporal_hgc2(x_tem, h_tem)

        node = self.node_fuse(torch.cat([x_spa, x_tem], dim=-1))
        pooled = torch.cat([node.mean(dim=1), node.amax(dim=1)], dim=-1)
        logits = self.classifier(pooled)
        return logits, pooled, node


class HOBNModel(nn.Module):
    def __init__(
        self,
        atlas_specs: List[AtlasSpec],
        total_rois: int,
        time_points: int,
        num_classes: int,
        cross_group_labels: torch.Tensor,
        cfg: Dict,
    ):
        super().__init__()
        self.atlas_specs = atlas_specs
        self.num_branches = len(atlas_specs) + 1
        self.branch_names = [a.name for a in atlas_specs] + ["multi"]
        common = dict(
            time_points=time_points,
            num_classes=num_classes,
            trans_hidden=cfg["trans_hidden"],
            trans_heads=cfg["trans_heads"],
            trans_layers=cfg["trans_layers"],
            trans_dropout=cfg["trans_dropout"],
            hyper_hidden=cfg["hyper_hidden"],
            hyper_out=cfg["hyper_out"],
            embedding_dim=cfg["embedding_dim"],
            classifier_hidden=cfg["classifier_hidden"],
            spatial_topk=cfg["spatial_topk"],
            temporal_topk=cfg["temporal_topk"],
            lambda_sparse=cfg["lambda_sparse"],
        )
        self.single_branches = nn.ModuleList(
            [BranchEncoder(roi_count=spec.size, mu_group=0.0, **common) for spec in atlas_specs]
        )
        self.cross_branch = BranchEncoder(
            roi_count=total_rois,
            mu_group=cfg["mu_group"],
            group_labels=cross_group_labels,
            **common,
        )

    def _mask_inputs(self, timeseries: torch.Tensor, corr: torch.Tensor, mask_atlas: Optional[str]):
        if mask_atlas is None:
            return timeseries, corr
        ts = timeseries.clone()
        fc = corr.clone()
        target = None
        for spec in self.atlas_specs:
            if spec.name == mask_atlas:
                target = spec
                break
        if target is None:
            raise ValueError(f"Unknown atlas mask: {mask_atlas}")
        ts[:, target.start:target.end, :] = 0.0
        fc[:, target.start:target.end, :] = 0.0
        fc[:, :, target.start:target.end] = 0.0
        return ts, fc

    def forward(self, timeseries: torch.Tensor, corr: torch.Tensor, mask_atlas: Optional[str] = None):
        timeseries, corr = self._mask_inputs(timeseries, corr, mask_atlas)
        logits: List[torch.Tensor] = []
        embeddings: List[torch.Tensor] = []
        nodes: List[torch.Tensor] = []

        for spec, branch in zip(self.atlas_specs, self.single_branches):
            ts_z = slice_atlas_tensor(timeseries, self.atlas_specs, spec)
            corr_z = slice_atlas_tensor(corr, self.atlas_specs, spec)
            out, emb, node = branch(ts_z, corr_z)
            logits.append(out)
            embeddings.append(emb)
            nodes.append(node)

        out, emb, node = self.cross_branch(timeseries, corr)
        logits.append(out)
        embeddings.append(emb)
        nodes.append(node)
        fused_logits = torch.stack(logits, dim=0).mean(dim=0)
        return {"logits": logits, "fused_logits": fused_logits, "embeddings": embeddings, "nodes": nodes}


def individual_loss(outputs: Dict, labels: torch.Tensor, model: HOBNModel, timeseries: torch.Tensor, corr: torch.Tensor, apc_weight: float):
    branch_ce = torch.stack([F.cross_entropy(logits, labels) for logits in outputs["logits"]]).mean()
    fused_ce = F.cross_entropy(outputs["fused_logits"], labels)
    loss = fused_ce + branch_ce
    apc = torch.tensor(0.0, device=labels.device)
    if apc_weight > 0:
        full_prob = safe_prob(outputs["fused_logits"]).detach()
        for spec in model.atlas_specs:
            masked = model(timeseries, corr, mask_atlas=spec.name)
            apc = apc + F.kl_div(
                F.log_softmax(masked["fused_logits"], dim=-1),
                full_prob,
                reduction="batchmean",
            )
        apc = apc / len(model.atlas_specs)
        loss = loss + apc_weight * apc
    return loss, {"fused_ce": float(fused_ce.detach().cpu()), "branch_ce": float(branch_ce.detach().cpu()), "apc": float(apc.detach().cpu())}
