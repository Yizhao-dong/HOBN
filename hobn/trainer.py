import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .atlas import make_cross_atlas_group_labels, slice_atlas_tensor
from .data import BrainDataset, load_hobn_data, standardize_by_train
from .functional_systems import build_functional_group_result, save_group_report, summarize_group_counts
from .metrics import binary_metrics, summarize_metrics
from .model import HOBNModel, individual_loss
from .population import PopulationModel, make_population_features, make_population_graphs, population_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _loader(indices, timeseries, corr, labels, batch_size, shuffle, num_workers):
    dataset = BrainDataset(indices, timeseries, corr, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


def _corr_branches(corr: np.ndarray, specs) -> List[np.ndarray]:
    tensors = [corr[:, spec.start:spec.end, spec.start:spec.end] for spec in specs]
    tensors.append(corr)
    return tensors


def train_stage1(model, loader, optimizer, device, epochs, apc_weight, grad_clip, log_interval):
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for _, ts, corr, labels in loader:
            ts, corr, labels = ts.to(device), corr.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(ts, corr)
            loss, _ = individual_loss(out, labels, model, ts, corr, apc_weight)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % log_interval == 0:
            print(f"  stage1 epoch {epoch:03d}/{epochs} loss={np.mean(losses):.4f}", flush=True)


@torch.no_grad()
def collect_individual_outputs(model, loader, device, branch_count) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
    model.eval()
    all_labels = []
    probs = []
    embs = [[] for _ in range(branch_count)]
    branch_probs = [[] for _ in range(branch_count)]
    for _, ts, corr, labels in loader:
        ts, corr = ts.to(device), corr.to(device)
        out = model(ts, corr)
        fused = F.softmax(out["fused_logits"], dim=-1).cpu().numpy()
        probs.append(fused)
        all_labels.append(labels.numpy())
        for i, (logit, emb) in enumerate(zip(out["logits"], out["embeddings"])):
            branch_probs[i].append(F.softmax(logit, dim=-1).cpu().numpy())
            embs[i].append(emb.cpu().numpy())
    labels_np = np.concatenate(all_labels, axis=0)
    fused_probs = np.concatenate(probs, axis=0)
    emb_np = [np.concatenate(x, axis=0) for x in embs]
    branch_prob_np = [np.concatenate(x, axis=0) for x in branch_probs]
    return labels_np, emb_np, branch_prob_np, fused_probs


def train_stage2(
    pop_model,
    train_features,
    train_graphs,
    labels,
    optimizer,
    device,
    epochs,
    grad_clip,
    log_interval,
    all_features=None,
    all_graphs=None,
    n_train=None,
    intra_probs=None,
    test_labels=None,
    gamma=0.4,
):
    y = torch.from_numpy(labels).long().to(device)
    best = None
    pop_model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = pop_model(train_features, train_graphs)
        loss = population_loss(logits, y)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(pop_model.parameters(), grad_clip)
        optimizer.step()
        epoch_msg = f"  stage2 epoch {epoch:03d}/{epochs} loss={float(loss.detach().cpu()):.4f}"
        if all_features is not None:
            pop_probs = infer_population(pop_model, all_features, all_graphs, n_train=n_train)
            final_prob = np.mean(
                [gamma * p_intra + (1.0 - gamma) * p_inter for p_intra, p_inter in zip(intra_probs, pop_probs)],
                axis=0,
            )
            pred = final_prob.argmax(axis=1)
            metrics = binary_metrics(test_labels, pred, final_prob)
            metrics["best_epoch"] = epoch
            if best is None or metrics["acc"] > best["acc"]:
                best = metrics
            pop_model.train()
            epoch_msg += f" test_acc={metrics['acc']:.4f} best_acc={best['acc']:.4f}"
        if epoch == 1 or epoch == epochs or epoch % log_interval == 0:
            print(epoch_msg, flush=True)
    return best


@torch.no_grad()
def infer_population(pop_model, all_features, all_graphs, n_train):
    pop_model.eval()
    logits = pop_model(all_features, all_graphs)
    probs = [F.softmax(out[n_train:], dim=-1).cpu().numpy() for out in logits]
    return probs


def run_fold(cfg: Dict, fold_id: int, train_idx: np.ndarray, test_idx: np.ndarray, data, device: torch.device) -> Dict[str, float]:
    print(f"Fold {fold_id}: train={len(train_idx)} test={len(test_idx)}", flush=True)
    ts_std = standardize_by_train(data.timeseries, train_idx)
    train_loader = _loader(train_idx, ts_std, data.corr, data.labels, cfg["batch_size"], True, cfg["num_workers"])
    train_eval_loader = _loader(train_idx, ts_std, data.corr, data.labels, cfg["batch_size"], False, cfg["num_workers"])
    test_loader = _loader(test_idx, ts_std, data.corr, data.labels, cfg["batch_size"], False, cfg["num_workers"])

    systems = cfg["functional_systems"]["names"]
    group_result = build_functional_group_result(
        data.atlas_specs,
        systems,
        cfg=cfg.get("functional_systems", {}),
        corr=data.corr,
        train_idx=train_idx,
    )
    cross_labels = make_cross_atlas_group_labels(data.atlas_specs, systems, group_result.atlas_groups)
    report_path = save_group_report(group_result, cfg["output_dir"], fold_id)
    print(f"  functional groups saved: {report_path}", flush=True)
    print(f"  group counts: {summarize_group_counts(group_result)}", flush=True)
    model = HOBNModel(
        atlas_specs=list(data.atlas_specs),
        total_rois=data.timeseries.shape[1],
        time_points=data.timeseries.shape[2],
        num_classes=cfg["num_classes"],
        cross_group_labels=cross_labels,
        cfg=cfg["model"],
    ).to(device)

    optimizer1 = torch.optim.Adam(model.parameters(), lr=cfg["training"]["stage1_lr"])
    train_stage1(
        model,
        train_loader,
        optimizer1,
        device,
        cfg["training"]["stage1_epochs"],
        cfg["model"]["apc_weight"],
        cfg["training"]["grad_clip"],
        cfg["training"]["log_interval"],
    )

    y_train, emb_train, prob_train_branches, _ = collect_individual_outputs(model, train_eval_loader, device, model.num_branches)
    y_test, emb_test, prob_test_branches, prob_test_fused = collect_individual_outputs(model, test_loader, device, model.num_branches)

    corr_train = _corr_branches(data.corr[train_idx], data.atlas_specs)
    corr_test = _corr_branches(data.corr[test_idx], data.atlas_specs)
    pop_train_np, pop_test_np = make_population_features(
        emb_train,
        emb_test,
        corr_train,
        corr_test,
        y_train,
        cfg["population"]["fc_dim"],
    )
    pop_train_x, pop_train_h, pop_all_x, pop_all_h = make_population_graphs(
        pop_train_np,
        pop_test_np,
        cfg["population"]["topk"],
        device,
    )

    pop_model = PopulationModel(
        branch_count=model.num_branches,
        in_dim=pop_train_x[0].shape[1],
        hidden_dim=cfg["population"]["hidden_dim"],
        num_classes=cfg["num_classes"],
        dropout=cfg["model"]["trans_dropout"],
    ).to(device)
    optimizer2 = torch.optim.Adam(
        pop_model.parameters(),
        lr=cfg["training"]["stage2_lr"],
        weight_decay=cfg["training"]["stage2_weight_decay"],
    )
    best_stage2 = train_stage2(
        pop_model,
        pop_train_x,
        pop_train_h,
        y_train,
        optimizer2,
        device,
        cfg["training"]["stage2_epochs"],
        cfg["training"]["grad_clip"],
        cfg["training"]["log_interval"],
        all_features=pop_all_x,
        all_graphs=pop_all_h,
        n_train=len(train_idx),
        intra_probs=prob_test_branches,
        test_labels=y_test,
        gamma=cfg["population"]["gamma"],
    )

    pop_probs = infer_population(pop_model, pop_all_x, pop_all_h, n_train=len(train_idx))
    gamma = cfg["population"]["gamma"]
    final_prob = np.mean(
        [gamma * p_intra + (1.0 - gamma) * p_inter for p_intra, p_inter in zip(prob_test_branches, pop_probs)],
        axis=0,
    )
    pred = final_prob.argmax(axis=1)
    metrics = binary_metrics(y_test, pred, final_prob)
    if best_stage2 is not None:
        metrics = dict(best_stage2)
    print(
        "  fold metrics: "
        + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        flush=True,
    )
    return metrics


def run_cross_validation(cfg: Dict):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    data = load_hobn_data(
        cfg["data_path"],
        aal_size=cfg["atlas"]["aal_size"],
        cc200_size=cfg["atlas"]["cc200_size"],
        limit_subjects=cfg.get("limit_subjects"),
    )
    print(
        f"Loaded {len(data.labels)} subjects, timeseries={data.timeseries.shape}, corr={data.corr.shape}, "
        f"atlases={[f'{a.name}:{a.size}' for a in data.atlas_specs]}, device={device}",
        flush=True,
    )

    skf = StratifiedKFold(n_splits=cfg["folds"], shuffle=True, random_state=cfg["seed"])
    fold_metrics = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(data.timeseries, data.labels), start=1):
        set_seed(cfg["seed"] + fold_id)
        metrics = run_fold(cfg, fold_id, train_idx, test_idx, data, device)
        fold_metrics.append(metrics)

    summary = summarize_metrics(fold_metrics)
    print("Summary:", flush=True)
    for key, (mean, std) in summary.items():
        print(f"  {key}: {mean:.4f} +/- {std:.4f}", flush=True)
    with open(os.path.join(cfg["output_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": fold_metrics, "summary": summary}, f, indent=2)
    return fold_metrics, summary
