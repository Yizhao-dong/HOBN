import argparse
import copy
from typing import Any, Dict

import yaml

from hobn.trainer import run_cross_validation


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    simple = ["data_path", "output_dir", "seed", "folds", "device", "batch_size", "limit_subjects"]
    for key in simple:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.stage1_epochs is not None:
        cfg["training"]["stage1_epochs"] = args.stage1_epochs
    if args.stage2_epochs is not None:
        cfg["training"]["stage2_epochs"] = args.stage2_epochs
    if args.stage1_lr is not None:
        cfg["training"]["stage1_lr"] = args.stage1_lr
    if args.stage2_lr is not None:
        cfg["training"]["stage2_lr"] = args.stage2_lr
    if args.gamma is not None:
        cfg["population"]["gamma"] = args.gamma
    if args.population_topk is not None:
        cfg["population"]["topk"] = args.population_topk
    if args.apc_weight is not None:
        cfg["model"]["apc_weight"] = args.apc_weight
    if args.stage2_weight_decay is not None:
        cfg["training"]["stage2_weight_decay"] = args.stage2_weight_decay
    for key in ["trans_hidden", "trans_heads", "trans_layers", "hyper_hidden", "hyper_out", "embedding_dim"]:
        value = getattr(args, key)
        if value is not None:
            cfg["model"][key] = value
    if args.population_hidden is not None:
        cfg["population"]["hidden_dim"] = args.population_hidden
    if args.fc_dim is not None:
        cfg["population"]["fc_dim"] = args.fc_dim
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train HOBN on concatenated multi-atlas brain data.")
    parser.add_argument("--config", default="configs/abide.yaml")
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-subjects", type=int)
    parser.add_argument("--stage1-epochs", type=int)
    parser.add_argument("--stage2-epochs", type=int)
    parser.add_argument("--stage1-lr", type=float)
    parser.add_argument("--stage2-lr", type=float)
    parser.add_argument("--stage2-weight-decay", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--population-topk", type=int)
    parser.add_argument("--apc-weight", type=float)
    parser.add_argument("--trans-hidden", type=int)
    parser.add_argument("--trans-heads", type=int)
    parser.add_argument("--trans-layers", type=int)
    parser.add_argument("--hyper-hidden", type=int)
    parser.add_argument("--hyper-out", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--population-hidden", type=int)
    parser.add_argument("--fc-dim", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    run_cross_validation(cfg)


if __name__ == "__main__":
    main()
