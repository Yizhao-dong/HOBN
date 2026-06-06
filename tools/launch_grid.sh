#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}"
CONFIG="${CONFIG:-configs/abide.yaml}"
PYTHON="${PYTHON:-python}"

cd "$PROJECT_DIR"
mkdir -p outputs/tuning logs

run_one() {
  local name="$1"
  local gpu="$2"
  shift 2
  screen -dmS "$name" bash -lc \
    "CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} train.py --config ${CONFIG} --output-dir outputs/tuning/${name} $* 2>&1 | tee logs/${name}.log"
}

run_one hobn_repro_g0 0 --stage1-epochs 8 --stage2-epochs 160 --batch-size 16 --device cuda --trans-hidden 64 --trans-heads 4 --hyper-hidden 64 --hyper-out 16 --embedding-dim 32 --population-hidden 256 --fc-dim 4096 --gamma 0.0 --apc-weight 0.0 --stage2-lr 0.0003 --stage2-weight-decay 0.0001 --population-topk 3
run_one hobn_alt_g1 1 --stage1-epochs 8 --stage2-epochs 160 --batch-size 16 --device cuda --trans-hidden 64 --trans-heads 4 --hyper-hidden 64 --hyper-out 16 --embedding-dim 32 --population-hidden 192 --fc-dim 2048 --gamma 0.0 --apc-weight 0.0 --stage2-lr 0.0002 --stage2-weight-decay 0.0001 --population-topk 5
run_one hobn_alt_g2 2 --stage1-epochs 10 --stage2-epochs 160 --batch-size 16 --device cuda --trans-hidden 64 --trans-heads 4 --hyper-hidden 64 --hyper-out 16 --embedding-dim 32 --population-hidden 256 --fc-dim 4096 --gamma 0.0 --apc-weight 0.0 --stage2-lr 0.0002 --stage2-weight-decay 0.0001 --population-topk 5

screen -ls
