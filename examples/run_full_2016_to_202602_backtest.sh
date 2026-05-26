#!/usr/bin/env bash
# 全量数据流水线（节选）：
# - 读取 daily（2016 起）、默认并入 moneyflow / metric / market；
# - 训练锚定：2016-01-01～2026-01-01；验证打分 + 组合回测锚定日历窗口：2026-01-02～2026-02-02；
# - 单机 8×GPU：`torchrun --nproc_per_node=8`
#
# 说明：SSE 日历中 2026-01-02～04 常为休市，验证段内**首日有样本的交易日**可能比 01-02 更晚，
#       但不会早于「字符串意义」上 ≥ val-start 的首个开市日。
#
set -euo pipefail
DL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DL_DATA_ROOT="${DL_DATA_ROOT:-$(cd "${DL_REPO}/.." && pwd)/data}"

cd "${DL_REPO}"

python workbench.py backtest \
  --data-root "${DL_DATA_ROOT}" \
  --train-start 2016-01-01 \
  --train-end 2026-01-01 \
  --val-start 2026-01-02 \
  --backtest-end 2026-02-02 \
  --scores-out outputs/full_train2016_scores_202602_val.csv \
  --out-curve outputs/full_train2016_equity_202602_val.csv \
  --out-summary outputs/full_train2016_summary_202602_val.csv \
  --cash 1000000 \
  --n-pool 30 \
  --k-hold 10 \
  --score-lag 1 \
  --commission-rate 0.0002 \
  --launcher torchrun \
  --nproc 8 \
  -- \
  --epochs 50 \
  --batch-size 1024 \
  --stock-pool all \
  --load-end 2026-04-01
