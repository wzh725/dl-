# `dl-/` 使用说明

本目录提供完整流程：`train.py` 训练打分、`backtest.py` 历史回测、`predict.py` 次日建议，统一由 `workbench.py` 编排。

---

## 1) 代码结构

| 文件 | 作用 |
|---|---|
| `data_preprocess.py` | 读取日线与侧车数据，构造标签和序列，做标准化。 |
| `train.py` | 训练 Transformer，导出验证打分或末日推理打分。 |
| `backtest.py` | 基于打分做历史回测，输出净值曲线与摘要指标。 |
| `predict.py` | 基于分数快照 + 账户状态，生成下一交易日操作建议。 |
| `workbench.py` | 推荐入口，串联训练、回测、推演。 |

---

## 2) 环境准备

```bash
conda activate dl
cd /path/to/fundamentals_for_deep_learning/dl-
pip install -r requirements.txt
export DL_DATA_ROOT=/path/to/fundamentals_for_deep_learning/data
```

---

## 2.5) 最新推荐命令（先看这里）

### A. 训练 + 回测（最新推荐）

```bash
python workbench.py backtest \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-01-06 \
  --val-start 2026-01-07 \
  --backtest-end 2026-04-07 \
  --scores-out outputs/wf_scores.csv \
  --out-curve outputs/wf_equity.csv \
  --out-summary outputs/wf_summary.csv \
  --n-pool 30 \
  --k-hold 8 \
  --score-lag 1 \
  --trade-price-col open \
  --commission-rate 0.0003 \
  --launcher torchrun \
  --nproc 8 \
  -- \
  --epochs 70 \
  --batch-size 1024 \
  --horizon 3 \
  --stock-pool all
```

### B. 仅生成次日指令（已有分数，不重训）

```bash
python workbench.py predict-next \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-01-06 \
  --export-scores outputs/wf_scores.csv \
  --state-in examples/state_holding.json \
  --ops-out outputs/final_ops.json \
  --next-trade-date 20260408 \
  --n-pool 30 \
  --k-hold 8 \
  --score-lag 1 \
  --trade-price-col open \
  --commission-rate 0.0003 \
  --skip-train
```

### C. `predict` 费率默认值说明（避免混淆）

- `predict.py --commission-rate` 默认是 `None`（不覆盖）
- `predict.py --commission-bps` 默认是 `None`（兼容旧参数）
- 当两者都不传时，使用 `state` JSON 里的 `commission_rate`
- 仅当你传了 `--commission-bps` 时，才按 `bps/10000` 覆盖 `--commission-rate`

---

## 3) 一条命令跑回测

```bash
python workbench.py backtest \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-01-06 \
  --val-start 2026-01-07 \
  --backtest-end 2026-04-07 \
  --scores-out outputs/wf_scores.csv \
  --out-curve outputs/wf_equity.csv \
  --out-summary outputs/wf_summary.csv \
  --n-pool 30 \
  --k-hold 8 \
  --score-lag 1 \
  --trade-price-col open \
  --commission-rate 0.0003 \
  --launcher torchrun \
  --nproc 8 \
  -- \
  --epochs 70 \
  --batch-size 1024 \
  --horizon 3 \
  --stock-pool all \
  --base-head-weight 0.80 \
  --rank-loss-weight 0.05 \
  --rank-loss-max-pairs 2048 \
  --feature-select-mode ic_prune \
  --keep-all-base-features \
  --sidecar-feature-budget 10 \
  --feature-report-path outputs/feature_report.json \
  --export-metrics-json outputs/wf_metrics.json \
  --export-daily-ic-csv outputs/wf_daily_ic.csv \
  --early-stop-min-epochs 25 \
  --early-stop-patience 12
```

---

## 4) 一条命令跑次日建议

```bash
python workbench.py predict-next \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-01-06 \
  --export-scores outputs/wf_scores.csv \
  --state-in examples/state_holding.json \
  --ops-out outputs/final_ops.json \
  --next-trade-date 20260408 \
  --n-pool 30 \
  --k-hold 8 \
  --score-lag 1 \
  --trade-price-col open \
  --commission-rate 0.0003 \
  --launcher torchrun \
  --nproc 8 \
  -- \
  --epochs 70 \
  --batch-size 1024 \
  --horizon 3 \
  --stock-pool all \
  --export-metrics-json outputs/wf_metrics.json
```

如已存在 `outputs/wf_scores.csv`，可加 `--skip-train` 跳过训练，仅生成建议（此时可不写 `--launcher/--nproc` 与 `--` 后训练参数）。

当 `--next-trade-date` 当天的 `daily/{date}.csv` 尚未落盘，且 `--trade-price-col open` 时，
脚本会显式采用“前一可用交易日 `close` 近似次日 `open`”的规则（终端会打印占位提示）。

---

## 5) 输出文件说明

| 文件 | 含义 |
|---|---|
| `outputs/wf_scores.csv` | 每日每股 `pred_score`（回测/推演输入）。 |
| `outputs/wf_equity.csv` | 回测逐日净值曲线（含成交额、持仓数、费用等）。 |
| `outputs/wf_summary.csv` | 回测摘要（含 `total_return`、`annual_return`、`max_drawdown`、`sharpe_ann_approx` 等）。 |
| `outputs/feature_report.json` | 特征筛选报告（保留列、剔除列、IC 打分）。 |
| `outputs/wf_metrics.json` | 训练阶段验证指标汇总（IC、RankIC、ICIR、方向胜率等）。 |
| `outputs/wf_daily_ic.csv` | 逐交易日 Pearson IC 序列（便于画时间序列图）。 |
| `outputs/final_ops.json` | 次日操作建议与收盘后状态。 |

---

## 6) 关键口径（建议写进实验报告）

- 时间因果：`score-lag=1` 表示交易日使用滞后 1 天打分；不足 lag 的起始日不交易。
- 标签防跨期：训练集自动做 purge，要求训练样本标签结束日严格早于验证起点。
- 推理窗口：`predict-next` 在末日推理时包含锚定日当日特征，用于下一交易日决策。
- 成交近似：若目标交易日 `open` 缺失（例如实时场景未落盘），显式使用前一可用交易日 `close` 近似该日 `open`。
- 费用模型：佣金双向（单笔最低 5 元）；印花税仅卖出 0.1%；过户费仅上证 60* 双向（按股数、单笔最低 1 元）。

---

## 7) 常用参数速查

`workbench.py`（`--` 前）：

- `--train-start --train-end --val-start --backtest-end`：样本锚定日区间
- `--n-pool --k-hold`：候选池大小与持仓只数
- `--score-lag`：打分滞后天数（推荐 1）
- `--trade-price-col`：撮合价格列（`open`/`close`）
- `--commission-rate`：券商佣金费率（小数，如 `0.0003`）
- `--skip-train`：仅 `predict-next` 可用，跳过训练

`train.py`（`--` 后）：

- `--window-len`：输入历史窗口长度（默认 `60`）
- `--horizon`：预测未来收益天数（默认 `3`）
- `--feature-select-mode`：`none` 或 `ic_prune`
- `--keep-all-base-features --sidecar-feature-budget`：base 全保留 + sidecar 限额
- `--rank-loss-weight --rank-loss-max-pairs`：排序辅助损失
- `--early-stop-*`：早停配置
- `--no-data-sidecars`：关闭侧车特征
