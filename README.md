# `dl-/` 最小使用说明

本 README 只保留最小必要信息：**代码结构 + 一组 backtest 命令 + 一组 predict-next 命令 + 参数意义说明**。

---

## 1) 代码结构

| 文件 | 作用 |
|---|---|
| `data_preprocess.py` | 数据读取与清洗、侧车并入（moneyflow/metric/market）、样本构造（窗口、标签、标准化）。 |
| `train.py` | Transformer 训练入口（支持 DDP、多种特征筛选、早停、导出打分）。 |
| `backtest.py` | 用打分 CSV 做回测，输出净值曲线和摘要指标。 |
| `predict.py` | 用最新打分 + 账户状态 JSON 生成次日交易建议 JSON。 |
| `workbench.py` | 统一编排入口（`backtest` / `predict-next`），自动串联 `train.py` 与后处理脚本。 |

---

## 2) 环境与数据根

```bash
conda activate dl
cd /path/to/fundamentals_for_deep_learning/dl-
pip install -r requirements.txt

export DL_DATA_ROOT=/path/to/fundamentals_for_deep_learning/data
```

---

## 3) Backtest 命令（推荐入口）

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
  --commission-rate 0.0002 \
  --launcher torchrun \
  --nproc 8 \
  -- \
  --epochs 70 \
  --batch-size 1024 \
  --horizon 7 \
  --stock-pool all \
  --base-head-weight 0.80 \
  --rank-loss-weight 0.05 \
  --rank-loss-max-pairs 2048 \
  --feature-select-mode ic_prune \
  --keep-all-base-features \
  --sidecar-feature-budget 10 \
  --feature-report-path outputs/feature_report.json \
  --early-stop-min-epochs 25 \
  --early-stop-patience 12
```

---

## 4) Predict-next 命令（次日建议）

```bash
python workbench.py predict-next \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-01-06 \
  --export-scores outputs/wf_scores.csv \
  --state-in examples/state_empty.json \
  --ops-out outputs/final_ops.json \
  --next-trade-date 20260408 \
  --n-pool 30 \
  --k-hold 8 \
  --score-lag 1 \
  --trade-price-col open \
  --commission-rate 0.0002 \
  --skip-train \
  --launcher python \
  -- \
  --epochs 1 \
  --batch-size 1024 \
  --horizon 7 \
  --stock-pool all
```

---

## 5) 参数意义速查

### A. `workbench.py`（`--` 前）

| 参数 | 含义 |
|---|---|
| `--data-root` | 数据根目录（包含 `daily/`、`trade_cal.csv`、`basic.csv` 等）。 |
| `--train-start` / `--train-end` | 训练样本锚定日区间。 |
| `--val-start` / `--backtest-end` | 回测模式下验证/回测锚定日区间。`val-start` 可不写（自动取 train-end 后首个交易日）。 |
| `--scores-out` | 训练后导出的打分 CSV 路径。 |
| `--out-curve` / `--out-summary` | 回测净值与摘要输出路径。 |
| `--n-pool` / `--k-hold` | 每日候选池大小 / 持仓数量。 |
| `--score-lag` | 打分滞后天数（常用 1，避免未来信息）。 |
| `--trade-price-col` | 撮合价格列（常用 `open` 或 `close`）。 |
| `--commission-rate` | 券商佣金费率（小数，如万二=0.0002）。 |
| `--launcher` / `--nproc` | 训练启动方式（`torchrun` 或 `python`）与进程数。 |
| `--skip-train` | 仅在 `predict-next` 模式可用；跳过训练直接用已有分数。 |
| `--` | 分隔符，后面参数会透传给 `train.py`。 |

### B. `train.py`（`--` 后）

| 参数 | 含义 |
|---|---|
| `--epochs` / `--batch-size` / `--lr` | 训练轮数、批大小、学习率。 |
| `--horizon` | 预测未来收益天数（标签步长）。 |
| `--stock-pool` | 股票池：`all`/`hs300`/`cyb`/`kcb`。 |
| `--base-head-weight` | 双头融合中 base 头权重（侧车头权重 = `1 - 该值`）。 |
| `--rank-loss-weight` | 排序辅助损失权重（0 表示关闭）。 |
| `--rank-loss-max-pairs` | 每个 batch 的排序采样对数量上限。 |
| `--feature-select-mode` | 特征筛选策略：`none` 或 `ic_prune`。 |
| `--keep-all-base-features` | 保留全部 base 特征，只筛 sidecar（推荐）。 |
| `--sidecar-feature-budget` | sidecar 最多保留特征数（在 `ic_prune` 下生效）。 |
| `--feature-report-path` | 特征筛选报告 JSON 输出路径。 |
| `--early-stop-metric` / `--early-stop-patience` / `--early-stop-min-epochs` | 早停指标与触发条件。 |
| `--ddp-pipeline-cache-dir` | DDP 下预处理缓存目录，减少重复预处理耗时。 |
| `--no-data-sidecars` | 关闭侧车数据并入。 |
| `--load-end` | 读取日线数据的右边界（不写则自动按验证段扩展）。 |

---

## 6) 备注

- 建议先用 `workbench.py`，避免手工拼参数出错。
- 若出现参数粘连（如 `allpython`），通常是命令行续行或复制粘贴问题，重输一遍单行命令最稳。
