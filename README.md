# 深度学习基础大作业 — `dl-/` 代码说明

端到端链路：**读取日频面板 → Transformer 回归打分 → （可选）历史回测或下一交易日推演**。  
默认股票池与作业 PDF 一致：除 ST、北交所外的 A 股（`basic.csv` ∩ 本地 `daily/`，详见下文）。

数据根目录：环境变量 **`DL_DATA_ROOT`**（默认：`/home/lhr/my_stuff/fundamentals_for_deep_learning/data`）。

---

## 1. 仓库结构（整合后）

```
dl-/
├── README.md                       # 本说明
├── .gitignore                      # 忽略 outputs/ 等运行产物
├── requirements.txt
├── calendar_trading.py             # trade_cal → 下一开市日（给 workflow_cli 用）
├── data_processor.py               # 面板加载、滑动窗口、标签、标准化（无训练逻辑）
├── train_transformer_baseline.py    # 模型定义 + DDP 训练 + 打分导出（--workflow）
├── eval_metrics.py                 # 验证 IC 等指标
├── backtest_score_weighted.py       # score_weighted 历史回测
├── portfolio_sim.py                # 持仓 JSON、整手撮合、T+1（与回测对齐）
├── next_trade_suggestions.py       # CLI：摘要或 --state 细单
├── workflow_cli.py                 # **推荐**：backtest | predict-next 一条命令编排
├── examples/
│   ├── README.md                   # JSON 示例说明
│   ├── state_empty.json            # 未建仓
│   └── state_holding.json          # 已建仓
└── outputs/                        # 运行产物（已默认 .gitignore）
```

**职责划分（避免重复）**

| 模块 | 职责 |
|------|------|
| `data_processor.py` | 仅数据与张量流水线 |
| `train_transformer_baseline.py` | 训练、验证、导出 pred_score |
| `backtest_score_weighted.py` | 多日打分 + 收盘价 → 净值曲线 |
| `portfolio_sim.py` | 单日 score_weighted 撮合、状态读写、**infer_position_mode_from_state_dict** |
| `next_trade_suggestions.py` | 调用 `portfolio_sim`，面向用户的交互与导出 |
| `workflow_cli.py` | 编排子进程（torchrun/python），**不写第二套训练/回测逻辑** |
| `calendar_trading.py` | 仅日历，不写业务策略 |

---

## 2. 依赖

```bash
cd /path/to/fundamentals_for_deep_learning/dl-
pip install -r requirements.txt
```

---

## 3. 推荐入口：`workflow_cli.py`

自动用 **`trade_cal.csv`** 计算：**验证/回测段的第一个锚定日 = `train-end` 的下一开市日**，无需手填 `val-start`。

### 3.1 历史回测

训练锚定：**[train-start, train-end]**；打分 + 回测锚定：**(train-end 的下个开市日, backtest-end]**。

```bash
export DL_DATA_ROOT=/path/to/data

python workflow_cli.py backtest \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-05-18 \
  --backtest-end 2026-05-25 \
  --scores-out outputs/wf_scores.csv \
  --out-curve outputs/wf_equity.csv \
  --out-summary outputs/wf_summary.csv \
  --launcher torchrun --nproc 8 \
  -- --epochs 50 --batch-size 1024 --stock-pool all
```

`--` 后为传给 `train_transformer_baseline.py` 的参数；不写则用脚本默认。**单进程**：`--launcher python`。

### 3.2 末日推理 + 下一日操作 JSON

`predict-next` 训练后用 **面板末日** 截面打分，再读 **状态 JSON**，写出 **`operations.json`**（含 `orders`、`portfolio_after_close` 等）。

```bash
python workflow_cli.py predict-next \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-05-19 \
  --export-scores outputs/wf_live_scores.csv \
  --state-in examples/state_empty.json \
  --ops-out outputs/next_operations.json \
  --next-trade-date 20260521 \
  --launcher torchrun --nproc 8 \
  -- --epochs 50 --batch-size 1024 --stock-pool all
```

**不重训**：`--skip-train` 写在 **`workflow_cli` 的参数区**，且须在传给训练脚本的 **`--` 之前**，例如：
`… --skip-train --launcher torchrun --nproc 8 -- --epochs …`。**不要**写成 `--stock-pool all--skip-train`（会与 `all` 粘成非法 stock-pool）。
workflow 也会对误粘的写法尽量自动解压并提示。

```bash
python workflow_cli.py predict-next \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-05-19 \
  --export-scores outputs/wf_live_scores.csv \
  --state-in examples/state_empty.json \
  --ops-out outputs/next_operations.json \
  --next-trade-date 20260521 \
  --skip-train \
  --launcher torchrun --nproc 8 \
  -- --epochs 50 --batch-size 1024 --stock-pool all
```

- **未建仓**：`examples/state_empty.json`  
- **已建仓**：`examples/state_holding.json`  
- **`--next-trade-date`**：语义上的「下一交易日」（例如打分面板末日后一天 `20260521`）。若有 `daily/{{该日}}.csv` 则用其收盘；**若无**，默认自动改用 **不大于该日的最近本地 CSV（多为面板末日）** 做占位成交；运行时在终端打印 **`[占位价格]`** 一行，且输出 JSON 含 `pricing_trade_date_note`。**必须使用该日收盘做真实推演**时请加 **`--strict-next-trade-csv`**。  
- 省略 `--next-trade-date` 时，仍尝试根据 `daily/` 上「比打分最后一日更晚」的首个文件名推断（若数据中尚无更晚文件则须手填语义日并利用占位规则）。  
- **仅重演指令、不重训**：**`--skip-train`**（见上，须在 **`--` 前**）；`--export-scores` 须指向已有 CSV。

---

## 4. `examples/` 状态 JSON

见 **`examples/README.md`**。

- **`state_empty.json`**：全现金、`sellable`/`locked` 均为 `{}`。  
- **`state_holding.json`**：含 `sellable`（早盘可卖）与 `locked`（昨买锁定）。  

可选 **`position_status`**：`empty`|`holding`（与股数不符时告警，仍以股数为准）。

---

## 5. 直接使用 `train_transformer_baseline.py`（进阶）

与子命令等价关系：

| CLI | `--workflow` | 用途 |
|-----|----------------|------|
| 手写命令 | **`backtest`** | 验证集 IC + 导出带 `label_return` 的打分（需 **train-end < val-start**） |
| 手写命令 | **`predict-next`** | 仅用监督全集训练 + `--export-infer-anchor auto` 末日截面 |

手写示例与边界条件（最后一天无 T+1、推理导出等）见脚本顶部 docstring；日常优先 **`workflow_cli`**，减少日期算错。

---

## 6. 股票、时间与作业约束（防信息泄漏）

本节对应大作业基本要求：**因果链清晰、不得在回测链路中混入「未来才知」的信息**。

| 要点 | 本仓库实现位置 |
|------|----------------|
| **训练 / 验证时间切分** | `train_transformer_baseline.py`：**`train-end` < `val-start`**（锚定样本日）；打乱仅发生在训练样本内，`create_sequences` 按时间切块。 |
| **标签与未来收益** | `data_processor.construct_labels`：`shift(-horizon)`，单行标签仅依赖 **该行之后 horizon 个交易日**的行情；最晚可监督锚定日随 `horizon`、`load-end`/`daily/` 覆盖面而定。 |
| **标准化（关键）** | `DataProcessor.fit_standardize`：**仅对训练集展开的 `(N×T,F)` fit `StandardScaler`**，再 transform 训练与验证；**禁止**在未按训练窗截断的全量时间上估计均值与方差。末日推理时使用 **同一 scaler**（仅用训练样本拟合）做 `transform`。 |
| **回测侧的分数** | 验证段导出的 **`pred_score` 只对各自 `trade_date`** 使用模型；默认 **`--score-lag 1`**：第 D 日交易用上一交易日截面分数，不把「同一天收盘后才算出的分数」误当开盘已知。 |

- **股票池**：默认 **`--stock-pool all`**（`basic` 中非北交所 + 与 `daily` 交集）；子池同上。  

更多：`data_processor` 日志中的 **`[daily] … 面板最后交易日`**。

---

## 7. `next_trade_suggestions.py`（不用 workflow 时）

```bash
python next_trade_suggestions.py --scores OUTPUT.csv --data-root "$DL_DATA_ROOT" \
  --state examples/state_holding.json \
  --next-trade-date 20260520 \
  --score-lag 1 --n 30 --k 10 \
  --out-orders outputs/orders.csv --out-next-state outputs/state_after.json
```

摘要模式不传 `--state`。

---

## 8. `backtest_score_weighted.py`（不用 workflow 时）

```bash
python backtest_score_weighted.py \
  --scores outputs/wf_scores.csv \
  --data-root "$DL_DATA_ROOT" \
  --cash 1000000 --n 30 --k 10 --score-lag 1 \
  --commission-bps 3 \
  --out-curve outputs/equity_curve.csv \
  --out-summary outputs/backtest_summary.csv
```

---

## 9. 交易费用（回测 / 细单推演一致）

- **印花税**：卖出成交金额 × **0.1%**（买入不收）。  
- **过户费**：仅 **上交所、代码六位以 `60` 开头**（如 `600000.SH`，不含深证、不含 `688`）；买卖双方按成交金额 × **万分之六**，单笔 **不足 1 元按 1 元**。  
- **券商佣金**：买卖双方均按 **`--commission-bps`**（基点，万三即 `3`）× 成交金额，**单笔不足 5 元按 5 元**；设为 `0` 则不计佣金，**仍会收印花税与沪市过户**。  
- 净值曲线中的 **`commission` 列** = 当日上述费用合计（元）。
- **`--n` / `--k`**：候选池 Top-n、池内持仓 Top-k；权重 ∝ **`pred_score`**（池内平移后归一）。

---

## 10. 默认模型规模

在典型 **`feat_dim=10`**、`d_model=128`、`layers=2`、`nhead=4`、`dim_ff=256` 下，裸参约 **26.7 万**，FP32 权重约 **1.1 MB**。噪声环境下不必盲目加深；先试特征、正则、时间切分，再微调容量与学习率。

---

## 附录：交割与实盘说明

**价格来源**：若有 `daily/{--next-trade-date}.csv`，用其收盘价；否则用不大于该交易日的最近 `daily/*.csv` 占位（见 `pricing_trade_date`）。费用规则见「交易费用」；无实盘下单接口；报告与复现请以 **`requirements.txt` + README** 为准。
