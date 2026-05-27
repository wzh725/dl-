# 每日交易系统说明文档

## 目录

1. [系统概览](#1-系统概览)
2. [文件结构](#2-文件结构)
3. [train.py 修改说明](#3-trainpy-修改说明)
4. [daily_trader.py 每日预测引擎](#4-daily_traderpy-每日预测引擎)
5. [generate_trade_plan.py 交易计划生成器](#5-generate_trade_planpy-交易计划生成器)
6. [portfolio_state.json 持仓状态](#6-portfolio_statejson-持仓状态)
7. [run_daily.bat 一键运行](#7-run_dailybat-一键运行)
8. [完整每日工作流程](#8-完整每日工作流程)
9. [命令行参数速查](#9-命令行参数速查)
10. [常见问题](#10-常见问题)

---

## 1. 系统概览

本系统基于训练好的 LSTM 多目标预测模型（AssociatedResidualNet），实现**每日全自动**的股票交易决策：

```
数据 16:00 更新 → 双击 run_daily.bat → 自动生成交易计划 CSV → 你在同花顺手动下单
```

核心流程图：

```
每日 16:05
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        run_daily.bat                            │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │  daily_trader.py     │    │  generate_trade_plan.py      │   │
│  │                      │    │                              │   │
│  │  加载最新日线数据     │    │  读取得分排名 CSV            │   │
│  │  → 技术指标计算      │───►│  → 对比 portfolio_state.json │   │
│  │  → 截面 Z-Score      │    │  → 计算买入/卖出清单         │   │
│  │  → 模型推理          │    │  → 计算买卖数量              │   │
│  │  → 得分排名 CSV      │    │  → 输出 trade_plan CSV       │   │
│  └──────────────────────┘    └──────────────────────────────┘   │
│                                                                 │
│  输出：                                                          │
│    daily_predictions_YYYYMMDD.csv     所有股票得分排名           │
│    trade_plans/trade_plan_YYYYMMDD.csv  今日买卖清单             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 文件结构

本次新增/修改的文件如下：

```
d:\lstm\
├── train.py                      # [修改] 训练时自动保存 scaler 和元数据
├── daily_trader.py               # [新建] 每日预测引擎
├── generate_trade_plan.py        # [新建] 交易计划生成器
├── portfolio_state.json          # [新建] 持仓状态持久化
├── run_daily.bat                 # [新建] 一键运行脚本
├── trade_plans/                  # [自动创建] 交易计划输出目录
│   └── trade_plan_YYYYMMDD.csv
└── daily_predictions_YYYYMMDD.csv  # [自动生成] 每日得分排名 CSV
```

与原有文件的关系：

```
                 train.py (训练)
                     │
                     ▼
            saved_models/production_v1/
            ├── best_model.pth      ──┐
            ├── scaler.pkl          ──┼── 被 daily_trader.py 加载
            └── model_meta.pkl      ──┘
```

---

## 3. train.py 修改说明

### 改动内容

在原有的训练脚本中，`train.py` 原来只保存模型权重（`best_model.pth`）。本次修改使其在保存最佳模型时**同时写入三个文件**：

| 文件 | 内容 | 用途 |
|------|------|------|
| `best_model.pth` | PyTorch 模型权重（State Dict） | 模型推理 |
| `scaler.pkl` | `sklearn.preprocessing.StandardScaler` 实例 | 对新数据执行与训练时完全相同的标准化 |
| `model_meta.pkl` | 字典格式的模型元数据 | 重建模型结构 + 记录训练参数 |

### model_meta.pkl 包含的字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `input_dim` | int | 输入特征维度 |
| `hidden_dim` | int | LSTM 隐藏层维度 |
| `num_layers` | int | ResidualLSTMBlock 层数 |
| `dropout` | float | LSTM 内部 Dropout |
| `fc_dropout` | float | 输出层 Dropout |
| `bidirectional` | bool | 是否双向 LSTM |
| `label_scale` | float | 标签缩放因子 |
| `seq_len` | int | 时间窗口长度 |
| `use_fundamental` | bool | 是否用了基本面 |
| `use_moneyflow` | bool | 是否用了资金流 |

### 为什么需要 scaler.pkl

模型在训练前对数据做了 **StandardScaler 标准化**（每维特征的均值变为 0，标准差变为 1）。每日预测时，新数据必须使用**完全相同**的 Scaler 来标准化，否则输入分布偏移会导致预测结果偏差。

### 关键代码位置

在 `train.py` 的保存最佳模型位置（约第 345 行），在 `torch.save(...)` 之后新增：

```python
# 保存 StandardScaler
scaler_path = os.path.join(args.save_dir, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(processor.scaler, f)

# 保存模型元数据
meta_path = os.path.join(args.save_dir, 'model_meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump({ ... }, f)
```

---

## 4. daily_trader.py 每日预测引擎

### 4.1 原理

`daily_trader.py` 是每日交易系统的**第一步**——将最新的全市场股票数据喂给训练好的模型，为每只股票生成一个"预期做多得分"。

#### 得分的含义

```
score = pred_hp − pred_op
```

- `pred_op`：模型预测的次日开盘价相对于当日收盘价的变化率
- `pred_hp`：模型预测的次日最高价相对于当日收盘价的变化率
- `score`：**预期次日能从开盘到最高点的做多空间**，值越大表示次日上涨潜力越大

#### 执行流程（6 步）

```
Step 1: load_artifacts()
    加载 best_model.pth + scaler.pkl + model_meta.pkl
    用 meta 中的参数重建 AssociatedResidualNet 结构

Step 2: load_and_prepare_data(lookback_days=45)
    ├─ 从 D:/zhw/A股数据/daily/ 加载最近 45 个交易日数据
    ├─ 自动寻找数据目录中最新的有效交易日（如 20260527）
    ├─ 通过交易日历文件 (trade_cal.csv) 精确定位交易日期
    ├─ 调用 DataProcessor.load_data() 做基础过滤：
    │    移除 ST 股票、北交所股票、停牌/涨跌停交易日
    ├─ 调用 DataProcessor.add_technical_indicators() 计算：
    │    MA5/10/20/60, EMA12/26, MACD(DIF/DEA), RSI(14)
    │    动量(5/10/20/60 日), 量比(5/10/20 日)
    │    量价相关性, 波动率(5/20 日), 量价配合度
    └─ 根据 meta 中的开关决定是否合并：
          基本面 (pe_ttm, pb, roe)  ← left join + 中位数填充
          资金流 (主力净/散户净比率) ← 大单净额/(总成交额+ε)

Step 3: cross_sectional_normalize()
    对每个交易日，所有股票在同一天内做截面 Z-Score：
        每只股票的特征值减去当天全市场均值，除以当天全市场标准差
    这消除了大盘日间波动的影响，使特征聚焦于截面上的相对强弱

Step 4: build_prediction_sequences()
    为每只有足够历史数据的股票，提取最近 seq_len 天的特征作为输入
    例如 seq_len=30，则取 T-29 到 T 共 30 天的特征矩阵 (30, N_features)

Step 5: standardize_sequences()
    使用训练时保存的 StandardScaler 对序列做标准化
    注意：这里是时间维度的标准化，与 Step 3 的截面标准化是不同的层次

Step 6: predict()
    分批送入模型做推理（默认 batch_size=256，避免 GPU 内存溢出）
    score = (pred_hp - pred_op) / label_scale
    按 score 降序排列，输出 daily_predictions_YYYYMMDD.csv
```

#### 双重标准化的设计

| 标准化层次 | 方法 | 目的 |
|-----------|------|------|
| **截面标准化** (Step 3) | 每日期货 Z-Score | 让模型看到相对强弱而非绝对数值 |
| **时间维标准化** (Step 5) | 训练时拟合的 StandardScaler | 让预测输入与训练时分布一致 |

两者缺一不可：截面标准化每次独立计算（无数据泄露），时间维标准化固定使用训练时的 Scaler。

#### 输出 CSV 格式

| 列名 | 说明 | 示例 |
|------|------|------|
| `rank` | 得分排名（1=最高分） | 1 |
| `ts_code` | 原生股票代码 | 600519.SH |
| `ths_code` | 同花顺格式（6 位数字） | 600519 |
| `name` | 股票名称 | 贵州茅台 |
| `score` | 预期做多得分 | 0.0234 |
| `pred_op` | 预测开盘价变化率 | -0.0012 |
| `pred_hp` | 预测最高价变化率 | 0.0222 |
| `close` | 当日收盘价 | 1680.50 |

### 4.2 使用方法

```bash
# 基础用法（使用默认模型目录）
python daily_trader.py

# 指定模型目录和数据路径
python daily_trader.py --model_dir ./saved_models/production_v1 --data_path D:/zhw/A股数据

# 指定输出文件名
python daily_trader.py --output my_predictions.csv
```

### 4.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | `./saved_models/production_v1` | 模型目录，需包含 best_model.pth, scaler.pkl, model_meta.pkl |
| `--data_path` | `D:/zhw/A股数据` | A股数据根目录 |
| `--output` | 自动命名 | 输出 CSV 路径，默认 `daily_predictions_YYYYMMDD.csv` |

---

## 5. generate_trade_plan.py 交易计划生成器

### 5.1 原理

`generate_trade_plan.py` 是每日交易系统的**第二步**——将 `daily_trader.py` 的得分排名转化为具体的买卖清单。

#### 核心逻辑

```
输入：
  ├── daily_predictions_YYYYMMDD.csv  (得分排名，按 score 降序)
  └── portfolio_state.json            (当前持仓)

处理流程：
  1. 取出得分排名前 N 的股票（默认 N=10）
  2. 计算持仓股票集合 与 前 N 名集合的差异：
      卖出集合 = 当前持仓 − 前 N 名   （持仓中但不够好的股票）
      买入集合 = 前 N 名 − 当前持仓   （值得买但尚未持有的股票）
  3. 每日最多调仓 K 只（默认 K=2）
      卖出：优先卖持仓市值最大的（释放更多资金）
      买入：优先买得分最高的
  4. 计算买卖数量：
      买入数量 = int(per_stock_budget / 收盘价 / 100) × 100
      其中 per_stock_budget = 总资产 / N   （满仓等权）
  5. 输出 trade_plan_YYYYMMDD.csv
  6. 更新 portfolio_state.json（记录新的持仓状态）
```

#### 买卖决策示意图

```
得分排名            当前持仓           决策
──────────────────────────────────────────────
第1名  600519.SH    ✓ 已持有         保留
第2名  000858.SZ    ✓ 已持有         保留
第3名  601318.SH    ✗ 未持有         → 买入
第4名  600036.SH    ✓ 已持有         保留
...
第10名 000001.SZ    ✗ 未持有         → 买入（K=2, 已满）
──────────────────────────────────────────────
第11名 600000.SH    ✓ 已持有         → 卖出（不在前10）
第12名 002415.SZ    ✓ 已持有         → 卖出（不在前10，但K=2已满，暂不卖）
```

#### 数量计算公式

```
per_stock_budget = 1,000,000 / 10 = 100,000 元
买入数量 = int(100,000 / 收盘价 / 100) × 100

例如：收盘价 1680.50 → int(100000 / 1680.50 / 100) × 100
                    → int(0.595) × 100
                    → 0 × 100 = 0  ← 价格太高买不起
需要手动调整，或系统自动保底买 100 股

例如：收盘价 48.00 → int(100000 / 48.00 / 100) × 100
                  → int(20.83) × 100
                  → 20 × 100 = 2000 股
```

### 5.2 输出 CSV 格式

| 列名 | 说明 |
|------|------|
| `操作` | "买入" 或 "卖出" |
| `ts_code` | 原生股票代码 |
| `ths_code` | 同花顺格式代码 |
| `名称` | 股票名称 |
| `价格` | 当日收盘价（买入参考价） |
| `数量` | 建议买卖股数 |
| `金额` | 预估成交金额（数量 × 价格） |
| `得分排名` | 在全市场中的排名 |
| `备注` | 买入："等权配置 ~100000元"；卖出："排名跌出前10" |

### 5.3 使用方法

```bash
# 基础用法
python generate_trade_plan.py --predictions daily_predictions_20260527.csv

# 自定义持仓数量和调仓上限
python generate_trade_plan.py --predictions daily_predictions_20260527.csv --n 10 --k 2

# 指定本金（覆盖持仓文件中的值）
python generate_trade_plan.py --predictions daily_predictions_20260527.csv --capital 1000000
```

### 5.4 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--predictions` | (必填) | 每日预测 CSV 文件路径 |
| `--portfolio` | `portfolio_state.json` | 持仓状态文件路径 |
| `--n` | 10 | 持仓股票数量（选得分前 N 名） |
| `--k` | 2 | 每日最大调仓数量（控制换手率） |
| `--capital` | 使用持仓文件中的 total_value | 覆盖本金 |
| `--output_dir` | `./trade_plans` | 交易计划 CSV 输出目录 |

### 5.5 首次建仓 vs 日常调仓

**首次建仓**（portfolio_state.json 为空仓时）：
- 所有 top10 股票都是"买入"
- 没有"卖出"条目
- 每只股票按等权预算分配

**日常调仓**（已有持仓时）：
- 只有排名掉出前 N 的持仓才会触发"卖出"
- 只有排名进入前 N 但尚未持有的才会触发"买入"
- 如果当前持仓全部在前 N 名内，输出"无需调仓"

---

## 6. portfolio_state.json 持仓状态

### 6.1 文件格式

```json
{
  "init_capital": 1000000,
  "current_capital": 1000000,
  "total_value": 1000000,
  "positions": {},
  "last_run_date": ""
}
```

### 6.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `init_capital` | float | 初始本金，固定不变 |
| `current_capital` | float | 当前现金余额（运行后由 generate_trade_plan.py 预估更新） |
| `total_value` | float | 当前总资产（现金 + 持仓市值估值） |
| `positions` | dict | 持仓明细，key=ts_code，value={quantity, cost_price, buy_date} |
| `last_run_date` | str | 上次运行日期（YYYYMMDD），用于追溯 |

### 6.3 持仓后的示例

```json
{
  "init_capital": 1000000,
  "current_capital": 5230,
  "total_value": 1000000,
  "positions": {
    "600519.SH": {
      "quantity": 500,
      "cost_price": 1680.50,
      "buy_date": "20260527"
    },
    "000858.SZ": {
      "quantity": 2000,
      "cost_price": 48.00,
      "buy_date": "20260527"
    }
  },
  "last_run_date": "20260527"
}
```

### 6.4 注意事项

- 此 JSON 中的 `current_capital` 和 `total_value` 是**系统预估**值（基于当日收盘价），与实际账户资产可能有偏差
- `cost_price` 记录的是系统建议的买入价（当日收盘价），你的实际成交价可能不同
- 如果你在同花顺中手动调整了数量，可以手动编辑这个 JSON 来保持与实际持仓一致
- 要"重置系统"（清仓重来），只需删除或清空 `positions` 对象即可

---

## 7. run_daily.bat 一键运行

### 7.1 功能

双击 `run_daily.bat` 自动执行完整的每日预测 + 交易计划流程，无需记忆命令行参数。

### 7.2 可配置变量

编辑 `run_daily.bat` 开头几行即可：

```batch
set MODEL_DIR=.\saved_models\production_v1
set DATA_PATH=D:\zhw\A股数据
set N_HOLD=10
set K_REBALANCE=2
```

### 7.3 执行流程

```batch
@echo off
chcp 65001 >nul                            ← 设置 UTF-8 编码

REM 【第 1 步】运行每日预测
python daily_trader.py ^
    --model_dir %MODEL_DIR% ^
    --data_path %DATA_PATH%

REM 检查错误 → 失败则暂停退出
if %errorlevel% neq 0 ( pause & exit /b )

REM 找到最新生成的预测文件（按文件时间倒序）
for /f "delims=" %%f in ('dir /b /o-d daily_predictions_*.csv') do (
    set PRED_FILE=%%f
    goto :found
)

REM 【第 2 步】生成交易计划
python generate_trade_plan.py ^
    --predictions %PRED_FILE% ^
    --n %N_HOLD% ^
    --k %K_REBALANCE%

REM 完成
echo 请在 trade_plans/trade_plan_YYYYMMDD.csv 中查看买卖清单
pause
```

### 7.4 输出位置

运行后，你会在 `d:\lstm\` 目录下看到：

- `daily_predictions_YYYYMMDD.csv` — 所有股票得分排名（参考用）
- `trade_plans/trade_plan_YYYYMMDD.csv` — **今天的买卖清单**

---

## 8. 完整每日工作流程

### 8.1 时间线

| 时间 | 操作 | 说明 |
|------|------|------|
| 15:00 | A股收盘 | 当日行情确认 |
| ~16:00 | 数据自动更新 | 日线数据写入 `D:/zhw/A股数据/daily/` |
| **16:05** | **双击 `run_daily.bat`** | **一键生成交易计划** |
| 16:10 | 查看 `trade_plans/trade_plan_YYYYMMDD.csv` | 确认买卖清单 |
| 16:15 | 打开同花顺 APP | 手动按照 CSV 中的数量下单 |
| 次日 9:30 | 开盘 | 持仓开始生效 |

### 8.2 首次使用（明天 5-28）的完整步骤

```bash
# 步骤 0：重新训练模型（只需做一次）
#         训练约 30-60 分钟，建议今晚或明早跑
python train.py ^
    --data_path D:/zhw/A股数据 ^
    --stock_pool all ^
    --start_date 20220101 ^
    --end_date 20260527 ^
    --train_start 20220101 ^
    --train_end 20240331 ^
    --val_start 20240401 ^
    --val_end 20250627 ^
    --save_dir ./saved_models/production_v1 ^
    --hidden_dim 32 --num_layers 2 --dropout 0.4 --fc_dropout 0.5 ^
    --lr 2e-4 --weight_decay 1e-3 --epochs 60 ^
    --ema_decay 0.999 --noise_std 0.005 ^
    --use_fundamental --use_moneyflow ^
    --seq_len 30 --label_scale 100.0

# 步骤 1：明天16:05，双击 run_daily.bat
#         首次运行会生成前10只股票的买入清单

# 步骤 2：在同花顺APP中手动买入
#         打开APP → 交易 → 买入 → 输入代码和数量 → 确认
```

### 8.3 之后每一天

```
16:05 → 双击 run_daily.bat → 查看 trade_plans/trade_plan_YYYYMMDD.csv → 同花顺下单
```

---

## 9. 命令行参数速查

### daily_trader.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | `./saved_models/production_v1` | 模型目录 |
| `--data_path` | `D:/zhw/A股数据` | 数据目录 |
| `--output` | 自动命名 | 输出 CSV 路径 |

### generate_trade_plan.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--predictions` | (必填) | 预测 CSV 路径 |
| `--portfolio` | `portfolio_state.json` | 持仓 JSON 路径 |
| `--n` | 10 | 持仓数量 |
| `--k` | 2 | 每日调仓上限 |
| `--capital` | 使用持仓文件值 | 本金 |
| `--output_dir` | `./trade_plans` | 输出目录 |

---

## 10. 常见问题

**Q: 运行 `daily_trader.py` 报 "scaler.pkl 不存在"？**

A: 说明你用的是旧模型（训练时未保存 scaler）。需要用更新后的 `train.py` 重新训练一次。更新后的 `train.py` 会在保存最佳模型时自动写入 `scaler.pkl` 和 `model_meta.pkl`。

**Q: 买入数量显示为 0？**

A: 当单只股票价格过高时（如每股 > 1000 元），等权分配的 10 万元买不到 1 手（100 股）。系统会自动保底分配 100 股，但这会导致该股票仓位偏离等权配置。建议手动调整或使用 `--n` 参数增加持仓数量以降低单只预算。

**Q: 交易计划显示"无需调仓"？**

A: 说明当前持仓的所有股票仍然在得分前 N 名内，不需要卖出或买入任何股票。这是正常现象，系统只在有需要时才会建议调仓。

**Q: 实际成交价和 CSV 中的价格不一致？**

A: CSV 中列出的是**当日收盘价**，作为参考。你在同花顺中用的是次日开盘价交易（或盘中限价单），实际成交价会有所不同。这属于正常的滑点。

**Q: 如何重置系统（清仓重来）？**

A: 删除 `portfolio_state.json` 中的 `positions` 内容，还原为空对象 `{}`，同时把 `current_capital` 和 `total_value` 还原为 1000000。

**Q: 如何回测验证模型效果？**

A: 使用原有的 `backtest.py`：

```bash
python backtest.py ^
    --model_path ./saved_models/production_v1/best_model.pth ^
    --val_start 20250401 --val_end 20250627 ^
    --n 10 --k 2 ^
    --hidden_dim 32 --num_layers 2 --dropout 0.4 --fc_dropout 0.5 ^
    --use_fundamental --use_moneyflow
```

关注输出中的 IC（信息系数）、Top10% Spread、夏普比率三个指标。
