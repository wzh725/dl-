# 基于关联残差 LSTM 的 A 股多目标预测与回测系统

## 目录

1. [项目概览](#1-项目概览)
2. [文件结构](#2-文件结构)
3. [数据处理全链路](#3-数据处理全链路)
4. [模型架构详解](#4-模型架构详解)
5. [损失函数设计](#5-损失函数设计)
6. [训练流程](#6-训练流程)
7. [策略与回测](#7-策略与回测)
8. [全部训练参数说明](#8-全部训练参数说明)
9. [推荐参数方案](#9-推荐参数方案)
10. [完整运行命令示例](#10-完整运行命令示例)

---

## 1. 项目概览

本项目使用多值关联残差 LSTM 网络，预测 A 股次日 K 线形态的三个关键维度：

- **label_op**：次日开盘价相对当日收盘价的变化率
- **label_lp**：次日最低价相对当日收盘价的变化率
- **label_hp**：次日最高价相对当日收盘价的变化率

三个标签联合刻画次日 K 线的完整形态——开盘跳空方向、振幅空间、多空博弈结果。模型输出三个预测值后，以 `score = pred_hp - pred_op` 作为选股得分（预期做多空间），在验证集上逐日调仓回测。

---

## 2. 文件结构

```
project/
├── data_processor.py   # 数据加载、特征工程、标签构建、标准化、序列化
├── model.py            # 模型定义（AssociatedResidualNet）+ 损失函数
├── train.py            # 训练入口（EMA、噪声增强、早停、调度器）
├── strategy.py         # TradingStrategy（推理）+ BackTester（回测）
├── backtest.py         # 回测入口（命令行参数）
├── dataset.py          # Dataset 封装（预留）
├── visual.py           # 可视化（预留）我已经不用了，之前用来看预测的三个量与真实的对比图
├── saved_models/       # 模型保存目录
└── README.md
```

---

## 3. 数据处理全链路

数据从磁盘到模型输入经过五个步骤：

### 3.1 第一步：加载日线行情 + 计算 33 维技术特征

程序扫描 `daily/` 目录下所有日期的 CSV，合并为一张大表，按股票代码分组计算技术指标。最终得到 **33 维** 基础特征：

| 类别 | 特征 | 说明 |
|------|------|------|
| **价格** | open, high, low, close, vwap | 日线 OHLCV |
| **均线** | ma5, ma10, ma20, ma60 | 收盘价 5/10/20/60 日均线 |
| **趋势** | ema12, ema26, dif, dea, macd | EMA 快慢线 + MACD 柱 |
| **动量** | momentum_5/10/20/60 | 不同窗口涨跌幅 |
| **成交量** | vol, vol_ma5/10/20, vol_ratio_5/10/20 | 成交量及其均值和比率 |
| **波动率** | volatility_5, volatility_20 | 涨跌幅滚动标准差 |
| **量价复合** | price_volume_corr, vol_price_ratio_5/10 | 量价相关性与配合度 |
| **相对强弱** | rsi, excess_return | RSI 指标 + 截面超额收益 |

### 3.2 第二步：可选合并基本面与资金流

两类额外数据通过 `--use_fundamental` 和 `--use_moneyflow` 开关控制，按 `(trade_date, ts_code)` 做左连接合并：

**基本面**（`metric/` 目录）：
| 特征 | 含义 | 性质 |
|------|------|------|
| pe_ttm | 滚动市盈率 | 定价因子——贵还是便宜 |
| pb | 市净率 | 定价因子——资产溢价程度 |
| roe | 净资产收益率 | 质量因子——盈利能力 |

缺失值用该列全量中位数填充。

**资金流**（`moneyflow/` 目录）：
| 原始列 | 含义 |
|------|------|
| super_large_net | 超大单净流入（≥100 万元/笔） |
| large_net | 大单净流入（20~100 万元/笔） |
| medium_net | 中单净流入（4~20 万元/笔） |
| small_net | 小单净流入（<4 万元/笔） |
| total_turnover | 当日总成交额 |

在此基础上派生两个比例特征（消除不同市值股票的成交规模差异）：

- `mf_main_force_ratio = (super_large_net + large_net) / total_turnover` —— 正值 = 主力净买入，负值 = 主力净卖出
- `mf_retail_ratio = (medium_net + small_net) / total_turnover` —— 正值 = 散户净买入，负值 = 散户净卖出

缺失值用 0 填充。

最终特征维度：
- 不加额外数据：**33 维**
- +基本面：**约 36 维**
- +基本面+资金流：**约 43 维**

### 3.3 第三步：截面 Z-Score 标准化

每天（同一个 `trade_date`）内，对所有股票的特征列计算均值和标准差后做标准化。公式：

```
标准化值 = (原值 - 当天截面均值) / (当天截面标准差)
```

这一步的核心目的是**消除量纲差异并保留截面排序信息**——PE（几十）和 momentum（0.01）在 Z-Score 后数值可比，同时某只股票在同一天比其他股票"贵多少"、"涨得多不多"的排序关系被完整保留。

### 3.4 第四步：滑动窗口序列化

对每只股票用固定长度的滑动窗口切出样本：

```
时间步 1~20 的特征矩阵 → 第 20 天的三个标签
时间步 2~21 的特征矩阵 → 第 21 天的三个标签
...
```

每个样本形状为 `(seq_len, feature_dim)`。对大样本量（百万级），使用 **np.memmap** 直接写磁盘映射，避免内存爆炸。

### 3.5 第五步：时间维 StandardScaler 二次标准化

在训练集上拟合 StandardScaler（按特征维度的全量均值和标准差），批量 transform 训练集和验证集，消除不同时间段之间的分布漂移。这一步采用分批处理，避免一次性展开全量数据导致内存溢出。

---

## 4. 模型架构详解

### 4.1 ResidualLSTMBlock（残差 LSTM 模块）

```
输入 x ──→ LSTM ──→ Dropout ──→ [+] ──→ LayerNorm ──→ 输出
            │                      ↑
            └──────────────────────┘
            (残差连接，维度不匹配时过 Linear 投影)
```

每个 Block 由一层 LSTM + Dropout + 残差连接 + LayerNorm 组成。残差连接让梯度直接穿透，缓解深层 LSTM 的梯度消失问题。

### 4.2 AssociatedResidualNet（多值关联残差网络）

核心思想：**三个分支不是独立预测，而是级联关联**——前一个分支的预测结果作为后一个分支的额外输入。

```
                        原始特征 x (batch, seq_len, F)
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ OP 分支  │        │ LP 分支  │        │ HP 分支  │
    │          │        │          │        │          │
    │ x → LSTM │        │ [x,op] → │        │ [x,op,lp]│
    │  → BN    │        │   LSTM   │        │  → LSTM  │
    │  → Drop  │        │  → BN    │        │  → BN    │
    │  → FC    │        │  → Drop  │        │  → Drop  │
    │  → pred  │        │  → FC    │        │  → FC    │
    │    _op   │───────→│  → pred  │───────→│  → pred  │
    └──────────┘        │    _lp   │        │    _hp   │
                        └──────────┘        └──────────┘
```

特点：
- **OP 分支**：纯从原始特征预测开盘价跳空方向
- **LP 分支**：在知道开盘价后，预测最低价——"开盘后能跌多深"
- **HP 分支**：在知道开盘价和最低价后，预测最高价——"全天的反弹高度"

每个分支包含 `num_layers` 个 ResidualLSTMBlock，取最后时间步的隐状态，经过 BatchNorm → Dropout → Linear 输出标量预测值。Batc  BN  + Dropout 双重正则化防止过拟合。

### 4.3 参数计算

| 配置 | hidden_dim | num_layers | 参数量（33 维输入） |
|------|:---:|:---:|:---:|
| 轻量（1年HS300） | 32 | 2 | ~55K |
| 标准（HS300多年） | 96 | 3 | ~350K |
| 大容量（全市场） | 128 | 3 | ~500K |

---

## 5. 损失函数设计

### 5.1 JointDirectionalLoss（联合方向回归损失）

三个分支的 MSE 取平均，同时对方向错误的样本施加额外惩罚：

```
Loss_op = MSE(pred_op, true_op × label_scale)
Loss_lp = MSE(pred_lp, true_lp × label_scale)
Loss_hp = MSE(pred_hp, true_hp × label_scale)

惩罚系数 penalty：
- OP 和 HP 方向都正确 → penalty = 1.0
- 任意一个方向错误 → penalty = penalty_weight（默认 1.5）

Total = mean(Loss_op × penalty + Loss_lp × penalty + Loss_hp × penalty) / 3
```

`label_scale`（默认 100）将标签从变化率（~0.01 级别）放大到 ~1.0 级别，使训练更稳定。推理时再除以相同倍数还原。

### 5.2 ListNetLoss（截面排序损失）

将预测得分和真实得分分别做 Softmax 归一化后计算交叉熵，优化选股排序质量：

```
score_pred = pred_hp - pred_op
score_true = (true_hp - true_op) × label_scale

P_pred = softmax(score_pred)
P_true = softmax(score_true)

Loss = -Σ P_true × log(P_pred)
```

### 5.3 CompositeLoss（复合损失）

```
Total = JointDirectionalLoss + ranking_lambda × ListNetLoss
```

注意：由于训练时 `shuffle=True` 随机打乱，每个 batch 内的股票来自不同日期，无法代表真实的截面排序。因此 `ranking_lambda` 默认设较小值（0.1~0.3），作为辅助正则项使用。在全市场大数据场景下，batch 内同一日期的概率增大，可以适当提高。

---

## 6. 训练流程

### 6.1 一条 Epoch 的执行顺序

```
1. 前向传播（可选加高斯噪声增强）
2. 计算 CompositeLoss
3. 反向传播
4. 梯度裁剪（clip_grad_norm_）
5. 优化器更新
6. EMA 更新模型权重
7. 用 EMA 权重做验证
8. ReduceLROnPlateau 根据验证损失调整学习率
9. 保存最佳模型（使用 EMA 权重）
10. 早停检查
```

### 6.2 关键技术

**EMA（指数移动平均）**：维护一份平滑的模型权重副本，验证时使用平滑权重。训练过程中参数在噪声中震荡，EMA 能显著提升验证集表现和泛化能力。

```
shadow = 0.999 × shadow + 0.001 × model
```

**高斯噪声增强**：每个 batch 的输入添加小幅度高斯噪声，防止模型记忆训练数据的具体模式。

```
x = x + N(0, noise_std²)
```

**ReduceLROnPlateau**：当验证损失连续 `scheduler_patience` 个 epoch 不下降时，学习率乘以 `scheduler_factor`（如 0.5）。

**早停（Early Stopping）**：验证损失连续 `early_stopping_patience` 个 epoch 不创出新低则停止训练。

---

## 7. 策略与回测

### 7.1 选股得分

模型预测三个值后，对验证集每个样本计算选股得分：

```
score = pred_hp - pred_op
```

得分越高，预期次日开盘到最高点的做多空间越大。得分在所有股票中排行，选前 N 只。

### 7.2 回测规则

| 参数 | 含义 | 默认值 |
|------|------|:---:|
| n | 持仓数量 | 20 |
| k | 每日调仓数量 | 3 |
| transaction_cost | 单边交易成本 | 0.1% |

- 第一天：选得分前 n 只，等权建仓
- 之后每天：当前持仓中保留得分最高的 `n - k` 只，再从剩余股票中选 k 只得分最高的新股票买入
- 收益计算：假设以当日收盘价买入，次日开盘价卖出，收益 = 次日开盘价变化率
- 交易成本在买卖时扣除

### 7.3 回测指标

| 指标 | 含义 | 计算公式 |
|------|------|------|
| 累计收益 | 总收益率 | (期末价值 - 初始资金) / 初始资金 |
| 夏普比率 | 风险调整后收益 | 日均收益 / 日收益标准差 × √252 |
| 最大回撤 | 最大峰值到谷底的跌幅 | max((峰值 - 当前值) / 峰值) |
| IC 值 | 预测得分与实际收益的相关性 | Pearson 相关系数 |
| Top10% / Bottom10% Spread | 高分组的超额收益 | 得分前 10% 均值 - 后 10% 均值 |
| 胜率 | 方向预测正确的比例 | 正确次数 / 总次数 |
| 偏度/峰度 | 收益分布形态 | 负偏=大亏频繁，高峰=黑天鹅风险 |

---

## 8. 全部训练参数说明

### 8.1 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `--data_path` | str | `D:/zhw/A股数据` | 数据根目录 |
| `--stock_pool` | str | `hs300` | `hs300`（沪深300）或 `all`（全市场） |
| `--start_date` | str | `2024-06-01` | 数据加载起始日期 |
| `--end_date` | str | `2025-06-30` | 数据加载结束日期 |
| `--train_start` | str | `2024-06-01` | 训练集起始 |
| `--train_end` | str | `2025-03-31` | 训练集结束 |
| `--val_start` | str | `2025-04-01` | 验证集起始 |
| `--val_end` | str | `2025-06-30` | 验证集结束 |
| `--batch_size` | int | 128 | 批次大小 |
| `--seq_len` | int | 30 | 时间窗口长度 |
| `--horizon` | int | 1 | 预测步长（始终为 1） |

### 8.2 特征开关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `--use_fundamental` | flag | False | 开启后合并 PE/PB/ROE |
| `--use_moneyflow` | flag | False | 开启后合并大中小单资金流 |

### 8.3 模型架构参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `--input_dim` | int | 34 | 特征维度（自动从数据更新，一般无需手动设） |
| `--hidden_dim` | int | 32 | LSTM 隐藏层宽度 |
| `--num_layers` | int | 2 | 每分支 ResidualLSTMBlock 层数 |
| `--dropout` | float | 0.4 | LSTM 内部 Dropout |
| `--fc_dropout` | float | 0.5 | 输出层前 Dropout |
| `--bidirectional` | flag | False | 是否双向 LSTM |

### 8.4 优化器与训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `--lr` | float | 2e-4 | 初始学习率 |
| `--epochs` | int | 60 | 最大训练轮数 |
| `--adamw` | flag | False | 使用 AdamW（推荐开启） |
| `--weight_decay` | float | 1e-3 | L2 正则化系数 |
| `--grad_clip` | float | 0.5 | 梯度裁剪最大范数 |

### 8.5 损失函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `--penalty_weight` | float | 1.5 | 方向错误惩罚系数 |
| `--ranking_lambda` | float | 0.1 | 排序损失权重 |
| `--label_scale` | float | 100.0 | 标签缩放因子 |

### 8.6 正则化与调度参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `--ema_decay` | float | 0.999 | EMA 衰减率，0 表示禁用 |
| `--noise_std` | float | 0.005 | 训练噪声标准差，0 表示不添加 |
| `--early_stopping_patience` | int | 12 | 验证不降时容忍的 epoch 数 |
| `--scheduler_patience` | int | 4 | LR 不降时容忍的 epoch 数 |
| `--scheduler_factor` | float | 0.5 | LR 衰减倍率 |

### 8.7 保存参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `--save_dir` | str | `./saved_models` | 模型保存目录 |
| `--device` | str | `cuda`/`cpu` | 自动检测 |

---

## 9. 推荐参数方案（这个可以不看）

### 方案 A：小数据集（1 年 HS300，~50K 样本）

| 参数 | 值 | 原因 |
|------|:---:|------|
| `hidden_dim` | 32 | 样本少，小模型防过拟合 |
| `num_layers` | 2 | 同上 |
| `dropout` | 0.4 | 强 Dropout |
| `fc_dropout` | 0.5 | 弱 FC 层 |
| `lr` | 2e-4 | 保守学习率 |
| `weight_decay` | 1e-3 | 强 L2 正则 |
| `ranking_lambda` | 0.1 | 排序损失降权 |
| `noise_std` | 0.005 | 数据增强 |
| `early_stopping_patience` | 12 | |
| `batch_size` | 128 | |

### 方案 B：沪深300 多年（8 年，~400K 样本）

| 参数 | 值 | 原因 |
|------|:---:|------|
| `hidden_dim` | 96 | 更多模式 |
| `num_layers` | 3 | 更深抽象 |
| `dropout` | 0.3 | 适度 Dropout |
| `fc_dropout` | 0.3 | 适度 |
| `lr` | 3e-4 | 可稍大 |
| `weight_decay` | 1e-4 | 数据够多，降低 L2 |
| `ranking_lambda` | 0.3 | 截面排序更有意义 |
| `noise_std` | 0.003 | 降低噪声 |
| `early_stopping_patience` | 8 | |
| `batch_size` | 256 | |

### 方案 C：全市场多年（8 年，~600 万+ 样本）

| 参数 | 值 | 原因 |
|------|:---:|------|
| `hidden_dim` | 128 | 大容量 |
| `num_layers` | 3 | |
| `dropout` | 0.3 | |
| `fc_dropout` | 0.3 | |
| `lr` | 5e-4 | 大数据更稳定 |
| `weight_decay` | 1e-4 | |
| `ranking_lambda` | 0.3~0.5 | 全市场截面更有效 |
| `noise_std` | 0.003 | |
| `early_stopping_patience` | 6 | 大数据收敛更快 |
| `scheduler_patience` | 3 | |
| `batch_size` | 512~1024 | 必须大 batch |
| `seq_len` | 20 | 减少内存和训练时间 |

---

## 10. 完整运行命令示例

### 10.1 训练

```powershell
# 全市场 8 年 + 基本面 + 资金流（完整版）
python train.py `
  --data_path "D:/zhw/A股数据" `
  --stock_pool all `
  --start_date "2017-01-01" `
  --end_date "2025-06-30" `
  --train_start "2017-01-01" `
  --train_end "2024-06-30" `
  --val_start "2024-07-01" `
  --val_end "2025-06-30" `
  --batch_size 1024 `
  --seq_len 20 `
  --hidden_dim 128 `
  --num_layers 3 `
  --dropout 0.3 `
  --fc_dropout 0.3 `
  --lr 5e-4 `
  --epochs 30 `
  --weight_decay 1e-4 `
  --adamw `
  --ema_decay 0.999 `
  --noise_std 0.003 `
  --ranking_lambda 0.3 `
  --early_stopping_patience 6 `
  --scheduler_patience 3 `
  --scheduler_factor 0.5 `
  --label_scale 100.0 `
  --use_fundamental `
  --use_moneyflow `
  --save_dir "./saved_models/all8y_full"
```

### 10.2 回测

```powershell
# 对应上面训练的模型
python backtest.py `
  --model_path "./saved_models/all8y_full/best_model.pth" `
  --data_path "D:/zhw/A股数据" `
  --stock_pool all `
  --start_date "20240601" --end_date "20250630" `
  --train_start "20170101" --train_end "20240630" `
  --val_start "20240701" --val_end "20250630" `
  --hidden_dim 128 --num_layers 3 --dropout 0.3 --fc_dropout 0.3 `
  --label_scale 100.0 `
  --use_fundamental --use_moneyflow `
  --seq_len 20 `
  --n 20 --k 3 --transaction_cost 0.001
```

### 10.3 对比实验示例

```powershell
# 实验1：纯量价（不加基本面/资金流）
python train.py ... --save_dir "./saved_models/all8y_base"
python backtest.py --model_path "./saved_models/all8y_base/best_model.pth" ...

# 实验2：+基本面
python train.py ... --use_fundamental --save_dir "./saved_models/all8y_fund"
python backtest.py --model_path "./saved_models/all8y_fund/best_model.pth" --use_fundamental ...

# 实验3：+基本面+资金流
python train.py ... --use_fundamental --use_moneyflow --save_dir "./saved_models/all8y_full"
python backtest.py --model_path "./saved_models/all8y_full/best_model.pth" --use_fundamental --use_moneyflow ...
```
