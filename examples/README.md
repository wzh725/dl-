# `examples/` 使用说明

本目录放的是 `predict-next` 的账户输入示例（`--state-in`）。
你只要按这里的口径把自己的账户写成 JSON，就能直接跑出次日买卖指令。

---

## 1) 文件说明

| 文件 | 场景 |
|---|---|
| `state_empty.json` | 空仓起步：`sellable`、`locked` 为空，仅现金。 |
| `state_holding.json` | 已持仓：同时有可卖仓位和锁仓仓位。 |

---

## 2) 必填字段（按当前策略）

- `cash`：可用现金（数字）
- `lot_size`：整手单位，A 股一般为 `100`
- `sellable`：当日可卖仓位，格式为 `{ "ts_code": 股数 }`
- `locked`：当日不可卖仓位（T+1），格式同上
- `commission_rate`：券商佣金率（小数，如万二写 `0.0002`）

可选字段：

- `position_status`：`empty` / `holding`，仅用于可读性；脚本最终以 `sellable/locked` 为准
- 其他自定义备注字段（如 `_说明`）会被忽略

---

## 3) 怎么从真实账户映射成 JSON

以“你现在要在下一交易日开盘执行”为准，写入规则如下：

1. `cash` 写“当前可用现金”
2. 今天就能卖出的仓位写进 `sellable`
3. 今天不能卖（昨日或当日新买，受 T+1 约束）的写进 `locked`
4. 股数尽量写整手（`lot_size` 的整数倍）
5. 代码统一交易所后缀格式（如 `000001.SZ`、`600000.SH`）

---

## 4) 价格与时序口径（当前实现）

- `score-lag=1`：用 `T-1` 的分数决策 `T` 日交易
- 默认 `--trade-price-col open`：按 `T` 日开盘价撮合
- 若 `daily/T.csv` 不存在且请求 `open`：显式采用“最近可用交易日 `close` 近似 `T` 日 `open`”

这和“`1.1` 出信号，`1.2` 开盘执行，近似 `1.2 open ≈ 1.1 close`”一致。

---

## 5) 推荐命令（持仓/空仓都适用）

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
  --commission-rate 0.0002
```

如果已有分数文件，记得加 `--skip-train` 只做推演。

---

## 6) 最小自检清单

- `cash >= 0`
- `sellable` / `locked` 的股数都是正整数
- 股票代码都带 `.SZ` / `.SH`
- `commission_rate` 用小数而不是 bps
- 预计执行日与 `--next-trade-date` 一致
