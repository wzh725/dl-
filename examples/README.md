# `examples/` 使用说明

本目录提供 `predict-next` 所需的示例账户状态 JSON。

---

## 1) 文件说明

| 文件 | 场景 |
|---|---|
| `state_empty.json` | 未建仓：`sellable`、`locked` 为空，仅现金。 |
| `state_holding.json` | 已建仓：同时含 `cash`、`sellable`、`locked`。 |

---

## 2) 字段说明

- `cash`：可用现金
- `lot_size`：最小交易单位（A 股默认 100）
- `sellable`：当日可卖股数
- `locked`：当日不可卖（T+1 锁仓）股数
- `commission_rate`：券商佣金费率（小数，万二=`0.0002`）
- `position_status`：可选，`empty` 或 `holding`（脚本最终以股数字段为准）

---

## 3) 使用方式

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
  --score-lag 1
```

---

## 4) 备注

- 示例文件可直接复制后修改，不影响脚本解析。
- JSON 中额外字段会被忽略，可用于写备注。
- 端到端流程与参数请看上级 `dl-/README.md`。
