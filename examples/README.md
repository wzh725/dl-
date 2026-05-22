# `examples/` 持仓状态 JSON

| 文件 | 场景 |
|------|------|
| `state_empty.json` | **未建仓**：`sellable`、`locked` 均为空对象，仅现金。 |
| `state_holding.json` | **已建仓**：填写可卖 `sellable`、锁定 `locked` 与 `cash`。 |

共用字段与 `next_trade_suggestions.py`、`workflow_cli.py predict-next` 的 `--state` / `--state-in` 一致；可选 **`position_status`**：`empty` | `holding`（与股数交叉校验，以 `sellable`/`locked` 为准）。

**`commission_bps`**：仅表示**券商**佣金基点（万分之几），默认示例为万三 (`3`)。回测 / 推演中另行按 README「交易费用」加收卖出印花税与上海 `600xxx` 过户费。
