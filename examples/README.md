# `examples/` 说明

## 持仓状态 JSON

| 文件 | 场景 |
|------|------|
| `state_empty.json` | **未建仓**：`sellable`、`locked` 均为空对象，仅现金。 |
| `state_holding.json` | **已建仓**：填写可卖 `sellable`、锁定 `locked` 与 `cash`。 |

共用字段与 `predict.py`、`workbench.py predict-next` 的 `--state` / `--state-in` 一致；可选 **`position_status`**：`empty` | `holding`（与股数交叉校验，以 `sellable`/`locked` 为准）。

---

## 全量回测示例脚本

- **`run_full_2016_to_202602_backtest.sh`**：2016 起训练、`workbench backtest` + 八卡 `torchrun`，数据为 `../data/README.md` 中的日线、侧车（`metric` / `moneyflow` / `market`）与 ST 过滤等；不含新闻。

端到端说明见 **`dl-/README.md`**。`workbench.py` 须在 **`--`** 后将训练参数传给 `train.py`。

**`commission_rate`**：仅表示**券商**佣金费率（小数，万二=`0.0002`）。回测 / 推演中另行按仓库 README「交易费用」加收卖出印花税与双向过户费。旧字段 `commission_bps` 仍兼容。
