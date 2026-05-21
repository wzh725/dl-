"""
下一交易日：根据持仓状态 + 打分截面 + 次日收盘价（来自 daily CSV），
推演与 backtest_score_weighted 一致的整手、T+1（locked）、佣金规则，并产出每笔买卖股数。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_score_weighted import (
    _trim_desired_cost_to_budget,
    _unlock_morning,
    floor_to_lot,
    score_weights_from_picks_df,
)


@dataclass
class PortfolioState:
    cash: float
    lot_size: int = 100
    sellable: Dict[str, int] = field(default_factory=dict)
    locked: Dict[str, int] = field(default_factory=dict)
    commission_bps: float = 0.0


def load_portfolio_json(path: str) -> PortfolioState:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    cash = float(raw.get("cash", 0.0))
    lot = int(raw.get("lot_size", 100))
    sellable = {str(k): int(v) for k, v in raw.get("sellable", {}).items() if int(v) > 0}
    locked = {str(k): int(v) for k, v in raw.get("locked", {}).items() if int(v) > 0}
    cbps = float(raw.get("commission_bps", 0.0))
    return PortfolioState(cash=cash, lot_size=lot, sellable=sellable, locked=locked, commission_bps=cbps)


def infer_position_mode_from_state_dict(raw: Dict[str, Any]) -> str:
    """
    依据 sellable/locked 推断 empty | holding；若存在 position_status / position_mode
    与推断不一致则在 stderr 打警告（以股数字段为准）。
    """
    explicit = raw.get("position_status") or raw.get("position_mode")
    sellable = raw.get("sellable") or {}
    locked = raw.get("locked") or {}
    has = sum(int(v) for v in sellable.values() if int(v) > 0) + sum(
        int(v) for v in locked.values() if int(v) > 0
    )
    inferred = "empty" if has == 0 else "holding"
    if explicit is None:
        return inferred
    ex = str(explicit).strip().lower()
    if ex in ("empty", "flat", "cash", "未建仓"):
        ex_norm = "empty"
    elif ex in ("holding", "持仓", "已建仓"):
        ex_norm = "holding"
    else:
        return inferred
    if ex_norm != inferred:
        print(
            f"[警告] JSON 中 position_status={explicit!r} 与 sellable/locked 推断的「{inferred}」不一致，以实际持仓字段为准。",
            file=sys.stderr,
        )
    return inferred


def save_portfolio_json(path: str, st: PortfolioState) -> None:
    payload = {
        "cash": float(st.cash),
        "lot_size": int(st.lot_size),
        "sellable": {k: int(v) for k, v in st.sellable.items() if v > 0},
        "locked": {k: int(v) for k, v in st.locked.items() if v > 0},
        "commission_bps": float(st.commission_bps),
        "_comment": "收盘后状态：当日买入在 locked，下一交易日早盘将并入 sellable（脚本推演已模拟解锁）。",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_close_map_for_day(data_root: str, trade_date: str) -> Dict[str, float]:
    fp = Path(data_root) / "daily" / f"{trade_date}.csv"
    if not fp.is_file():
        raise FileNotFoundError(f"缺少行情文件: {fp}")
    df = pd.read_csv(fp, usecols=["ts_code", "close"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[np.isfinite(df["close"]) & (df["close"] > 0)]
    return dict(zip(df["ts_code"].astype(str), df["close"].astype(np.float64)))


class OrderLog:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def sell(self, ts_code: str, shares: int, price: float, phase: str = "") -> None:
        if shares <= 0:
            return
        self.rows.append(
            {
                "side": "卖出",
                "ts_code": ts_code,
                "shares": int(shares),
                "price": float(price),
                "amount": float(shares * price),
                "phase": phase,
            }
        )

    def buy(self, ts_code: str, shares: int, price: float, phase: str = "") -> None:
        if shares <= 0:
            return
        self.rows.append(
            {
                "side": "买入",
                "ts_code": ts_code,
                "shares": int(shares),
                "price": float(price),
                "amount": float(shares * price),
                "phase": phase,
            }
        )


def _mv(px_map: Dict[str, float], sellable: Dict[str, int], locked: Dict[str, int]) -> float:
    mv = 0.0
    for c, sh in sellable.items():
        if sh <= 0:
            continue
        px = px_map.get(c, float("nan"))
        if np.isfinite(px):
            mv += sh * px
    for c, sh in locked.items():
        if sh <= 0:
            continue
        px = px_map.get(c, float("nan"))
        if np.isfinite(px):
            mv += sh * px
    return mv


def pick_picks_df(day_idx: pd.DataFrame, n: int, k: int) -> pd.DataFrame:
    """候选池 Top-n，持仓 Top-k（分数），与 backtest_score_weighted.run_backtest 一致。"""
    sorted_df = day_idx.sort_values("pred_score", ascending=False).reset_index(drop=True)
    pool = sorted_df.head(min(n, len(sorted_df)))
    k_eff = min(k, len(pool))
    return pool.head(k_eff).copy()


def weighted_allocate_shares(
    px_map: Dict[str, float],
    budget_cash: float,
    picks_df: pd.DataFrame,
    scores_map: Dict[str, float],
    lot_size: int,
) -> Tuple[Dict[str, int], float]:
    """按 pred_score 加权分配整手股数；返回 (代码→股数, 买入成交额)。"""
    if picks_df.empty or budget_cash <= 1e-9:
        return {}, 0.0
    weights = score_weights_from_picks_df(picks_df)
    tradable: List[str] = []
    tpx: Dict[str, float] = {}
    for _, row in picks_df.iterrows():
        code = str(row["ts_code"])
        px = float(px_map.get(code, float("nan")))
        if np.isfinite(px) and px > 0:
            tradable.append(code)
            tpx[code] = px
    if not tradable:
        return {}, 0.0
    sw = sum(weights.get(c, 0.0) for c in tradable)
    if sw <= 1e-18:
        return {}, 0.0
    wt = {c: weights[c] / sw for c in tradable}
    nav_mid = budget_cash
    desired: Dict[str, int] = {}
    for code in tradable:
        tgt = floor_to_lot(int(nav_mid * wt[code] / tpx[code]), lot_size)
        if tgt > 0:
            desired[code] = tgt
    desired = _trim_desired_cost_to_budget(desired, tpx, lot_size, nav_mid, scores_map)
    tmp_locked: Dict[str, int] = {}
    spent = 0.0
    remaining = budget_cash
    for code in tradable:
        tgt = desired.get(code, 0)
        if tgt <= 0:
            continue
        px = tpx[code]
        cost = tgt * px
        if cost > remaining + 1e-6:
            max_lots = int(remaining // (px * lot_size))
            tgt = max_lots * lot_size
            if tgt <= 0:
                continue
            cost = tgt * px
        remaining -= cost
        spent += cost
        tmp_locked[code] = tmp_locked.get(code, 0) + tgt
    return tmp_locked, spent


def simulate_score_weighted_day(
    panel: pd.DataFrame,
    px_map: Dict[str, float],
    scores_map: Dict[str, float],
    st: PortfolioState,
    n: int,
    k: int,
    commission_rate: float,
) -> Tuple[OrderLog, PortfolioState, float, float, str]:
    """与 run_backtest 一致：早盘解锁 → 卖光可卖 → 清仓记账 → 按分数加权满仓买入 locked。"""
    log = OrderLog()
    sellable = dict(st.sellable)
    locked = dict(st.locked)
    cash = float(st.cash)
    lot_size = st.lot_size

    _unlock_morning(sellable, locked)

    day_idx = panel.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
    picks_df = pick_picks_df(day_idx, n, k)
    if picks_df.empty:
        nav = cash + _mv(px_map, sellable, locked)
        ps = PortfolioState(cash, lot_size, sellable, locked, st.commission_bps)
        return log, ps, 0.0, nav, "无候选标的"

    turnover_sell = 0.0
    no_position = sum(sellable.values()) == 0 and sum(locked.values()) == 0
    if not no_position:
        for code in list(sellable.keys()):
            sh0 = sellable.get(code, 0)
            qty = floor_to_lot(sh0, lot_size)
            if qty <= 0:
                continue
            px = float(px_map.get(code, float("nan")))
            if not np.isfinite(px):
                continue
            proceeds = qty * px
            sellable[code] = sh0 - qty
            if sellable[code] <= 0:
                sellable.pop(code, None)
            cash += proceeds
            turnover_sell += proceeds
            log.sell(code, qty, px, "score_weighted-清仓可卖")

    sellable.clear()
    locked.clear()

    nav_before = cash + _mv(px_map, sellable, locked)
    if nav_before <= 1e-9:
        nav_after = cash + _mv(px_map, sellable, locked)
        ps = PortfolioState(cash, lot_size, sellable, locked, st.commission_bps)
        return log, ps, 0.0, nav_after, "现金不足以建仓"

    tmp_locked, spent = weighted_allocate_shares(px_map, cash, picks_df, scores_map, lot_size)
    for code, sh in tmp_locked.items():
        log.buy(code, sh, float(px_map[code]), "score_weighted-建仓")

    fee_day = commission_rate * (turnover_sell + spent)
    cash = cash - spent - fee_day
    locked.update(tmp_locked)
    nav_after = cash + _mv(px_map, sellable, locked)
    ps = PortfolioState(cash=cash, lot_size=lot_size, sellable=sellable, locked=locked, commission_bps=st.commission_bps)
    return log, ps, fee_day, nav_after, "score_weighted：Top-n 池内 Top-k，pred_score 加权满仓"


def run_simulation(
    panel: pd.DataFrame,
    px_map: Dict[str, float],
    scores_map: Dict[str, float],
    st: PortfolioState,
    n: int,
    k: int,
    commission_rate: float,
) -> Tuple[OrderLog, PortfolioState, float, float, str]:
    """commission_rate 为小数（如 3bps = 0.0003）。唯一策略：预测分数加权全日换手。"""
    return simulate_score_weighted_day(panel, px_map, scores_map, st, n, k, commission_rate)
