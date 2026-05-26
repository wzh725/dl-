#!/usr/bin/env python3
"""
predict：下一交易日持仓推演（组合仿真）与操作建议 CLI。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from data_preprocess import resolve_data_root
from backtest import (
    fees_on_sell_turnover,
    floor_to_lot,
    score_weighted_buys_for_cash_budget,
    load_scores,
    score_weights_from_picks_df,
    _unlock_morning,
)



@dataclass
class PortfolioState:
    cash: float
    lot_size: int = 100
    sellable: Dict[str, int] = field(default_factory=dict)
    locked: Dict[str, int] = field(default_factory=dict)
    commission_rate: float = 0.0002


def load_portfolio_json(path: str) -> PortfolioState:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    cash = float(raw.get("cash", 0.0))
    lot = int(raw.get("lot_size", 100))
    sellable = {str(k): int(v) for k, v in raw.get("sellable", {}).items() if int(v) > 0}
    locked = {str(k): int(v) for k, v in raw.get("locked", {}).items() if int(v) > 0}
    if "commission_rate" in raw:
        crate = float(raw.get("commission_rate", 0.0002))
    else:
        # 兼容旧字段 commission_bps
        crate = float(raw.get("commission_bps", 0.0)) / 10000.0
    return PortfolioState(cash=cash, lot_size=lot, sellable=sellable, locked=locked, commission_rate=crate)


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
        "commission_rate": float(st.commission_rate),
        "_comment": "收盘后状态：当日买入在 locked，下一交易日早盘将并入 sellable（脚本推演已模拟解锁）。",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_trade_price_map_for_day(
    data_root: str,
    trade_date: str,
    *,
    price_col: str = "open",
) -> Dict[str, float]:
    fp = Path(data_root) / "daily" / f"{trade_date}.csv"
    if not fp.is_file():
        raise FileNotFoundError(f"缺少行情文件: {fp}")
    df = pd.read_csv(fp, usecols=["ts_code", price_col])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    df = df[np.isfinite(df[price_col]) & (df[price_col] > 0)]
    return dict(zip(df["ts_code"].astype(str), df[price_col].astype(np.float64)))


def resolve_equity_trade_price_date(
    data_root: str,
    logical_trade_date_yyyymmdd: str,
    *,
    strict: bool = False,
    price_col: str = "open",
) -> Tuple[str, str]:
    """
    将「你希望标注的下一交易日」映射到本地实际用于读取成交价的 CSV 交易日。

    - 存在 daily/{{logical}}.csv → 用当日成交价列。
    - 否则在非 strict 下：取 daily/ 中 **<= logical** 的最近一个交易日的 CSV。
      适用于数据只到今天、但想把「语义上的次日」仍为 YYYYMMDD 的情境（占位成交近似）。

    返回 (pricing_date_yyyymmdd, stderr_note_or_empty)。
    strict=True → 必须由 logical 当天的 CSV；否则报错。
    """
    d = str(logical_trade_date_yyyymmdd).strip().replace("-", "")
    if len(d) != 8 or not d.isdigit():
        raise ValueError(f"非法交易日 logical_trade_date: {logical_trade_date_yyyymmdd!r}")

    fp = Path(data_root) / "daily" / f"{d}.csv"
    if fp.is_file():
        return d, ""
    if strict:
        raise FileNotFoundError(
            f"--strict-next-trade-csv：要求行情文件存在但缺失：{fp}"
        )

    daily_dir = Path(data_root) / "daily"
    if not daily_dir.is_dir():
        raise FileNotFoundError(f"daily 目录不存在: {daily_dir}")

    dated = sorted(
        p.stem for p in daily_dir.glob("*.csv") if len(p.stem) == 8 and p.stem.isdigit()
    )
    cand = [x for x in dated if x <= d]
    if not cand:
        raise FileNotFoundError(
            f"未找到不晚于 {d} 的 daily/*.csv（数据根目录：{data_root}）。请补行情或改用已有交易日。"
        )
    best = cand[-1]
    if best == d:
        return d, ""
    note = (
        f"[占位价格] daily/{d}.csv 不存在 → 用「不晚于 {d}」的最近行情日 {best} 的 {price_col} 价模拟撮合。"
        "（数据尚未到车时，用你的历史末尾价近似语义上的次日）"
    )
    return best, note


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
    """候选池 Top-n，持仓 Top-k（分数），与 backtest.run_backtest 一致。"""
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
    commission_rate: float,
) -> Tuple[Dict[str, int], float, float]:
    """按 pred_score 加权分配整手股数；返回 (代码→股数, 买入成交额 gross, 买入侧费用合计)。"""
    return score_weighted_buys_for_cash_budget(
        px_map, budget_cash, picks_df, scores_map, lot_size, commission_rate
    )


def simulate_score_weighted_day(
    panel: pd.DataFrame,
    px_map: Dict[str, float],
    scores_map: Dict[str, float],
    st: PortfolioState,
    n: int,
    k: int,
    commission_rate: float,
) -> Tuple[OrderLog, PortfolioState, float, float, str]:
    """与 run_backtest 一致：早盘解锁 → 卖出最低分 k 只（不在目标持仓）→ 买入最高分缺口（含 A 股费用）。"""
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
        ps = PortfolioState(cash, lot_size, sellable, locked, st.commission_rate)
        return log, ps, 0.0, nav, "无候选标的"

    turnover_sell = 0.0
    sell_fees = 0.0
    held_codes = sorted(
        {c for c, sh in sellable.items() if sh > 0} | {c for c, sh in locked.items() if sh > 0}
    )
    target_codes = [str(c) for c in picks_df["ts_code"].astype(str).tolist()]
    target_set = set(target_codes)
    sell_candidates = [c for c in held_codes if c in sellable and c not in target_set]
    sell_candidates = sorted(sell_candidates, key=lambda c: (scores_map.get(c, -np.inf), c))
    sell_codes = sell_candidates[: max(0, int(k))]
    for code in sell_codes:
        sh0 = sellable.get(code, 0)
        qty = floor_to_lot(sh0, lot_size)
        if qty <= 0:
            continue
        px = float(px_map.get(code, float("nan")))
        if not np.isfinite(px):
            continue
        proceeds = qty * px
        sf = fees_on_sell_turnover(code, proceeds, qty, commission_rate)
        sellable[code] = sh0 - qty
        if sellable[code] <= 0:
            sellable.pop(code, None)
        cash += proceeds - sf
        turnover_sell += proceeds
        sell_fees += sf
        log.sell(code, qty, px, "score_weighted-卖出低分持仓")

    nav_before = cash + _mv(px_map, sellable, locked)
    if nav_before <= 1e-9:
        nav_after = cash + _mv(px_map, sellable, locked)
        ps = PortfolioState(cash, lot_size, sellable, locked, st.commission_rate)
        return log, ps, 0.0, nav_after, "现金不足以建仓"

    held_after_sell = {c for c, sh in sellable.items() if sh > 0} | {c for c, sh in locked.items() if sh > 0}
    slots = max(0, int(k) - len(held_after_sell))
    buy_candidates = [c for c in target_codes if c not in held_after_sell]
    buy_codes = buy_candidates[:slots]
    if buy_codes:
        picks_buy_df = picks_df[picks_df["ts_code"].astype(str).isin(set(buy_codes))].copy()
        tmp_locked, spent, buy_fees = weighted_allocate_shares(
            px_map, cash, picks_buy_df, scores_map, lot_size, commission_rate
        )
    else:
        tmp_locked, spent, buy_fees = {}, 0.0, 0.0
    fee_day = sell_fees + buy_fees
    for code, sh in tmp_locked.items():
        log.buy(code, sh, float(px_map[code]), "score_weighted-买入高分候选")

    cash = cash - spent - buy_fees
    locked.update(tmp_locked)
    nav_after = cash + _mv(px_map, sellable, locked)
    ps = PortfolioState(
        cash=cash,
        lot_size=lot_size,
        sellable=sellable,
        locked=locked,
        commission_rate=st.commission_rate,
    )
    return log, ps, fee_day, nav_after, "score_weighted：卖出低分持仓 k 只，买入高分候选补齐至 Top-k"


def run_simulation(
    panel: pd.DataFrame,
    px_map: Dict[str, float],
    scores_map: Dict[str, float],
    st: PortfolioState,
    n: int,
    k: int,
    commission_rate: float,
) -> Tuple[OrderLog, PortfolioState, float, float, str]:
    """commission_rate 为券商佣金费率小数（万二=0.0002）；另含卖出印花税 0.1% 与仅上证 60* 双向过户费（按成交股数、单笔最低 1 元）。"""
    return simulate_score_weighted_day(panel, px_map, scores_map, st, n, k, commission_rate)



def load_holdings_csv(path: str) -> Dict[str, int]:
    df = pd.read_csv(path)
    if "ts_code" not in df.columns or "sellable_shares" not in df.columns:
        raise ValueError("持仓 CSV 至少需要列: ts_code, sellable_shares")
    out: Dict[str, int] = {}
    for _, r in df.iterrows():
        code = str(r["ts_code"])
        sh = int(pd.to_numeric(r["sellable_shares"], errors="coerce") or 0)
        if sh > 0:
            out[code] = sh
    return out


def infer_next_trade_date_from_daily(data_root: str, scores_last: str) -> Optional[str]:
    daily = Path(data_root) / "daily"
    if not daily.is_dir():
        return None
    names = sorted(p.stem for p in daily.glob("*.csv") if p.stem.isdigit() and len(p.stem) == 8)
    for d in names:
        if d > scores_last:
            return d
    return None


def codes_tradable_next_day(
    data_root: str,
    next_d: str,
    *,
    price_col: str = "open",
) -> Optional[Set[str]]:
    fp = Path(data_root) / "daily" / f"{next_d}.csv"
    if not fp.is_file():
        return None
    df = pd.read_csv(fp, usecols=["ts_code", price_col])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    df = df[np.isfinite(df[price_col]) & (df[price_col] > 0)]
    return set(df["ts_code"].astype(str))


def score_snapshot_date_for_day(
    sorted_score_dates: List[str],
    trade_day: str,
    score_lag: int,
) -> Tuple[str, str]:
    if not sorted_score_dates:
        raise ValueError("打分 CSV trade_date 列表为空")

    scores_last = sorted_score_dates[-1]

    if score_lag <= 0:
        if trade_day and trade_day in sorted_score_dates:
            return trade_day, "lag=0：使用当日截面（trade_date 与交易日对齐）"
        return scores_last, "lag=0：使用打分 CSV 中最后一个 trade_date 截面"

    # 语义上的执行日晚于打分文件中最晚截面：仅用历史已产生的 pred_score（典型：末日锚定 20260520 → 推演 20260521）
    if trade_day and len(trade_day) == 8 and trade_day.isdigit() and trade_day > scores_last:
        idx = max(0, len(sorted_score_dates) - score_lag)
        snap = sorted_score_dates[idx]
        return snap, (
            f"语义执行日 {trade_day} 晚于打分最后截面 {scores_last}；"
            f"lag={score_lag} → 使用 trade_date={snap}（与 predict-next 末日推理对齐）"
        )

    if trade_day and trade_day in sorted_score_dates:
        di = sorted_score_dates.index(trade_day)
        idx = max(0, di - score_lag)
        snap = sorted_score_dates[idx]
        return snap, f"lag={score_lag}：执行日在打分枚举内 → trade_date={snap}"

    di = len(sorted_score_dates)
    idx = max(0, di - score_lag)
    snap = sorted_score_dates[idx]
    label = trade_day if trade_day else "（未指定 / 推断）"
    return snap, (
        f"执行日线索 {label} 与打分 trade_date 未对齐 → lag={score_lag}，使用 trade_date={snap}"
    )


def print_advisory_summary(
    *,
    next_d: str,
    score_snap: str,
    snap_note: str,
    panel: pd.DataFrame,
    n: int,
    k: int,
    top_df: pd.DataFrame,
) -> None:
    print("=== 交易日与分数快照 ===")
    print(f"下一交易日（推断/指定）: {next_d}")
    print(f"score-lag → 使用的打分快照 trade_date = {score_snap}")
    print(f"说明: {snap_note}")
    print(f"快照股票数（去重后）: {len(panel)}")
    print(f"\n=== score_weighted：候选池 Top-{n}，持仓 Top-{k}；权重 ∝ pred_score（池内平移后归一）===")
    print("\n--- 目标持仓（代码 / 分数 / 目标权重）---")
    cols = [c for c in ("ts_code", "pred_score", "target_weight") if c in top_df.columns]
    print(top_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下一交易日操作建议：摘要模式 或 --state 细单模式（买卖股数）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="未建仓: examples/state_empty.json；已建仓: examples/state_holding.json",
    )
    parser.add_argument("--scores", required=True)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DL_DATA_ROOT", ""),
    )
    parser.add_argument("--next-trade-date", default="", help="下一交易日 YYYYMMDD（细单模式必填或可自动推断）")
    parser.add_argument("--n", type=int, default=30, help="打分截面上取分数最高的 n 只组成候选池")
    parser.add_argument("--k", type=int, default=10, help="在候选池内持有分数最高的 k 只")
    parser.add_argument("--score-lag", type=int, default=1)
    parser.add_argument("--lot-size", type=int, default=None, help="不传则用 state 或默认 100")
    parser.add_argument("--holdings", default="", help="摘要模式：简易持仓 CSV；细单模式：可在 state 空仓时合并进来")
    parser.add_argument(
        "--state",
        default="",
        help="portfolio_state.json：现金 + sellable + locked；启用细单模式",
    )
    parser.add_argument(
        "--strict-next-trade-csv",
        action="store_true",
        help="必须为当日生成 daily/{{--next-trade-date}}.csv；禁止在无文件时用更早交易日的成交价占位",
    )
    parser.add_argument(
        "--trade-price-col",
        choices=("open", "close"),
        default="open",
        help="交易撮合价格列（默认 open，满足每日开盘买卖）",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=None,
        help="覆盖 state JSON 内 commission_rate（券商费率，小数）；不传则用 state",
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="兼容旧参数：基点制（万三=3），若提供则覆盖 --commission-rate",
    )
    parser.add_argument("--out-csv", default="", help="摘要模式：写出目标池 CSV")
    parser.add_argument("--out-orders", default="", help="细单模式：写出指令明细 CSV")
    parser.add_argument("--out-next-state", default="", help="细单模式：写出推演收盘后状态 JSON（次日链式）")
    args = parser.parse_args()

    args.data_root = resolve_data_root(args.data_root)

    scores = load_scores(args.scores)
    dates = sorted(scores["trade_date"].unique())
    if not dates:
        raise SystemExit("scores 为空")

    last_s = dates[-1]
    next_d = args.next_trade_date.strip().replace("-", "")
    if not next_d:
        next_d = infer_next_trade_date_from_daily(args.data_root, last_s) or ""

    trade_day_for_snap = next_d if (next_d.isdigit() and len(next_d) == 8) else ""
    score_snap, snap_note = score_snapshot_date_for_day(dates, trade_day_for_snap, args.score_lag)

    panel = scores[scores["trade_date"] == score_snap].drop_duplicates(subset=["ts_code"]).copy()
    panel = panel.sort_values("pred_score", ascending=False).reset_index(drop=True)
    scores_map = panel.set_index("ts_code")["pred_score"].to_dict()

    if args.state:
        if not (next_d.isdigit() and len(next_d) == 8):
            raise SystemExit("细单模式需要有效的下一交易日 YYYYMMDD（请传 --next-trade-date 或确保 daily/ 可推断）")

        px_date, px_note = resolve_equity_trade_price_date(
            args.data_root,
            next_d,
            strict=args.strict_next_trade_csv,
            price_col=str(args.trade_price_col),
        )
        if px_note:
            print(px_note, flush=True)
        px_map = load_trade_price_map_for_day(
            args.data_root,
            px_date,
            price_col=str(args.trade_price_col),
        )
        tradable = set(px_map.keys())
        panel = panel[panel["ts_code"].astype(str).isin(tradable)].reset_index(drop=True)
        scores_map = panel.set_index("ts_code")["pred_score"].to_dict()

        st = load_portfolio_json(args.state)
        if args.lot_size is not None:
            st.lot_size = int(args.lot_size)
        if args.holdings:
            extra = load_holdings_csv(args.holdings)
            for c, sh in extra.items():
                st.sellable[c] = st.sellable.get(c, 0) + sh

        if args.commission_bps is not None:
            crate = float(args.commission_bps) / 10000.0
        elif args.commission_rate is not None:
            crate = float(args.commission_rate)
        else:
            crate = float(st.commission_rate)

        log, ps_end, fee, nav_after, sim_note = run_simulation(
            panel,
            px_map,
            scores_map,
            st,
            args.n,
            args.k,
            crate,
        )

        print("=== 细单模式：下一交易日买卖指令（与回测规则对齐） ===")
        print(
            f"语义下一交易日: {next_d}  |  用于成交价的 CSV 交易日: {px_date}"
            + ("（与语义日相同）" if px_date == next_d else "（价格占位说明见上文）")
        )
        print(f"使用打分快照: {score_snap}  |  {snap_note}")
        print(f"推演说明: {sim_note}")
        print(
            f"交易费用（印花税+过户+券商佣金）：commission_rate={crate}（仅券商项）"
            f"  →  当日估算合计 ≈ {fee:.2f} 元"
        )
        print(
            f"推演净值（按 {args.trade_price_col} 价计价）≈ {nav_after:.2f} 元；"
            f"现金余额 ≈ {ps_end.cash:.2f} 元"
        )
        print("\n--- 指令明细（买入当日计入 locked，次日才可卖）---")
        if not log.rows:
            print("（无成交指令）")
        else:
            od = pd.DataFrame(log.rows)
            print(od.to_string(index=False))

        print("\n--- 收盘后账户状态（可写入 --out-next-state；次日加载时会自动早盘解锁）---")
        print(f"cash: {ps_end.cash:.6g}")
        print(f"sellable: {ps_end.sellable}")
        print(f"locked: {ps_end.locked}")

        if args.out_orders:
            outp = Path(args.out_orders)
            outp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(log.rows).to_csv(outp, index=False)
            print(f"\n已写指令: {outp}")

        if args.out_next_state:
            ps_save = PortfolioState(
                cash=ps_end.cash,
                lot_size=ps_end.lot_size,
                sellable=dict(ps_end.sellable),
                locked=dict(ps_end.locked),
                commission_rate=crate,
            )
            save_portfolio_json(args.out_next_state, ps_save)
            print(f"已写下一状态: {args.out_next_state}")
            print(
                "提示：此为当日收盘后状态（新买入在 locked）。次日再用同一脚本加载时会自动先做 "
                "T+1 解锁（与回测一致），无需手工合并。"
            )

        return

    # ---------- 摘要模式 ----------
    if not next_d:
        print("=== 提示 ===")
        print(f"打分 CSV 中最后截面日 trade_date = {last_s}")
        print("未在 daily/ 中找到更晚交易日；请手动传入 --next-trade-date")
        next_d = "（请指定下一交易日）"

    tradable: Optional[Set[str]] = None
    if next_d.isdigit() and len(next_d) == 8:
        tradable_set: Optional[Set[str]] = None
        try:
            px_date, px_note = resolve_equity_trade_price_date(
                args.data_root,
                next_d,
                strict=args.strict_next_trade_csv,
                price_col=str(args.trade_price_col),
            )
            if px_note:
                print(px_note, flush=True)
            tradable_set = codes_tradable_next_day(
                args.data_root,
                px_date,
                price_col=str(args.trade_price_col),
            )
        except FileNotFoundError as e:
            raise SystemExit(str(e))
        if tradable_set is not None:
            tradable = tradable_set
            panel = panel[panel["ts_code"].astype(str).isin(tradable)].reset_index(drop=True)
            scores_map = panel.set_index("ts_code")["pred_score"].to_dict()

    picks_df = pick_picks_df(panel, args.n, args.k)
    wmap = score_weights_from_picks_df(picks_df)
    top_df = picks_df.copy()
    top_df["target_weight"] = top_df["ts_code"].astype(str).map(lambda c: wmap.get(c, float("nan")))

    print_advisory_summary(
        next_d=str(next_d),
        score_snap=score_snap,
        snap_note=snap_note,
        panel=panel,
        n=args.n,
        k=args.k,
        top_df=top_df,
    )

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        top_df.assign(next_trade_date=str(next_d), score_snapshot_date=score_snap, strategy="score_weighted").to_csv(
            outp, index=False
        )
        print(f"\n已写: {outp}")


if __name__ == "__main__":
    main()
