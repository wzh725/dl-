#!/usr/bin/env python3
"""
根据验证集导出的 pred_score 回测净值。

机制概要：
  - **T+1**：当日买入记入 locked，下一交易日开盘前并入 sellable。
  - **整手**：买卖数量向下取整到手。
  - **打分滞后**：默认 `--score-lag 1`，第 D 日使用上一交易日的截面分数。
  - **唯一策略：预测分数加权**：在当日截面上取分数最高的 **--n** 只作为候选池，再在其中持有分数最高的 **--k** 只；
    组合权重与 **pred_score** 成正比（候选池内减去最小值后归一化，避免负权重）；**每日**清仓可卖仓位后按权重满仓重建。

成交价：当日收盘价；`--commission-bps` 按成交额合计扣除；不含滑点、涨跌停。
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _ensure_parent_dir(path: str) -> None:
    p = os.path.dirname(os.path.abspath(path))
    if p:
        os.makedirs(p, exist_ok=True)


def load_close_lookup(data_root: str, dates: List[str]) -> Dict[Tuple[str, str], float]:
    lookup: Dict[Tuple[str, str], float] = {}
    daily_dir = os.path.join(data_root, "daily")
    for d in dates:
        fp = os.path.join(daily_dir, f"{d}.csv")
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp, usecols=["ts_code", "close"])
        for _, row in df.iterrows():
            code = str(row["ts_code"])
            close = float(row["close"])
            if np.isfinite(close) and close > 0:
                lookup[(d, code)] = close
    return lookup


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"trade_date", "ts_code", "pred_score"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"scores CSV 缺少列: {miss}")
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    df["ts_code"] = df["ts_code"].astype(str)
    df["pred_score"] = pd.to_numeric(df["pred_score"], errors="coerce")
    df = df.dropna(subset=["pred_score"])
    return df


def load_benchmark_daily_returns(data_root: str, bench_file: str = "000300.SH.csv") -> pd.Series:
    """指数日收益率（小数），索引 trade_date 字符串 YYYYMMDD。"""
    path = os.path.join(data_root, "market", bench_file)
    if not os.path.isfile(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, usecols=["trade_date", "pct_chg"])
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    r = df.set_index("trade_date")["pct_chg"].astype(np.float64) / 100.0
    return r.sort_index()


def attach_benchmark(curve: pd.DataFrame, bench_ret: pd.Series, initial_cash: float) -> pd.DataFrame:
    out = curve.copy()
    dates = out["trade_date"].astype(str)
    br = bench_ret.reindex(dates).fillna(0.0).astype(np.float64)
    out["benchmark_daily_ret"] = br.values
    out["benchmark_nav"] = float(initial_cash) * (1.0 + br).cumprod().values
    out["excess_daily_ret"] = out["daily_ret"].astype(np.float64) - br.values
    out["nav_vs_benchmark_ratio"] = out["nav"].astype(np.float64) / out["benchmark_nav"].replace(0, np.nan)
    return out


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0


def sharpe_ratio(daily_ret: np.ndarray, trading_days: int = 252) -> float:
    x = daily_ret[np.isfinite(daily_ret)]
    if len(x) < 2:
        return float("nan")
    s = float(np.std(x, ddof=1))
    if s < 1e-12:
        return float("nan")
    return float(np.mean(x) / s * np.sqrt(trading_days))


def floor_to_lot(shares: int, lot_size: int) -> int:
    if lot_size <= 0:
        raise ValueError("lot_size 必须为正整数")
    if shares <= 0:
        return 0
    return (shares // lot_size) * lot_size


def _unlock_morning(sellable: Dict[str, int], locked: Dict[str, int]) -> None:
    """T+1：上一交易日收盘买入的 locked，在本交易日开始时可卖。"""
    codes = set(sellable) | set(locked)
    for c in codes:
        tot = sellable.get(c, 0) + locked.get(c, 0)
        if tot > 0:
            sellable[c] = tot
        else:
            sellable.pop(c, None)
        locked.pop(c, None)


def _total_shares(sellable: Dict[str, int], locked: Dict[str, int], code: str) -> int:
    return sellable.get(code, 0) + locked.get(code, 0)


def _mv_shares(
    close_fn,
    d: str,
    sellable: Dict[str, int],
    locked: Dict[str, int],
) -> float:
    mv = 0.0
    for c, sh in sellable.items():
        if sh <= 0:
            continue
        px = close_fn(d, c)
        if np.isfinite(px):
            mv += sh * px
    for c, sh in locked.items():
        if sh <= 0:
            continue
        px = close_fn(d, c)
        if np.isfinite(px):
            mv += sh * px
    return mv


def _trim_desired_cost_to_budget(
    desired: Dict[str, int],
    px_map: Dict[str, float],
    lot_size: int,
    nav_budget: float,
    scores_map: Dict[str, float],
) -> Dict[str, int]:
    """若 desired 对应总市值超过 nav_budget，优先从 pred_score 最低的标的减手。"""
    desired = dict(desired)
    if lot_size <= 0:
        return desired

    def total_cost() -> float:
        return sum(desired[c] * px_map[c] for c in desired if desired[c] > 0 and np.isfinite(px_map.get(c, np.nan)))

    while True:
        tc = total_cost()
        if tc <= nav_budget + 1e-6:
            break
        cand = [c for c, sh in desired.items() if sh >= lot_size]
        if not cand:
            break
        # 分数越低越先减（若无分数则优先减字典序靠后，避免随机）
        c_drop = min(cand, key=lambda c: (scores_map.get(c, -np.inf), c))
        desired[c_drop] -= lot_size
        if desired[c_drop] <= 0:
            desired.pop(c_drop, None)

    return desired


def score_weights_from_picks_df(picks_df: pd.DataFrame) -> Dict[str, float]:
    """在 picks_df 截面内，权重与 pred_score 成正比（平移使最小值为正后归一化）。"""
    if picks_df.empty:
        return {}
    s = picks_df["pred_score"].astype(np.float64).values
    smin = float(np.min(s))
    raw = np.maximum(s - smin + 1e-12, 1e-18)
    tot = float(np.sum(raw))
    if tot <= 1e-18:
        return {}
    codes = picks_df["ts_code"].astype(str).tolist()
    return {c: float(w) for c, w in zip(codes, raw / tot)}


def run_backtest(
    scores: pd.DataFrame,
    close_lut: Dict[Tuple[str, str], float],
    initial_cash: float,
    n: int,
    k: int,
    lot_size: int = 100,
    score_lag: int = 1,
    commission_rate: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    STRATEGY = "score_weighted"
    dates = sorted(scores["trade_date"].unique())
    if len(dates) < 1:
        raise ValueError("交易日数量过少，无法回测。")

    cash = float(initial_cash)
    sellable: Dict[str, int] = {}
    locked: Dict[str, int] = {}
    last_close: Dict[str, float] = {}

    def close_on(d_: str, code: str) -> float:
        key = (d_, code)
        if key in close_lut:
            c = close_lut[key]
            last_close[code] = c
            return c
        return float(last_close.get(code, np.nan))

    rows: List[dict] = []

    def _curve_row(
        d_: str,
        nav_: float,
        cash_: float,
        n_pos_: int,
        *,
        score_date_: float | str = np.nan,
        turnover_sell_: float = np.nan,
        turnover_buy_: float = np.nan,
        commission_: float = 0.0,
    ) -> dict:
        return {
            "trade_date": d_,
            "nav": nav_,
            "cash": cash_,
            "n_pos": n_pos_,
            "turnover_sell": turnover_sell_,
            "turnover_buy": turnover_buy_,
            "score_date_used": score_date_,
            "strategy": STRATEGY,
            "commission": float(commission_),
        }

    def _n_positions() -> int:
        return len(
            {c for c, sh in sellable.items() if sh > 0} | {c for c, sh in locked.items() if sh > 0}
        )

    def _pick_picks_df(day_idx: pd.DataFrame) -> pd.DataFrame:
        sorted_df = day_idx.sort_values("pred_score", ascending=False).reset_index(drop=True)
        pool = sorted_df.head(min(n, len(sorted_df)))
        k_eff = min(k, len(pool))
        return pool.head(k_eff)

    def _weighted_buy_with_cash(
        trade_day: str,
        budget_cash: float,
        picks_df: pd.DataFrame,
        scores_map_local: Dict[str, float],
    ) -> Tuple[Dict[str, int], float]:
        """用 budget_cash 按分数加权买入（整手），返回 (locked 增量, 买入成交额)。"""
        if picks_df.empty or budget_cash <= 1e-9:
            return {}, 0.0
        weights = score_weights_from_picks_df(picks_df)
        tradable: List[str] = []
        px_map: Dict[str, float] = {}
        for _, row in picks_df.iterrows():
            code = str(row["ts_code"])
            px = close_on(trade_day, code)
            if np.isfinite(px) and px > 0:
                tradable.append(code)
                px_map[code] = float(px)
        if not tradable:
            return {}, 0.0
        sw = sum(weights.get(c, 0.0) for c in tradable)
        if sw <= 1e-18:
            return {}, 0.0
        wt = {c: weights[c] / sw for c in tradable}
        nav_mid = budget_cash
        desired: Dict[str, int] = {}
        for code in tradable:
            tgt = floor_to_lot(int(nav_mid * wt[code] / px_map[code]), lot_size)
            if tgt > 0:
                desired[code] = tgt
        desired = _trim_desired_cost_to_budget(desired, px_map, lot_size, nav_mid, scores_map_local)
        tmp_locked: Dict[str, int] = {}
        spent = 0.0
        remaining = budget_cash
        for code in tradable:
            tgt = desired.get(code, 0)
            if tgt <= 0:
                continue
            px = px_map[code]
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

    for di, d in enumerate(dates):
        _unlock_morning(sellable, locked)

        day_panel = scores[scores["trade_date"] == d]
        codes_today: List[str] = []
        for code in day_panel["ts_code"].unique():
            px = close_on(d, str(code))
            if np.isfinite(px):
                codes_today.append(str(code))
        if not codes_today:
            nav = cash + _mv_shares(close_on, d, sellable, locked)
            rows.append(_curve_row(d, nav, cash, _n_positions()))
            continue

        if score_lag <= 0 or di == 0:
            score_date = d
        else:
            idx = max(0, di - score_lag)
            score_date = dates[idx]

        day_scores = scores[(scores["trade_date"] == score_date) & (scores["ts_code"].isin(codes_today))].copy()
        if day_scores.empty:
            nav = cash + _mv_shares(close_on, d, sellable, locked)
            rows.append(_curve_row(d, nav, cash, _n_positions(), score_date_=score_date))
            continue

        day_idx = day_scores.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
        scores_map = day_idx.set_index("ts_code")["pred_score"].to_dict()

        picks_df = _pick_picks_df(day_idx)
        if picks_df.empty:
            nav = cash + _mv_shares(close_on, d, sellable, locked)
            rows.append(_curve_row(d, nav, cash, _n_positions(), score_date_=score_date))
            continue

        turnover_sell = 0.0
        no_position = sum(sellable.values()) == 0 and sum(locked.values()) == 0
        if not no_position:
            for code in list(sellable.keys()):
                sh0 = sellable.get(code, 0)
                qty = floor_to_lot(sh0, lot_size)
                if qty <= 0:
                    continue
                px = close_on(d, code)
                if not np.isfinite(px):
                    continue
                proceeds = qty * px
                sellable[code] = sh0 - qty
                if sellable[code] <= 0:
                    sellable.pop(code, None)
                cash += proceeds
                turnover_sell += proceeds

        sellable.clear()
        locked.clear()

        nav_before = cash + _mv_shares(close_on, d, sellable, locked)
        if nav_before <= 1e-9:
            rows.append(_curve_row(d, nav_before, cash, 0, score_date_=score_date))
            continue

        tmp_locked, spent = _weighted_buy_with_cash(d, cash, picks_df, scores_map)
        fee_day = commission_rate * (turnover_sell + spent)
        cash = cash - spent - fee_day
        locked.update(tmp_locked)
        nav_after = cash + _mv_shares(close_on, d, sellable, locked)
        rows.append(
            _curve_row(
                d,
                nav_after,
                cash,
                len({c for c, sh in locked.items() if sh > 0}),
                score_date_=score_date,
                turnover_sell_=turnover_sell if turnover_sell > 1e-9 else np.nan,
                turnover_buy_=spent,
                commission_=fee_day,
            )
        )

    curve = pd.DataFrame(rows)
    curve["daily_ret"] = curve["nav"].pct_change()

    total_ret = float(curve["nav"].iloc[-1] / initial_cash - 1.0)
    mdd = max_drawdown(curve["nav"].values.astype(float))
    sharpe = sharpe_ratio(curve["daily_ret"].values.astype(float))
    comm_sum = float(curve["commission"].fillna(0.0).sum()) if "commission" in curve.columns else 0.0
    calmar = float(total_ret / abs(mdd)) if mdd < -1e-12 else float("nan")

    stats = {
        "initial_cash": float(initial_cash),
        "final_nav": float(curve["nav"].iloc[-1]),
        "total_return": total_ret,
        "max_drawdown": mdd,
        "sharpe_ann_approx": sharpe,
        "calmar_ratio_approx": calmar,
        "trading_days": float(len(curve)),
        "lot_size": float(lot_size),
        "score_lag": float(score_lag),
        "strategy": STRATEGY,
        "commission_rate": float(commission_rate),
        "total_commission": comm_sum,
    }
    return curve, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="沪深300 pred_score 调仓回测（T+1 + 整手 + 可滞后打分）")
    parser.add_argument("--scores", default="outputs/val_scores.csv")
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "DL_DATA_ROOT",
            "/home/lhr/my_stuff/fundamentals_for_deep_learning/data",
        ),
    )
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="实际持仓只数：在候选池 Top-n 中取分数最高的 k 只，按 pred_score 加权配置权重",
    )
    parser.add_argument(
        "--lot-size",
        type=int,
        default=100,
        help="最小成交量单位（股），A 股 1 手=100；买卖向下取整到手",
    )
    parser.add_argument(
        "--score-lag",
        type=int,
        default=1,
        help="使用滞后 lag 个交易日的 pred_score 作为当日决策分数（默认 1，更贴近盘后信息）；设 0 则使用当日分数",
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=0.0,
        help="佣金（基点）：当日成交额合计 × (bps/10000)，从现金扣除；0 表示不计佣金",
    )
    parser.add_argument(
        "--benchmark",
        default="000300.SH.csv",
        help="相对于 data-root/market 下指数收益曲线（默认沪深300）；配合 pct_chg 列",
    )
    parser.add_argument("--no-benchmark", action="store_true", help="不拼接基准净值与超额收益列")
    parser.add_argument("--out-curve", default="outputs/equity_curve.csv")
    parser.add_argument("--out-summary", default="outputs/backtest_summary.csv")
    args = parser.parse_args()

    scores = load_scores(args.scores)
    dates = sorted(scores["trade_date"].unique())
    close_lut = load_close_lookup(args.data_root, dates)
    if not close_lut:
        raise RuntimeError("未能加载任何收盘价；请检查 --data-root。")

    curve, stats = run_backtest(
        scores,
        close_lut,
        args.cash,
        args.n,
        args.k,
        lot_size=args.lot_size,
        score_lag=args.score_lag,
        commission_rate=args.commission_bps / 10000.0,
    )

    if not args.no_benchmark:
        bench_ret = load_benchmark_daily_returns(args.data_root, args.benchmark)
        if len(bench_ret) > 0:
            curve = attach_benchmark(curve, bench_ret, args.cash)
            bench_tot = float(curve["benchmark_nav"].iloc[-1] / args.cash - 1.0)
            stats["benchmark_total_return"] = bench_tot
            stats["excess_total_return_vs_benchmark"] = float(stats["total_return"]) - bench_tot
        else:
            stats["benchmark_note"] = "file_missing_or_empty"

    _ensure_parent_dir(args.out_curve)
    _ensure_parent_dir(args.out_summary)
    curve.to_csv(args.out_curve, index=False)
    pd.DataFrame([stats]).to_csv(args.out_summary, index=False)

    print("=== backtest summary ===")
    for kk, vv in stats.items():
        print(f"{kk}: {vv}")
    print(f"wrote curve: {args.out_curve}")
    print(f"wrote summary: {args.out_summary}")


if __name__ == "__main__":
    main()
