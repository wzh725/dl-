#!/usr/bin/env python3
"""
回测：截面 IC / Rank IC / 回归与方向指标（训练验证同定义）+ pred_score CSV 加权调仓仿真与 CLI。
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_preprocess import resolve_data_root


def global_pearson_ic(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 2:
        return float("nan")
    p = pred[mask]
    t = target[mask]
    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def daily_cross_section_ic_series(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    method: str = "pearson",
    min_names: int = 10,
) -> pd.Series:
    """
    每个交易日截面上 pred vs target 的相关（样本为该日所有股票）。
    method: 'pearson' | 'spearman'
    """
    df = pd.DataFrame(
        {
            "trade_date": pd.Series(dates).astype(str).values,
            "pred": np.asarray(pred, dtype=np.float64).reshape(-1),
            "y": np.asarray(target, dtype=np.float64).reshape(-1),
        }
    )
    ics = []
    idx_labels = []
    for d, g in df.groupby("trade_date", sort=True):
        g = g.dropna(subset=["pred", "y"])
        if len(g) < min_names:
            continue
        if float(g["pred"].std(ddof=1)) < 1e-12 or float(g["y"].std(ddof=1)) < 1e-12:
            continue
        if method == "pearson":
            ic = g["pred"].corr(g["y"])
        elif method == "spearman":
            ic = g["pred"].corr(g["y"], method="spearman")
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")
        if ic is not None and np.isfinite(ic):
            ics.append(float(ic))
            idx_labels.append(d)
    return pd.Series(ics, index=idx_labels, name=f"daily_ic_{method}")


def icir_from_daily(ic_daily: pd.Series) -> Tuple[float, float, float]:
    if len(ic_daily) < 2:
        return float("nan"), float("nan"), float("nan")
    m = float(ic_daily.mean())
    s = float(ic_daily.std(ddof=1))
    ir = m / s if s > 1e-12 else float("nan")
    return m, s, ir


def positive_ic_day_fraction(ic_daily: pd.Series) -> float:
    """有效 IC 交易日中，IC>0 的比例（衡量截面预测方向的持续性）。"""
    if len(ic_daily) < 1:
        return float("nan")
    return float((ic_daily.astype(np.float64) > 0.0).mean())


def prediction_error_metrics(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
    """全局 MAE / RMSE / MSE（与 IC 互补：刻画幅度误差）。"""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 1:
        return float("nan"), float("nan"), float("nan")
    err = pred[mask] - target[mask]
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    return mae, rmse, mse


def directional_hit_rate(
    pred: np.ndarray,
    target: np.ndarray,
    label_eps: float = 1e-8,
) -> Tuple[float, int]:
    """pred 与 label_return 同号的比例；剔除 |y| < label_eps 的样本。"""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(target) & (np.abs(target) >= label_eps)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    hit = np.mean(np.sign(pred[mask]) == np.sign(target[mask]))
    return float(hit), n


def validation_metrics_bundle(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    min_names_per_day: int = 10,
    label_eps: float = 1e-8,
) -> Dict[str, Any]:
    """汇总打印 / 写 JSON 用。"""
    ic_pearson_daily = daily_cross_section_ic_series(
        pred, target, dates, method="pearson", min_names=min_names_per_day
    )
    ic_rank_daily = daily_cross_section_ic_series(
        pred, target, dates, method="spearman", min_names=min_names_per_day
    )
    ic_mean, ic_std, icir = icir_from_daily(ic_pearson_daily)
    ric_mean, ric_std, ricir = icir_from_daily(ic_rank_daily)
    hit, hit_n = directional_hit_rate(pred, target, label_eps=label_eps)
    pos_pearson = positive_ic_day_fraction(ic_pearson_daily)
    pos_rank = positive_ic_day_fraction(ic_rank_daily)
    mae, rmse, mse = prediction_error_metrics(pred, target)
    return {
        "ic_global_pearson_all_samples": global_pearson_ic(pred, target),
        "ic_daily_pearson_mean": ic_mean,
        "ic_daily_pearson_std": ic_std,
        "icir_pearson": icir,
        "ic_daily_positive_day_frac": pos_pearson,
        "rank_ic_daily_mean": ric_mean,
        "rank_ic_daily_std": ric_std,
        "rank_icir": ricir,
        "rank_ic_daily_positive_day_frac": pos_rank,
        "n_days_ic_pearson": float(len(ic_pearson_daily)),
        "n_days_ic_rank": float(len(ic_rank_daily)),
        "mae_pred_vs_label": mae,
        "rmse_pred_vs_label": rmse,
        "mse_pred_vs_label": mse,
        "directional_hit_rate": hit,
        "directional_hit_rate_n": float(hit_n),
        "_series_ic_pearson_daily": ic_pearson_daily,
        "_series_ic_rank_daily": ic_rank_daily,
    }


def format_metrics_line(m: Dict[str, Any]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            vv = float(v)
            if np.isnan(vv):
                return "nan"
            return f"{vv:.6g}"
        return str(v)

    keys = [
        "ic_global_pearson_all_samples",
        "ic_daily_pearson_mean",
        "ic_daily_pearson_std",
        "icir_pearson",
        "ic_daily_positive_day_frac",
        "rank_ic_daily_mean",
        "rank_icir",
        "rank_ic_daily_positive_day_frac",
        "mae_pred_vs_label",
        "rmse_pred_vs_label",
        "directional_hit_rate",
        "n_days_ic_pearson",
    ]
    parts = [f"{k}={fmt(m.get(k))}" for k in keys]
    return " | ".join(parts)


def metrics_dict_for_json(m: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in m.items() if not k.startswith("_")}
    for k, v in list(out.items()):
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
    return out


def _ensure_parent_dir(path: str) -> None:
    p = os.path.dirname(os.path.abspath(path))
    if p:
        os.makedirs(p, exist_ok=True)


def load_trade_price_lookup(
    data_root: str,
    dates: List[str],
    *,
    price_col: str = "open",
) -> Dict[Tuple[str, str], float]:
    lookup: Dict[Tuple[str, str], float] = {}
    daily_dir = os.path.join(data_root, "daily")
    for d in dates:
        fp = os.path.join(daily_dir, f"{d}.csv")
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp, usecols=["ts_code", price_col])
        for _, row in df.iterrows():
            code = str(row["ts_code"])
            px = float(row[price_col])
            if np.isfinite(px) and px > 0:
                lookup[(d, code)] = px
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
    if "label_return" in df.columns:
        df["label_return"] = pd.to_numeric(df["label_return"], errors="coerce")
    df = df.dropna(subset=["pred_score"])
    return df


def append_scores_panel_metrics(
    stats: Dict[str, Any],
    scores: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    restrict_to_equity_calendar: bool = True,
    min_names_ic: int = 10,
    out_daily_ic_csv: Optional[str] = None,
) -> None:
    """
    与 `validation_metrics_bundle` 同定义：截面 Pearson/Rank IC、ICIR、方向胜率、全局 MAE/RMSE。
    依赖 scores CSV 的 `label_return`（训练中 `workflow=backtest` 导出）。
    restrict_to_equity_calendar=True：只统计净值曲线上出现的 `trade_date`，与仿真日历对齐；
    False：使用 scores 中带标签的全部行。
    """
    if "label_return" not in scores.columns:
        stats["panel_metrics_note"] = "no_label_return_column（仅 pred_score；无法在本次回测 summary 中间接写 IC）"
        return

    sub = scores[
        ["trade_date", "ts_code", "pred_score", "label_return"]
    ].drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
    sub = sub[np.isfinite(sub["pred_score"].astype(float)) & np.isfinite(sub["label_return"].astype(float))]
    if restrict_to_equity_calendar:
        allow = set(curve["trade_date"].astype(str).unique())
        sub = sub[sub["trade_date"].astype(str).isin(allow)]
    if sub.empty:
        stats["panel_metrics_note"] = "no_finite_pred_label_rows_for_panel_metrics"
        return

    bundle = validation_metrics_bundle(
        sub["pred_score"].to_numpy(dtype=np.float64),
        sub["label_return"].to_numpy(dtype=np.float64),
        sub["trade_date"].to_numpy(dtype=object),
        min_names_per_day=min_names_ic,
        label_eps=1e-8,
    )
    flat = metrics_dict_for_json(bundle)
    for key, val in flat.items():
        if val is None or isinstance(val, (bool, int, float, str)):
            stats[f"panel_{key}"] = val
        else:
            stats[f"panel_{key}"] = float(val) if hasattr(val, "__float__") else val

    n_ic = stats.get("panel_n_days_ic_pearson")
    if isinstance(n_ic, float) and (np.isnan(n_ic) or n_ic <= 0):
        stats.setdefault(
            "panel_metrics_note",
            "label_return_present_but_ic_days_empty_check_min_names",
        )

    if out_daily_ic_csv:
        s_p = bundle.get("_series_ic_pearson_daily")
        s_r = bundle.get("_series_ic_rank_daily")
        parts = []
        if isinstance(s_p, pd.Series) and len(s_p):
            parts.append(pd.DataFrame({"trade_date": s_p.index.astype(str), "ic_pearson": s_p.values}))
        if isinstance(s_r, pd.Series) and len(s_r):
            parts.append(pd.DataFrame({"trade_date": s_r.index.astype(str), "ic_rank": s_r.values}))
        if parts:
            _ensure_parent_dir(out_daily_ic_csv)
            merged = parts[0]
            for df_i in parts[1:]:
                merged = merged.merge(df_i, on="trade_date", how="outer")
            merged.sort_values("trade_date").to_csv(out_daily_ic_csv, index=False)
            stats["panel_daily_ic_csv"] = out_daily_ic_csv


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


# --------- A 股费用（按当前项目约定）---------

STAMP_DUTY_RATE_SELL = 0.001  # 印花税：卖出成交金额千分之一
TRANSFER_FEE_RATE = 0.00006  # 过户费：买卖双向，成交股数系数（仅 60 开头）
TRANSFER_FEE_MIN_CNY = 1.0
BROKER_COMMISSION_MIN_FEE_CNY = 5.0


def transfer_fee(ts_code: str, amount: float, shares: int) -> float:
    """
    过户费：仅上证 60 开头代码收取，买卖双向，按成交股数计，单笔最低 1 元。
    """
    if amount <= 1e-12 or shares <= 0:
        return 0.0
    code = str(ts_code or "")
    if not code.startswith("60"):
        return 0.0
    raw = float(shares) * TRANSFER_FEE_RATE
    return max(TRANSFER_FEE_MIN_CNY, raw)


def broker_commission_fee(amount: float, commission_rate: float) -> float:
    if amount <= 1e-12 or commission_rate <= 1e-15:
        return 0.0
    raw = abs(amount) * commission_rate
    return max(BROKER_COMMISSION_MIN_FEE_CNY, raw)


def fees_on_sell_turnover(ts_code: str, turnover: float, shares: int, commission_rate: float) -> float:
    """单笔卖出合计费用（佣金 + 印花税 + 条件过户）。"""
    fees = broker_commission_fee(turnover, commission_rate)
    fees += abs(turnover) * STAMP_DUTY_RATE_SELL
    fees += transfer_fee(ts_code, turnover, shares)
    return float(fees)


def fees_on_buy_turnover(ts_code: str, turnover: float, shares: int, commission_rate: float) -> float:
    """单笔买入合计费用（佣金 + 条件过户）；无印花税。"""
    fees = broker_commission_fee(turnover, commission_rate)
    fees += transfer_fee(ts_code, turnover, shares)
    return float(fees)


def score_weighted_buys_for_cash_budget(
    px_map: Dict[str, float],
    budget_cash: float,
    picks_df: pd.DataFrame,
    scores_map: Dict[str, float],
    lot_size: int,
    commission_rate: float,
) -> Tuple[Dict[str, int], float, float]:
    """
    按 pred_score 加权分配整手买入；每笔扣佣金/沪市过户。
    Returns (locked 增量字典, 买入成交额 gross, 买入侧费用合计)
    """
    if picks_df.empty or budget_cash <= 1e-9:
        return {}, 0.0, 0.0
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
        return {}, 0.0, 0.0
    sw = sum(weights.get(c, 0.0) for c in tradable)
    if sw <= 1e-18:
        return {}, 0.0, 0.0
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
    buy_fees_total = 0.0
    remaining = budget_cash
    for code in tradable:
        tgt = desired.get(code, 0)
        if tgt <= 0:
            continue
        px = tpx[code]
        cost_gross = tgt * px
        bf = fees_on_buy_turnover(code, cost_gross, tgt, commission_rate)
        while cost_gross + bf > remaining + 1e-6 and tgt >= lot_size:
            tgt -= lot_size
            cost_gross = tgt * px
            bf = fees_on_buy_turnover(code, cost_gross, tgt, commission_rate)
        if tgt <= 0:
            continue
        cost_gross = tgt * px
        bf = fees_on_buy_turnover(code, cost_gross, tgt, commission_rate)
        total_out = cost_gross + bf
        if total_out > remaining + 1e-6:
            continue
        remaining -= total_out
        spent += cost_gross
        buy_fees_total += bf
        tmp_locked[code] = tmp_locked.get(code, 0) + tgt
    return tmp_locked, spent, buy_fees_total


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
    trade_price_lut: Dict[Tuple[str, str], float],
    initial_cash: float,
    n: int,
    k: int,
    lot_size: int = 100,
    score_lag: int = 1,
    commission_rate: float = 0.0002,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    STRATEGY = "score_weighted"
    dates = sorted(scores["trade_date"].unique())
    if len(dates) < 1:
        raise ValueError("交易日数量过少，无法回测。")

    cash = float(initial_cash)
    sellable: Dict[str, int] = {}
    locked: Dict[str, int] = {}
    last_trade_px: Dict[str, float] = {}

    def trade_px_on(d_: str, code: str) -> float:
        key = (d_, code)
        if key in trade_price_lut:
            c = trade_price_lut[key]
            last_trade_px[code] = c
            return c
        return float(last_trade_px.get(code, np.nan))

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
        # 「commission」列：印花税 + 过户费 + 券商佣金（按 commission_rate）当日合计（元）
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
    ) -> Tuple[Dict[str, int], float, float]:
        """用 budget_cash 按分数加权买入（整手）；返回 (locked 增量, 买入成交额 gross, 买入侧费用合计)。"""
        px_map: Dict[str, float] = {}
        for _, row in picks_df.iterrows():
            code = str(row["ts_code"])
            px = trade_px_on(trade_day, code)
            if np.isfinite(px) and px > 0:
                px_map[code] = float(px)
        return score_weighted_buys_for_cash_budget(
            px_map, budget_cash, picks_df, scores_map_local, lot_size, commission_rate
        )

    for di, d in enumerate(dates):
        _unlock_morning(sellable, locked)

        day_panel = scores[scores["trade_date"] == d]
        codes_today: List[str] = []
        for code in day_panel["ts_code"].unique():
            px = trade_px_on(d, str(code))
            if np.isfinite(px):
                codes_today.append(str(code))
        if not codes_today:
            nav = cash + _mv_shares(trade_px_on, d, sellable, locked)
            rows.append(_curve_row(d, nav, cash, _n_positions()))
            continue

        if score_lag <= 0 or di == 0:
            score_date = d
        else:
            idx = max(0, di - score_lag)
            score_date = dates[idx]

        day_scores = scores[(scores["trade_date"] == score_date) & (scores["ts_code"].isin(codes_today))].copy()
        if day_scores.empty:
            nav = cash + _mv_shares(trade_px_on, d, sellable, locked)
            rows.append(_curve_row(d, nav, cash, _n_positions(), score_date_=score_date))
            continue

        day_idx = day_scores.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
        scores_map = day_idx.set_index("ts_code")["pred_score"].to_dict()

        picks_df = _pick_picks_df(day_idx)
        if picks_df.empty:
            nav = cash + _mv_shares(trade_px_on, d, sellable, locked)
            rows.append(_curve_row(d, nav, cash, _n_positions(), score_date_=score_date))
            continue

        # 交易逻辑（作业推荐）：卖出现有持仓中得分最低的 k 只，再买入候选里得分最高且当前未持有的标的。
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
            px = trade_px_on(d, code)
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

        nav_before = cash + _mv_shares(trade_px_on, d, sellable, locked)
        if nav_before <= 1e-9:
            rows.append(_curve_row(d, nav_before, cash, 0, score_date_=score_date))
            continue

        held_after_sell = {
            c for c, sh in sellable.items() if sh > 0
        } | {c for c, sh in locked.items() if sh > 0}
        slots = max(0, int(k) - len(held_after_sell))
        buy_candidates = [c for c in target_codes if c not in held_after_sell]
        buy_codes = buy_candidates[:slots]
        if buy_codes:
            picks_buy_df = picks_df[picks_df["ts_code"].astype(str).isin(set(buy_codes))].copy()
            tmp_locked, spent, buy_fees = _weighted_buy_with_cash(d, cash, picks_buy_df, scores_map)
        else:
            tmp_locked, spent, buy_fees = {}, 0.0, 0.0
        fee_day = sell_fees + buy_fees
        cash = cash - spent - buy_fees
        locked.update(tmp_locked)
        nav_after = cash + _mv_shares(trade_px_on, d, sellable, locked)
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
        "broker_commission_rate": float(commission_rate),
        "fee_model_cn_a_note": (
            "stamp_sell_rate=0.1%; transfer=SSE 60* only, by shares * 0.00006 per leg (min 1 CNY); "
            "broker_commission=min(5,max) from commission_rate on each leg"
        ),
        "total_commission": comm_sum,
    }
    return curve, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="沪深300 pred_score 调仓回测（T+1 + 整手 + 可滞后打分）")
    parser.add_argument("--scores", default="outputs/val_scores.csv")
    parser.add_argument("--data-root", default=os.environ.get("DL_DATA_ROOT", ""))
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
        "--trade-price-col",
        choices=("open", "close"),
        default="open",
        help="交易撮合价格列（默认 open，满足每日开盘买卖）。",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.0002,
        help=(
            "券商佣金费率（小数，万二=0.0002）；单笔不足 5 元按 5 元。"
            "另自动计：卖出印花税 0.1%%；上证 60* 买卖双向过户费（按成交股数，单笔最低 1 元）。"
        ),
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="兼容旧参数：基点制（万三=3），若提供则覆盖 --commission-rate",
    )
    parser.add_argument(
        "--benchmark",
        default="000300.SH.csv",
        help="相对于 data-root/market 下指数收益曲线（默认沪深300）；配合 pct_chg 列",
    )
    parser.add_argument("--no-benchmark", action="store_true", help="不拼接基准净值与超额收益列")
    parser.add_argument("--out-curve", default="outputs/equity_curve.csv")
    parser.add_argument("--out-summary", default="outputs/backtest_summary.csv")
    parser.add_argument(
        "--no-panel-metrics",
        action="store_true",
        help="不在 summary 中写入基于 label_return 的 panel_* IC/胜率（仍会跑组合回测）",
    )
    parser.add_argument(
        "--panel-metrics-all-score-dates",
        action="store_true",
        help="截面指标改用 scores 中带标签的全部行（默认仅用净值曲线出现的 trade_date 以对齐仿真区间）",
    )
    parser.add_argument(
        "--min-names-panel-ic",
        type=int,
        default=10,
        help="逐日截面 IC 时该日至少需要的股票数（与训练侧 --min-names-ic 一致）",
    )
    parser.add_argument(
        "--out-panel-daily-ic",
        default="",
        help="可选：写出按日 Pearson/Rank IC 的 CSV；留空则不写",
    )
    args = parser.parse_args()
    if args.commission_bps is not None:
        args.commission_rate = float(args.commission_bps) / 10000.0

    args.data_root = resolve_data_root(args.data_root)

    scores = load_scores(args.scores)
    dates = sorted(scores["trade_date"].unique())
    trade_price_lut = load_trade_price_lookup(
        args.data_root,
        dates,
        price_col=str(args.trade_price_col),
    )
    if not trade_price_lut:
        raise RuntimeError("未能加载任何撮合价格；请检查 --data-root 与 --trade-price-col。")

    curve, stats = run_backtest(
        scores,
        trade_price_lut,
        args.cash,
        args.n,
        args.k,
        lot_size=args.lot_size,
        score_lag=args.score_lag,
        commission_rate=float(args.commission_rate),
    )
    stats["trade_price_col"] = str(args.trade_price_col)

    if not args.no_benchmark:
        bench_ret = load_benchmark_daily_returns(args.data_root, args.benchmark)
        if len(bench_ret) > 0:
            curve = attach_benchmark(curve, bench_ret, args.cash)
            bench_tot = float(curve["benchmark_nav"].iloc[-1] / args.cash - 1.0)
            stats["benchmark_total_return"] = bench_tot
            stats["excess_total_return_vs_benchmark"] = float(stats["total_return"]) - bench_tot
        else:
            stats["benchmark_note"] = "file_missing_or_empty"

    if not args.no_panel_metrics:
        daily_ic_path = (args.out_panel_daily_ic or "").strip() or None
        append_scores_panel_metrics(
            stats,
            scores,
            curve,
            restrict_to_equity_calendar=not args.panel_metrics_all_score_dates,
            min_names_ic=args.min_names_panel_ic,
            out_daily_ic_csv=daily_ic_path,
        )

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
