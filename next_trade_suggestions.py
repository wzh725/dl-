#!/usr/bin/env python3
"""
根据 val_scores + 下一交易日日线收盘价，给出下一交易日的**可执行买卖建议**（股数、参考成交额）。

两种用法：
  1) **细单模式（推荐）**：`--state portfolio_state.json`，输出每笔买入/卖出多少股（整手），并可选写出收盘后状态 JSON 供次日链式使用。
  2) **摘要模式**：不传 `--state`，列出候选池 Top-n 内持仓 Top-k 及分数加权目标权重（无具体股数）。

策略固定为与 `backtest_score_weighted.py` 一致的 **score_weighted**（每日卖光可卖后按 pred_score 加权满仓）。

状态文件 JSON 字段：
  - **position_status**（可选）：`empty` / `holding`（或 `未建仓` / `已建仓`）— 标注当前是否已建仓，仅作文档与校验提示，实际持仓仍以 **sellable / locked** 为准。
  - **cash**：可用现金（元）
  - **lot_size**：最小交易单位，A 股通常为 100
  - **sellable**：{ "代码": 股数 } — 当日开盘即可卖的持仓（已不含昨买锁定）
  - **locked**：{ "代码": 股数 } — 昨日收盘前买入、按 T+1 **下一交易日早盘才可卖**的部分
  - **commission_bps**：佣金基点（可选），若 CLI 也传 `--commission-bps` 则以 CLI 为准

从未建仓：`sellable`/`locked` 均写 `{}` 即可。

成交价假设为 **下一交易日 daily/*.csv 中的收盘价**（与回测一致）；不涉及实盘下单接口。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from backtest_score_weighted import score_weights_from_picks_df
from portfolio_sim import (
    OrderLog,
    PortfolioState,
    load_close_map_for_day,
    load_portfolio_json,
    pick_picks_df,
    run_simulation,
    save_portfolio_json,
)


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


def codes_tradable_next_day(data_root: str, next_d: str) -> Optional[Set[str]]:
    fp = Path(data_root) / "daily" / f"{next_d}.csv"
    if not fp.is_file():
        return None
    df = pd.read_csv(fp, usecols=["ts_code", "close"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[np.isfinite(df["close"]) & (df["close"] > 0)]
    return set(df["ts_code"].astype(str))


def score_snapshot_date_for_day(
    sorted_score_dates: List[str],
    trade_day: str,
    score_lag: int,
) -> Tuple[str, str]:
    if score_lag <= 0:
        if trade_day in sorted_score_dates:
            return trade_day, "lag=0：使用当日截面（trade_date 与交易日对齐）"
        return sorted_score_dates[-1], "lag=0：指定/推断的交易日不在 CSV 中且无当日分数 → 暂用最后快照（请补充当日 pred_score）"

    if trade_day in sorted_score_dates:
        di = sorted_score_dates.index(trade_day)
        idx = max(0, di - score_lag)
        return sorted_score_dates[idx], f"lag={score_lag}：交易日位于打分样本内 → 使用 trade_date={sorted_score_dates[idx]} 的截面"

    di = len(sorted_score_dates)
    idx = max(0, di - score_lag)
    snap = sorted_score_dates[idx]
    label = trade_day if trade_day else "假定紧接打分样本之后的首个交易日"
    return snap, f"下一交易日 {label}（不在 CSV 交易日枚举中）→ lag={score_lag}，使用快照 trade_date={snap}"


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
        default=os.environ.get(
            "DL_DATA_ROOT",
            "/home/lhr/my_stuff/fundamentals_for_deep_learning/data",
        ),
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
    parser.add_argument("--commission-bps", type=float, default=None, help="覆盖 state 内 commission_bps")
    parser.add_argument("--out-csv", default="", help="摘要模式：写出目标池 CSV")
    parser.add_argument("--out-orders", default="", help="细单模式：写出指令明细 CSV")
    parser.add_argument("--out-next-state", default="", help="细单模式：写出推演收盘后状态 JSON（次日链式）")
    args = parser.parse_args()

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

        px_map = load_close_map_for_day(args.data_root, next_d)
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

        cbps = float(args.commission_bps if args.commission_bps is not None else st.commission_bps)
        commission_rate = cbps / 10000.0

        log, ps_end, fee, nav_after, sim_note = run_simulation(
            panel,
            px_map,
            scores_map,
            st,
            args.n,
            args.k,
            commission_rate,
        )

        print("=== 细单模式：下一交易日买卖指令（与回测规则对齐） ===")
        print(f"下一交易日: {next_d}  |  使用打分快照: {score_snap}  |  {snap_note}")
        print(f"推演说明: {sim_note}")
        print(f"佣金: commission_bps={cbps}  →  当日估算佣金 ≈ {fee:.2f} 元")
        print(f"推演收盘净值（收盘价计价）≈ {nav_after:.2f} 元；现金余额 ≈ {ps_end.cash:.2f} 元")
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
                commission_bps=cbps,
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
        tradable = codes_tradable_next_day(args.data_root, next_d)
        if tradable is not None:
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
