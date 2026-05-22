"""
交易日历工具：基于 data/trade_cal.csv（SSE）推导「下一开市日」等。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def _norm_ymd(s: str) -> str:
    y = str(s).strip().replace("-", "")[:8]
    if len(y) != 8 or not y.isdigit():
        raise ValueError(f"日期须为 YYYY-MM-DD 或 YYYYMMDD: {s!r}")
    return y


def load_open_trading_days(data_root: str) -> List[str]:
    path = Path(data_root) / "trade_cal.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少交易日历: {path}")
    cal = pd.read_csv(path)
    cal["cal_date"] = cal["cal_date"].astype(str)
    o = cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).unique()
    return sorted(o.tolist())


def next_trading_day_strictly_after(anchor_yyyymmdd: str, data_root: str) -> str:
    """
    若某日收盘后只能使用 T 及以前的数据，则监督验证 / 回测截面的首日锚定应不早于
    「T 的下一个交易日」。本函数返回日历中严格晚于 anchor 的第一个开市日（字符串 8 位）。"""
    d = _norm_ymd(anchor_yyyymmdd)
    days = load_open_trading_days(data_root)
    for od in days:
        if od > d:
            return od
    raise ValueError(
        f"在 trade_cal 中找不到严格晚于 {d} 的开市日（请检查 data-root 与日历区间）。"
    )


def fmt_yyyy_mm_dd(yyyymmdd: str) -> str:
    s = _norm_ymd(yyyymmdd)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
