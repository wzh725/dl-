"""
与大作业 PDF 5.1 对齐的基础评估指标（验证 / 测试集）：
- 全局 Pearson IC（全体样本混池，仅供参考）
- **按交易日截面 Pearson IC**（常见 IC 定义），均值、标准差、ICIR、**IC>0 交易日占比**
- **按交易日 Rank IC**（Spearman，秩相关）及 Rank ICIR、**Rank IC>0 占比**
- **回归误差**：全局 MAE / RMSE / MSE（刻画幅度误差，与秩相关互补）
- **方向胜率**：pred 与 label 同号比例（可选剔除 |label| 过小的样本）
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


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
