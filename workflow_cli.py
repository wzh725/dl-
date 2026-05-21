#!/usr/bin/env python3
"""
统一入口：两种工作流

1) backtest：训练（train-end）→ 从「训练截止日的下一交易日」到 backtest-end 的监督打分 → 历史回测
2) predict-next：用可监督样本尽可能训满 + 面板末日推理打分 → 读持仓 JSON → 写出「下一步操作」JSON

示例见 README「workflow_cli」小节。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from calendar_trading import fmt_yyyy_mm_dd, next_trading_day_strictly_after, _norm_ymd
from portfolio_sim import infer_position_mode_from_state_dict

_DL_DIR = Path(__file__).resolve().parent


def _fmt_input_date(s: str) -> str:
    """训练脚本使用 YYYY-MM-DD。"""
    return fmt_yyyy_mm_dd(s)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_next_trade_date_from_scores_and_daily(data_root: str, scores_path: str) -> str:
    from next_trade_suggestions import infer_next_trade_date_from_daily, load_scores

    scores = load_scores(scores_path)
    dates = sorted(scores["trade_date"].unique())
    if not dates:
        raise SystemExit("scores CSV 为空")
    last_s = dates[-1]
    next_d = infer_next_trade_date_from_daily(data_root, last_s)
    if not next_d:
        raise SystemExit(
            "无法从 daily/ 推断下一交易日：请显式传入 --next-trade-date YYYYMMDD。"
        )
    return next_d


def cmd_backtest(ns: argparse.Namespace, train_argv: List[str]) -> None:
    data_root = ns.data_root
    train_end_norm = _norm_ymd(ns.train_end)
    backtest_end_norm = _norm_ymd(ns.backtest_end)
    val_start_d = next_trading_day_strictly_after(train_end_norm, data_root)
    if val_start_d > backtest_end_norm:
        raise SystemExit(
            f"回测结束日 {backtest_end_norm} 早于首个可行验证锚定日 {val_start_d}（训练截止下一交易日）。"
            "请增大 --backtest-end 或减小 --train-end。"
        )

    train_cmd: List[str] = []
    script_path = str(_DL_DIR / "train_transformer_baseline.py")
    inner: List[str] = [
        "--workflow",
        "backtest",
        "--data-root",
        data_root,
        "--train-start",
        _fmt_input_date(ns.train_start),
        "--train-end",
        _fmt_input_date(ns.train_end),
        "--val-start",
        fmt_yyyy_mm_dd(val_start_d),
        "--val-end",
        fmt_yyyy_mm_dd(backtest_end_norm),
        "--export-scores",
        ns.scores_out,
    ]
    inner.extend(train_argv)

    if ns.launcher == "torchrun":
        train_cmd.extend(
            [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={ns.nproc}",
                script_path,
            ]
        )
        train_cmd.extend(inner)
    else:
        train_cmd.append(sys.executable)
        train_cmd.append(script_path)
        train_cmd.extend(inner)

    print("[workflow_cli] 训练 + 验证导出:", " ".join(train_cmd), flush=True)
    subprocess.run(train_cmd, check=True)

    bt_cmd = [
        sys.executable,
        str(_DL_DIR / "backtest_score_weighted.py"),
        "--scores",
        ns.scores_out,
        "--data-root",
        data_root,
        "--out-curve",
        ns.out_curve,
        "--out-summary",
        ns.out_summary,
        "--cash",
        str(ns.cash),
        "--n",
        str(ns.n_pool),
        "--k",
        str(ns.k_hold),
        "--score-lag",
        str(ns.score_lag),
        "--commission-bps",
        str(ns.commission_bps),
    ]
    if ns.no_benchmark:
        bt_cmd.append("--no-benchmark")
    print("[workflow_cli] 历史回测:", " ".join(bt_cmd), flush=True)
    subprocess.run(bt_cmd, check=True)
    print(f"[workflow_cli] 完成。净值: {ns.out_curve} ；摘要: {ns.out_summary}", flush=True)


def cmd_predict_next(ns: argparse.Namespace, train_argv: List[str]) -> None:
    data_root = ns.data_root
    raw_state = _read_json(ns.state_in)
    pos = infer_position_mode_from_state_dict(raw_state)

    if not ns.skip_train:
        train_cmd: List[str] = []
        script_path = str(_DL_DIR / "train_transformer_baseline.py")
        inner = [
            "--workflow",
            "predict-next",
            "--data-root",
            data_root,
            "--train-start",
            _fmt_input_date(ns.train_start),
            "--train-end",
            _fmt_input_date(ns.train_end),
            "--export-scores",
            ns.export_scores,
        ]
        inner.extend(train_argv)
        if ns.launcher == "torchrun":
            train_cmd.extend(
                ["torchrun", "--standalone", f"--nproc_per_node={ns.nproc}", script_path]
            )
            train_cmd.extend(inner)
        else:
            train_cmd.append(sys.executable)
            train_cmd.append(script_path)
            train_cmd.extend(inner)
        print("[workflow_cli] 训练（predict-next）:", " ".join(train_cmd), flush=True)
        subprocess.run(train_cmd, check=True)
    else:
        print("[workflow_cli] 跳过训练，使用已有 scores:", ns.export_scores, flush=True)

    next_d = (ns.next_trade_date or "").strip().replace("-", "")
    if not next_d:
        next_d = _infer_next_trade_date_from_scores_and_daily(data_root, ns.export_scores)
    if len(next_d) != 8 or not next_d.isdigit():
        raise SystemExit(f"无效 next_trade_date: {next_d!r}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp_o:
        orders_path = tmp_o.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp_s:
        next_state_tmp = tmp_s.name
    try:
        sim_cmd = [
            sys.executable,
            str(_DL_DIR / "next_trade_suggestions.py"),
            "--scores",
            ns.export_scores,
            "--data-root",
            data_root,
            "--state",
            ns.state_in,
            "--next-trade-date",
            next_d,
            "--n",
            str(ns.n_pool),
            "--k",
            str(ns.k_hold),
            "--score-lag",
            str(ns.score_lag),
            "--out-orders",
            orders_path,
            "--out-next-state",
            next_state_tmp,
        ]
        if ns.commission_bps is not None:
            sim_cmd.extend(["--commission-bps", str(ns.commission_bps)])
        print("[workflow_cli] 下一交易日推演:", " ".join(sim_cmd), flush=True)
        subprocess.run(sim_cmd, check=True)

        orders_rows: List[Dict[str, Any]] = []
        if os.path.isfile(orders_path) and os.path.getsize(orders_path) > 0:
            orders_df = pd.read_csv(orders_path)
            orders_rows = orders_df.to_dict(orient="records")

        next_state_raw: Optional[Dict[str, Any]] = None
        if os.path.isfile(next_state_tmp) and os.path.getsize(next_state_tmp) > 0:
            with open(next_state_tmp, "r", encoding="utf-8") as f:
                next_state_raw = json.load(f)

        scores = pd.read_csv(ns.export_scores)
        dates = sorted(scores["trade_date"].astype(str).str.replace("-", "", regex=False).unique())
        from next_trade_suggestions import score_snapshot_date_for_day

        snap_used, snap_note = score_snapshot_date_for_day(dates, next_d, int(ns.score_lag))

        out_payload = {
            "workflow": "predict-next",
            "input_position_mode": pos,
            "next_trade_date": next_d,
            "score_snapshot_trade_date": str(snap_used),
            "score_snapshot_note": snap_note,
            "score_lag": int(ns.score_lag),
            "candidate_pool_n": ns.n_pool,
            "hold_top_k": ns.k_hold,
            "orders": orders_rows,
            "portfolio_after_close": next_state_raw,
            "artifacts": {
                "scores_csv": ns.export_scores,
            },
            "notes": (
                "orders 列为整手买卖指令；成交价假设为下一交易日 daily 收盘价。"
                "portfolio_after_close 为当日收盘后状态（当日买入在 locked）。"
            ),
        }

        outp = Path(ns.ops_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        print(f"[workflow_cli] 已写下一步操作 JSON: {outp}", flush=True)
    finally:
        try:
            os.unlink(orders_path)
        except OSError:
            pass
        try:
            os.unlink(next_state_tmp)
        except OSError:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="深度学习作业：统一工作流 CLI")
    subs = p.add_subparsers(dest="cmd", required=True)

    pb = subs.add_parser("backtest", help="训练 → 验证段打分 → 历史回测（验证起点=训练截止的下一交易日）")
    pb.add_argument("--data-root", default=os.environ.get("DL_DATA_ROOT", ""))
    pb.add_argument("--train-start", required=True, help="训练锚定起始 YYYY-MM-DD / YYYYMMDD")
    pb.add_argument("--train-end", required=True, help="训练锚定结束（含）；回测截面从下一交易日开始")
    pb.add_argument(
        "--backtest-end",
        required=True,
        help="回测 / 打分导出的最后锚定日（含），对应验证 val-end",
    )
    pb.add_argument("--scores-out", default="outputs/workflow_backtest_scores.csv")
    pb.add_argument("--out-curve", default="outputs/workflow_equity_curve.csv")
    pb.add_argument("--out-summary", default="outputs/workflow_backtest_summary.csv")
    pb.add_argument("--cash", type=float, default=1_000_000.0)
    pb.add_argument("--n-pool", type=int, default=30, dest="n_pool")
    pb.add_argument("--k-hold", type=int, default=10, dest="k_hold")
    pb.add_argument("--score-lag", type=int, default=1)
    pb.add_argument("--commission-bps", type=float, default=0.0)
    pb.add_argument("--no-benchmark", action="store_true")
    pb.add_argument("--launcher", choices=("python", "torchrun"), default="python")
    pb.add_argument("--nproc", type=int, default=1, help="torchrun 时每机进程数（GPU 数）")
    pb.add_argument(
        "train_argv",
        nargs=argparse.REMAINDER,
        help="传给 train_transformer_baseline.py 的额外参数，前置 -- ，例如: -- --epochs 3 --batch-size 512",
    )

    pn = subs.add_parser(
        "predict-next",
        help="predict-next 训练 + 末日打分 + state JSON → 写出下一步操作 JSON",
    )
    pn.add_argument("--data-root", default=os.environ.get("DL_DATA_ROOT", ""))
    pn.add_argument("--train-start", required=True)
    pn.add_argument(
        "--train-end",
        required=True,
        help="仍可计算标签的最后锚定日（通常为面板末日的前一交易日，当 horizon=1）",
    )
    pn.add_argument(
        "--export-scores",
        default="outputs/workflow_predict_next_scores.csv",
        help="训练写出 pred_score；predict-next 也用它喂给 next_trade_suggestions",
    )
    pn.add_argument("--state-in", required=True, help="账户 JSON（示例见 examples/）")
    pn.add_argument("--ops-out", required=True, help="输出：下一步操作 JSON")
    pn.add_argument(
        "--next-trade-date",
        default="",
        help="下一交易日 YYYYMMDD；省略则从 daily/ 推断（须有比打分最后一日更晚的 CSV）",
    )
    pn.add_argument("--skip-train", action="store_true", help="跳过训练，仅基于已有 export-scores 推演")
    pn.add_argument("--n-pool", type=int, default=30, dest="n_pool")
    pn.add_argument("--k-hold", type=int, default=10, dest="k_hold")
    pn.add_argument("--score-lag", type=int, default=1)
    pn.add_argument("--commission-bps", type=float, default=None)
    pn.add_argument("--launcher", choices=("python", "torchrun"), default="python")
    pn.add_argument("--nproc", type=int, default=1)
    pn.add_argument(
        "train_argv",
        nargs=argparse.REMAINDER,
        help="传给 train_transformer_baseline.py 的额外参数，前置 --",
    )

    args = p.parse_args()
    if not getattr(args, "data_root", None) or not str(args.data_root).strip():
        args.data_root = "/home/lhr/my_stuff/fundamentals_for_deep_learning/data"

    tv = getattr(args, "train_argv", None) or []
    if tv and tv[0] == "--":
        tv = tv[1:]

    if args.cmd == "backtest":
        cmd_backtest(args, tv)
    elif args.cmd == "predict-next":
        cmd_predict_next(args, tv)
    else:
        raise SystemExit("unknown cmd")


if __name__ == "__main__":
    main()
