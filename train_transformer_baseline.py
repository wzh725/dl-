#!/usr/bin/env python3
"""
默认使用本地 daily/ 覆盖的「全市场」股票池（排除 ST、北交所），日线全部数值列（不含进阶 metric/资金流/新闻）。
可用 `--stock-pool` 换成 hs300 / 创业板 / 科创板子池。
小型 Transformer Encoder 训练基线：

使用 `--workflow` 区分两条主线：
- **backtest**（默认）：训练 / 验证切分 → 导出验证集打分 → `backtest_score_weighted.py` 历史回测；
- **predict-next**：用历史上全部「可监督」样本训练（无验证集），再对 **面板最后交易日**（或 `--export-infer-anchor`）做推理导出 → `next_trade_suggestions.py` 生成下一交易日操作。

默认（非 --quick）示例：
- **训练样本锚定日**：`--train-start`（默认 2016-01-01）～ `--train-end`（默认 2025-12-31）
- **验证 / 样本外测试**：`--val-start`～`--val-end`（默认 2026-01-01～2026-02-01）
- **日线加载右边界**：`--load-end`（默认在 val-end 之后再顺延若干日历日，便于 horizon 末端算标签）

使用 `--quick` 做短区间冒烟测试。训练/验证时间切分见 README。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from data_processor import DataProcessor
from eval_metrics import (
    format_metrics_line,
    metrics_dict_for_json,
    validation_metrics_bundle,
)


def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup() -> tuple[bool, int, int]:
    if not ddp_enabled():
        return False, 0, 0
    if not torch.cuda.is_available():
        raise RuntimeError("WORLD_SIZE>1 时需要 CUDA + nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank


def default_load_end(val_end: str, buffer_days: int = 45) -> str:
    """验证期末之后多加载一段日历日，避免 horizon 末端标签缺少未来日线。"""
    dt = datetime.strptime(val_end.strip(), "%Y-%m-%d")
    return (dt + timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def _ensure_parent_dir(file_path: str) -> None:
    """写入 CSV 前创建父目录，避免 outputs/ 等路径不存在报错。"""
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _panel_last_trade_date_str(processor: DataProcessor) -> str:
    """面板索引上的最后交易日，规范为 YYYYMMDD。"""
    lv = processor.df.index.get_level_values("trade_date")
    cand: list[str] = []
    for x in lv:
        s = str(x).replace("-", "").strip()
        if len(s) >= 8:
            s8 = s[:8]
            if s8.isdigit():
                cand.append(s8)
    if not cand:
        raise RuntimeError("无法从面板推断最后交易日（索引为空或格式异常）")
    return max(cand)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        return x + self.pe[:, :t, :]


class TsTransformer(nn.Module):
    """纯 Attention Encoder：输入 (B, T, F)，回归预测下一窗口锚点收益。"""

    def __init__(
        self,
        feat_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=512)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


def _to_tensors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[TensorDataset, TensorDataset]:
    def clean(a: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(a, dtype=np.float32)

    Xt = torch.from_numpy(clean(X_train))
    yt = torch.from_numpy(clean(y_train))
    Xv = torch.from_numpy(clean(X_val))
    yv = torch.from_numpy(clean(y_val))
    return TensorDataset(Xt, yt), TensorDataset(Xv, yv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="日线 Transformer 基线训练（默认全市场池：排除 ST/北交所，见 --stock-pool）"
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "DL_DATA_ROOT",
            "/home/lhr/my_stuff/fundamentals_for_deep_learning/data",
        ),
        help="数据根目录",
    )
    parser.add_argument(
        "--stock-pool",
        choices=("all", "hs300", "cyb", "kcb"),
        default="all",
        help="股票池：all=本地 daily 有行情的标的（排除 ST/北交所）；hs300 需 data/index_weight；cyb/kcb 按 basic.csv 市场列",
    )
    parser.add_argument(
        "--workflow",
        choices=("backtest", "predict-next"),
        default="backtest",
        help=(
            "backtest：训练+验证切分，可导出验证打分并做历史回测；"
            "predict-next：仅用监督样本全集训练（无验证 IC），必须配合 --export-scores，"
            "默认 --export-infer-anchor auto（面板末日推理打分 → 次日操作建议）"
        ),
    )
    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="缩短日期区间快速试跑")
    parser.add_argument(
        "--train-start",
        default="2016-01-01",
        help="训练集样本锚定日起始（含），YYYY-MM-DD",
    )
    parser.add_argument(
        "--train-end",
        default="2025-12-31",
        help="训练集样本锚定日结束（含），YYYY-MM-DD",
    )
    parser.add_argument(
        "--val-start",
        default="2026-01-01",
        help="验证集样本锚定日起始（含），YYYY-MM-DD",
    )
    parser.add_argument(
        "--val-end",
        default="2026-02-01",
        help="验证集样本锚定日结束（含），YYYY-MM-DD",
    )
    parser.add_argument(
        "--load-end",
        default=None,
        metavar="DATE",
        help="读取 daily/ 的结束日期（含）；默认 = val-end + 45 日历日",
    )
    parser.add_argument(
        "--export-scores",
        default=None,
        metavar="CSV",
        help="训练结束后写出打分 CSV：默认为验证集；若指定 --export-infer-anchor 则仅写出该锚定日的推理打分（label_return 为空）",
    )
    parser.add_argument(
        "--export-infer-anchor",
        default=None,
        metavar="DATE",
        help=(
            "无需 T+1 行情：在面板最后交易日即可导出锚定日 pred_score（YYYY-MM-DD）。"
            "可用关键字 auto（默认用于 workflow=predict-next）：自动取面板最后交易日。"
            "须同时指定 --export-scores；特征窗口与训练一致。"
            "典型用法：日线只到 20260519 时，取 --val-* 为倒数第二个交易日作验证，"
            "本参数填面板末日用于次日决策。"
        ),
    )
    parser.add_argument(
        "--export-top-per-day",
        default=None,
        metavar="CSV",
        help="每日 pred_score 最高的 Top-N 标的（含 rank），便于对照作业「买入得分最高」策略",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="与 --export-top-per-day 配合：每个交易日保留前 N 只股票",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="单机多卡时使用 nn.DataParallel（单进程）。若已用 torchrun 多进程会自动走 DDP",
    )
    parser.add_argument(
        "--export-metrics-json",
        default=None,
        metavar="PATH",
        help="最后一个 epoch 的验证集指标写入 JSON（IC/ICIR/RankIC/方向胜率等）",
    )
    parser.add_argument(
        "--export-daily-ic-csv",
        default=None,
        metavar="PATH",
        help="导出按日 Pearson IC 时间序列 CSV（便于画图）",
    )
    parser.add_argument(
        "--min-names-per-day",
        type=int,
        default=10,
        help="计算截面 IC 时每个交易日至少需要的股票数量",
    )
    args = parser.parse_args()

    def _ymd(s: str) -> str:
        return s.strip().replace("-", "")

    if args.export_infer_anchor and not args.export_scores:
        raise ValueError("使用 --export-infer-anchor 时必须指定 --export-scores（输出路径）")

    if args.quick and args.workflow == "predict-next":
        raise ValueError("--workflow predict-next 不能与 --quick 同时使用")

    if args.workflow == "predict-next":
        if not args.export_scores:
            raise ValueError("workflow=predict-next 时必须指定 --export-scores（写出用于次日决策的打分 CSV）")
        if args.export_infer_anchor is None:
            args.export_infer_anchor = "auto"

    if not args.quick:
        if _ymd(args.train_start) > _ymd(args.train_end):
            raise ValueError("train-start 不能晚于 train-end")
        if args.workflow == "backtest":
            if _ymd(args.train_end) >= _ymd(args.val_start):
                raise ValueError(
                    "train-end 必须早于 val-start（样本锚定日的严格时间因果）；"
                    f"当前 train-end={args.train_end}, val-start={args.val_start}"
                )
            if _ymd(args.val_start) > _ymd(args.val_end):
                raise ValueError("val-start 不能晚于 val-end")

    use_ddp, rank, local_rank = ddp_setup()
    main_rank = (not use_ddp) or (rank == 0)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.quick:
        start_date = "2024-01-01"
        end_date = "2025-06-30"
        train_range = ("2024-01-01", "2024-12-31")
        val_range = ("2025-01-01", "2025-06-30")
    elif args.workflow == "predict-next":
        start_date = args.train_start
        end_date = args.load_end or default_load_end(args.train_end)
        train_range = (args.train_start, args.train_end)
        val_range = ("2999-12-31", "2999-12-31")
    else:
        start_date = args.train_start
        end_date = args.load_end or default_load_end(args.val_end)
        train_range = (args.train_start, args.train_end)
        val_range = (args.val_start, args.val_end)

    if main_rank:
        val_note = (
            f"{val_range[0]}..{val_range[1]}"
            if args.workflow != "predict-next"
            else "(跳过，workflow=predict-next)"
        )
        print(
            f"[workflow={args.workflow}] load_daily: {start_date} .. {end_date} | "
            f"train_anchor: {train_range[0]}..{train_range[1]} | "
            f"val_anchor: {val_note} | stock_pool={args.stock_pool}"
        )

    if use_ddp:
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if main_rank:
        if use_ddp:
            ws = int(os.environ.get("WORLD_SIZE", "1"))
            print(f"DDP: world_size={ws}, rank={rank}, device={device}")
        else:
            print(f"device={device}")

    processor = DataProcessor(args.data_root)
    X_train, y_train, X_val, y_val = processor.run_pipeline(
        start_date=start_date,
        end_date=end_date,
        stock_pool=args.stock_pool,
        window_len=args.window_len,
        horizon=args.horizon,
        train_range=train_range,
        val_range=val_range,
        add_ta=False,
        use_all_daily_columns=True,
        allow_empty_val=(args.workflow == "predict-next"),
    )
    if (
        args.export_infer_anchor is not None
        and str(args.export_infer_anchor).strip().lower() == "auto"
    ):
        args.export_infer_anchor = _panel_last_trade_date_str(processor)
        if main_rank:
            print(f"[infer] export-infer-anchor=auto → 使用面板最后交易日 {args.export_infer_anchor}")

    if main_rank:
        print("feature_cols:", processor.feature_cols)
        print("shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    if len(X_train) == 0:
        raise RuntimeError(
            "训练样本数为 0：请检查本地 daily 是否覆盖所选日期区间，或放宽 train-start/train-end。"
        )
    if args.workflow == "backtest" and len(X_val) == 0:
        raise RuntimeError(
            "验证样本数为 0：请检查 train/val 日期与 horizon（详见报错上方的 create_sequences 提示）。"
        )

    _, _, feat_dim = X_train.shape
    train_ds, val_ds = _to_tensors(X_train, y_train, X_val, y_val)
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TsTransformer(
        feat_dim=feat_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if main_rank:
            print(f"DataParallel: using {torch.cuda.device_count()} GPUs")

    core = unwrap_model(model)
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss(beta=0.01)

    last_bundle = None
    for epoch in range(1, args.epochs + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        train_loss = total / max(n_seen, 1)

        has_val = len(val_ds) > 0
        if main_rank:
            # DDP 下不可仅在 rank0 对包装后的 model 做 forward（会触发 NCCL 死锁）；对裸模块 core 推理。
            core.eval()
            if has_val:
                v_total = 0.0
                v_n = 0
                preds = []
                targs = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        pred = core(xb)
                        v_total += float(loss_fn(pred, yb).item()) * xb.size(0)
                        v_n += xb.size(0)
                        preds.append(pred.detach().cpu().numpy())
                        targs.append(yb.detach().cpu().numpy())
                val_loss = v_total / max(v_n, 1)
                pv = np.concatenate(preds, axis=0)
                tv = np.concatenate(targs, axis=0)
                vd = np.asarray(processor.val_dates).astype(str)
                bundle = validation_metrics_bundle(
                    pv.reshape(-1),
                    tv.reshape(-1),
                    vd,
                    min_names_per_day=args.min_names_per_day,
                )
                last_bundle = bundle
                print(
                    f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"{format_metrics_line(bundle)}"
                )
            else:
                print(
                    f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}  "
                    f"(无验证集，跳过 val_loss / IC；见训练结束后的推理打分 CSV)"
                )
        else:
            core.eval()

        if use_ddp:
            dist.barrier()

    if use_ddp:
        dist.barrier()

    if main_rank and last_bundle is not None and args.export_metrics_json:
        _ensure_parent_dir(args.export_metrics_json)
        with open(args.export_metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics_dict_for_json(last_bundle), f, indent=2, ensure_ascii=False)
        print(f"wrote metrics json: {args.export_metrics_json}")

    if main_rank and last_bundle is not None and args.export_daily_ic_csv:
        ser = last_bundle.get("_series_ic_pearson_daily")
        if ser is not None and len(ser) > 0:
            _ensure_parent_dir(args.export_daily_ic_csv)
            df_ic = ser.reset_index()
            df_ic.columns = ["trade_date", "ic_pearson_daily"]
            df_ic.to_csv(args.export_daily_ic_csv, index=False)
            print(f"wrote daily IC csv: {args.export_daily_ic_csv}")

    # -------- 打分导出（仅主进程）：验证集 或 末日推理 ----------
    if main_rank and (args.export_scores or args.export_top_per_day):
        core.eval()
        if args.export_infer_anchor:
            anchor_s = _ymd(args.export_infer_anchor)
            pinf = DataProcessor(args.data_root)
            pinf.load_data(start_date, end_date, args.stock_pool, exclude_st=True, exclude_bj=True)
            pinf.select_features(add_ta=False, use_all_daily_columns=True)
            pinf.construct_labels(horizon=args.horizon)
            X_inf, st_inf = pinf.build_inference_X_at_anchor(anchor_s, args.window_len)
            if tuple(pinf.feature_cols) != tuple(processor.feature_cols):
                raise RuntimeError(
                    "推理分支特征列与训练不一致；请检查两次 load/select 参数是否相同。"
                )
            flat = X_inf.reshape(-1, feat_dim)
            X_inf_s = processor.scaler.transform(flat).astype(np.float32).reshape(X_inf.shape)
            infer_preds: list[np.ndarray] = []
            inf_ds = TensorDataset(torch.from_numpy(X_inf_s), torch.zeros(len(X_inf_s), 1, dtype=torch.float32))
            inf_loader = DataLoader(
                inf_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for xb, _ in inf_loader:
                    xb = xb.to(device, non_blocking=True)
                    infer_preds.append(core(xb).detach().cpu().numpy())
            pv_all = np.concatenate(infer_preds, axis=0).reshape(-1)
            score_df = pd.DataFrame(
                {
                    "trade_date": anchor_s,
                    "ts_code": st_inf,
                    "pred_score": pv_all,
                    "label_return": np.nan,
                }
            )
            note = "（推理导出：无 T+1 标签，label_return 为空）"
        else:
            val_preds: list[np.ndarray] = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    val_preds.append(core(xb).detach().cpu().numpy())
            pv_all = np.concatenate(val_preds, axis=0).reshape(-1)
            vd = np.asarray(processor.val_dates).astype(str)
            vs = np.asarray(processor.val_stocks).astype(str)
            yv_all = y_val.reshape(-1).astype(np.float64)
            score_df = pd.DataFrame(
                {
                    "trade_date": vd,
                    "ts_code": vs,
                    "pred_score": pv_all,
                    "label_return": yv_all,
                }
            )
            note = ""

        if args.export_scores:
            _ensure_parent_dir(args.export_scores)
            score_df.to_csv(args.export_scores, index=False)
            print(f"wrote scores: {args.export_scores} ({len(score_df)} rows){note}")
        if args.export_top_per_day:
            _ensure_parent_dir(args.export_top_per_day)
            tops = []
            for d, g in score_df.groupby("trade_date", sort=True):
                g = g.nlargest(args.top_n, "pred_score").copy()
                g.insert(0, "rank", range(1, len(g) + 1))
                tops.append(g)
            top_df = pd.concat(tops, ignore_index=True)
            top_df.to_csv(args.export_top_per_day, index=False)
            print(
                f"wrote top-{args.top_n} per day: {args.export_top_per_day} "
                f"({len(top_df)} rows, {top_df['trade_date'].nunique()} days)"
            )

    if use_ddp:
        dist.barrier()

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
