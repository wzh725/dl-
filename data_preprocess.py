"""
data_preprocess：数据根路径、交易日历、侧车并入、日线面板与训练用样本流水线。
"""
from __future__ import annotations

import bisect
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
_DEFAULT_RELATIVE_TO_REPO = REPO_ROOT / "data"


def resolve_data_root(data_root: Optional[str]) -> str:
    """
    规范为绝对路径。优先级：传入参数 > 环境变量 DL_DATA_ROOT >
    「仓库上一级的 data/」（与克隆路径无关）。
    """
    cand = (data_root or "").strip()
    if not cand:
        cand = os.environ.get("DL_DATA_ROOT", "").strip()
    if not cand:
        cand = str(_DEFAULT_RELATIVE_TO_REPO)
    return os.path.abspath(os.path.expanduser(cand))


# --- 交易日历（基于 trade_cal.csv）---

_OPEN_DAYS_CACHE: Dict[str, List[str]] = {}


def normalize_trade_calendar_key(s: str) -> str:
    """日历比较用 8 位数字串 YYYYMMDD。"""
    y = str(s).strip().replace("-", "")[:8]
    if len(y) != 8 or not y.isdigit():
        raise ValueError(f"日期须为 YYYY-MM-DD 或 YYYYMMDD: {s!r}")
    return y


def load_open_trading_days(data_root: str) -> List[str]:
    root_key = str(Path(data_root).resolve())
    cached = _OPEN_DAYS_CACHE.get(root_key)
    if cached is not None:
        return cached
    path = Path(root_key) / "trade_cal.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少交易日历: {path}")
    cal = pd.read_csv(path)
    cal["cal_date"] = cal["cal_date"].astype(str)
    o = cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).unique()
    out = sorted(o.tolist())
    _OPEN_DAYS_CACHE[root_key] = out
    return out


def next_trading_day_strictly_after(anchor_yyyymmdd: str, data_root: str) -> str:
    """
    若某日收盘后只能使用 T 及以前的数据，则监督验证 / 回测截面的首日锚定应不早于
    「T 的下一个交易日」。本函数返回日历中严格晚于 anchor 的第一个开市日（字符串 8 位）。"""
    d = normalize_trade_calendar_key(anchor_yyyymmdd)
    days = load_open_trading_days(data_root)
    i = bisect.bisect_right(days, d)
    if i >= len(days):
        raise ValueError(
            f"在 trade_cal 中找不到严格晚于 {d} 的开市日（请检查 data-root 与日历区间）。"
        )
    return days[i]


def first_open_trading_day_on_or_after(anchor_yyyymmdd: str, data_root: str) -> Optional[str]:
    """sorted 开市日序列上首个 >= anchor 的开市日；找不到则返回 None。"""
    d = normalize_trade_calendar_key(anchor_yyyymmdd)
    days = load_open_trading_days(data_root)
    i = bisect.bisect_left(days, d)
    if i >= len(days):
        return None
    return days[i]


def fmt_yyyy_mm_dd(yyyymmdd: str) -> str:
    s = normalize_trade_calendar_key(yyyymmdd)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _slug_from_market_file(fname: str) -> str:
    base = os.path.splitext(fname)[0]
    return re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_").lower()


def merge_moneyflow_metric_market_into_panel(
    df: pd.DataFrame,
    data_root: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    df: MultiIndex (trade_date, ts_code)，与 load_data 输出一致。
    返回同源索引的扩展面板（numeric 缺失左连接后为 NaN，由上游 ffill/Nan 策略处理）。
    """
    if df is None or len(df.index) == 0:
        return df

    orig_cols = set(df.columns)
    trade_days: List[str] = sorted(
        {
            str(d).replace("-", "").replace("/", "")[:8]
            for d in df.index.get_level_values(0).unique()
        }
    )
    trade_days = [d for d in trade_days if len(d) == 8 and d.isdigit()]
    mf_dir = os.path.join(data_root, "moneyflow")
    mt_dir = os.path.join(data_root, "metric")
    mkt_dir = os.path.join(data_root, "market")

    # -------- moneyflow --------
    mf_parts: List[pd.DataFrame] = []
    for d in tqdm(trade_days, desc=".merge moneyflow", disable=not verbose):
        fp = os.path.join(mf_dir, f"{d}.csv")
        if not os.path.isfile(fp):
            continue
        m = pd.read_csv(fp)
        if "ts_code" not in m.columns:
            continue
        ren = {}
        skip = {"ts_code", "trade_date"}
        for c in m.columns:
            if c in skip:
                continue
            ren[c] = f"mf_{c}"
        m = m.rename(columns=ren)
        m["ts_code"] = m["ts_code"].astype(str)
        m["trade_date"] = d
        mf_parts.append(m)
    if mf_parts:
        mf_all = pd.concat(mf_parts, ignore_index=True)
        mf_idx = mf_all.set_index(["trade_date", "ts_code"]).sort_index()
    else:
        mf_idx = None

    # -------- metric --------
    mt_parts: List[pd.DataFrame] = []
    for d in tqdm(trade_days, desc=".merge metric", disable=not verbose):
        fp = os.path.join(mt_dir, f"{d}.csv")
        if not os.path.isfile(fp):
            continue
        m = pd.read_csv(fp)
        if "ts_code" not in m.columns:
            continue
        ren = {}
        for c in m.columns:
            if c in ("ts_code", "trade_date"):
                continue
            ren[c] = f"mtr_{c}"
        m = m.rename(columns=ren)
        m["ts_code"] = m["ts_code"].astype(str)
        m["trade_date"] = d
        mt_parts.append(m)
    if mt_parts:
        mt_all = pd.concat(mt_parts, ignore_index=True)
        mt_idx = mt_all.set_index(["trade_date", "ts_code"]).sort_index()
    else:
        mt_idx = None

    out = df
    dup_drop_msgs: List[str] = []
    if mf_idx is not None:
        dup = set(mf_idx.columns) & set(out.columns)
        if dup:
            mf_idx = mf_idx.drop(columns=list(dup), errors="ignore")
            dup_drop_msgs.append(f"moneyflow overlap drop {dup}")
        out = out.join(mf_idx, how="left")
        if verbose and dup_drop_msgs:
            print(f"[panel_sidecars] {'; '.join(dup_drop_msgs)}", flush=True)

    dup_drop_msgs = []
    if mt_idx is not None:
        dup = set(mt_idx.columns) & set(out.columns)
        if dup:
            mt_idx = mt_idx.drop(columns=list(dup), errors="ignore")
            dup_drop_msgs.append(f"metric overlap drop {dup}")
        out = out.join(mt_idx, how="left")
        if verbose and dup_drop_msgs:
            print(f"[panel_sidecars] {'; '.join(dup_drop_msgs)}", flush=True)

    # -------- market indices (broadcast) --------
    idx_frames: List[pd.DataFrame] = []
    if os.path.isdir(mkt_dir):
        for fn in sorted(os.listdir(mkt_dir)):
            if not fn.endswith(".csv"):
                continue
            fp = os.path.join(mkt_dir, fn)
            mk = pd.read_csv(fp)
            if "trade_date" not in mk.columns:
                continue
            slug = _slug_from_market_file(fn)
            mk = mk.copy()
            mk["trade_date"] = mk["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
            numeric_cols: List[str] = []
            for c in mk.columns:
                if c in ("ts_code", "trade_date"):
                    continue
                if pd.api.types.is_numeric_dtype(mk[c]):
                    numeric_cols.append(c)
            prio = ["pct_chg", "change", "close", "vol", "amount", "open", "high", "low", "pre_close"]
            ordered = [c for c in prio if c in numeric_cols] + [c for c in numeric_cols if c not in prio]
            take = ordered[:24]
            if not take:
                continue
            piece = mk[["trade_date"] + take].copy()
            ren2 = {c: f"idx_{slug}_{c}" for c in piece.columns if c != "trade_date"}
            piece = piece.rename(columns=ren2)
            idx_frames.append(piece.drop_duplicates(subset=["trade_date"]))

    if idx_frames:
        mkt_wide = idx_frames[0]
        for p in idx_frames[1:]:
            mkt_wide = mkt_wide.merge(p, on="trade_date", how="outer")
        mkt_wide["trade_date"] = mkt_wide["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
        mkt_wide = mkt_wide.drop_duplicates("trade_date").set_index("trade_date").sort_index()
        idx_cols_all = [c for c in mkt_wide.columns if str(c).startswith("idx_")]
        dates_arr = out.index.get_level_values(0).astype(str).str.replace("-", "", regex=False).str[:8]
        for c in idx_cols_all:
            ser = mkt_wide[c].reindex(dates_arr)
            out[c] = pd.to_numeric(ser, errors="coerce").to_numpy(dtype=np.float64, copy=False)

    if verbose:
        new_ix = [c for c in out.columns if c not in orig_cols]
        n_mf = sum(1 for c in new_ix if str(c).startswith("mf_"))
        n_mt = sum(1 for c in new_ix if str(c).startswith("mtr_"))
        n_ix = sum(1 for c in new_ix if str(c).startswith("idx_"))
        print(
            f"[panel_sidecars] 新增数值列共 {len(new_ix)}（mf_/mtr_/idx_ → {n_mf}/{n_mt}/{n_ix}）",
            flush=True,
        )

    return out

def panel_stock_codes(panel: pd.DataFrame) -> Set[str]:
    return set(panel.index.get_level_values("ts_code").astype(str).unique())



def _numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """返回面板中所有数值列名（稳定排序），用于保留日线全部基础字段。"""
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols)


class DataProcessor:
    def __init__(self, data_root: Optional[str] = None, *, verbose: bool = True):
        """
        Args:
            data_root: 数据根目录；空则环境变量 DL_DATA_ROOT，再退回仓库根目录下 ``data/``（见 ``data_paths.resolve_data_root``）。
            verbose: False 时关闭 tqdm 与信息类 print（供 DDP 非主进程消音）
        """
        self.data_root = resolve_data_root(data_root)
        self.verbose = bool(verbose)
        self.daily_dir = os.path.join(self.data_root, "daily")
        self.basic_path = os.path.join(self.data_root, "basic.csv")
        self.trade_cal_path = os.path.join(self.data_root, "trade_cal.csv")
        self.st_dir = os.path.join(self.data_root, "stock_st")
        
        self.df = None          # 合并后的面板数据 (date, ts_code, features)
        self.feature_cols = None
        self.label_col = None
        self.scaler = None      # 训练集拟合的标准化器
    
    def _vprint(self, *args: object, **kwargs: object) -> None:
        if self.verbose:
            print(*args, **kwargs)
    
    def _load_hs300_components(self) -> Dict[str, set]:
        """
        读取 index_weight 目录下形如 YYYYMM_000300.SH.csv 的文件，
        每个文件内部有 trade_date 列（可能只有一行或整个月的每日数据），
        返回 {trade_date: set(ts_code)} 字典，用于后续向前填充。
        """
        index_weight_dir = os.path.join(self.data_root, 'index_weight')
        if not os.path.exists(index_weight_dir):
            raise FileNotFoundError(f"index_weight directory not found: {index_weight_dir}")

        all_files = [f for f in os.listdir(index_weight_dir) 
                     if f.endswith('.csv') and '_000300.SH.csv' in f]
        if not all_files:
            raise ValueError("No HS300 weight files found in index_weight directory.")

        components = {}
        for fname in all_files:
            filepath = os.path.join(index_weight_dir, fname)
            df = pd.read_csv(filepath)
            # 确定股票代码列名
            code_col = 'con_code' if 'con_code' in df.columns else 'ts_code'
            if 'trade_date' not in df.columns or code_col not in df.columns:
                self._vprint(f"Warning: {fname} missing required columns, skipping.")
                continue
            # 将 trade_date 转换为字符串 YYYYMMDD
            df['trade_date'] = df['trade_date'].astype(str)
            # 按 trade_date 分组，每组股票代码集合
            for date, group in df.groupby('trade_date'):
                if date not in components:
                    components[date] = set()
                components[date].update(group[code_col].unique())

        self._vprint(f"Loaded HS300 components for {len(components)} distinct dates.")
        return components
        
    # ==================== 1. 数据加载与过滤 ====================
    def load_data(self, 
                  start_date: str = '2022-01-01',
                  end_date: str = '2025-12-31',
                  stock_pool: str = 'all',           # 'all', 'hs300', 'cyb', 'kcb'
                  exclude_st: bool = True,
                  exclude_bj: bool = True) -> pd.DataFrame:
        """
        加载日频数据并过滤股票池。
        stock_pool=all 时与作业 PDF 一致：除 ST、北交所以外的全部 A 股（以 basic.csv 登记的非北交所板块为准），
        再与本地 daily 行取交集；实际只数取决于本地行情覆盖。
        Returns:
            DataFrame: MultiIndex (trade_date, ts_code) 包含所有量价特征
        """
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(
                "数据根目录不存在或不可访问。\n"
                f"  解析后路径: {self.data_root}\n"
                "  请检查 `--data-root`、`DL_DATA_ROOT` 是否为空或指向错误。"
                "\n示例: `--data-root /home/lhr/my_stuff/fundamentals_for_deep_learning/data`"
            )
        if not os.path.isfile(self.trade_cal_path):
            raise FileNotFoundError(
                "找不到交易日历（应在数据根目录下）。\n"
                f"  期望文件: {self.trade_cal_path}\n"
                f"  当前 data_root: {self.data_root}\n"
                "若你曾经把 `DL_DATA_ROOT=` 设为**空**，会导致 join 后对 `trade_cal.csv` 的解析落在错误的工作目录；"
                "请取消空导出或使用绝对路径。"
            )
        # 获取交易日历
        cal = pd.read_csv(self.trade_cal_path)
        # 统一日期列为字符串格式
        cal['cal_date'] = cal['cal_date'].astype(str)
        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')
        cal = cal[(cal['is_open'] == 1) & (cal['cal_date'] >= start_str) & (cal['cal_date'] <= end_str)]
        trade_dates = sorted(cal['cal_date'].unique())
        
        # 逐日读取daily文件
        all_dfs = []
        for date in tqdm(trade_dates, desc="Loading daily data", disable=not self.verbose):
            file_path = os.path.join(self.daily_dir, f"{date}.csv")
            if not os.path.exists(file_path):
                continue
            df_day = pd.read_csv(file_path)
            df_day['trade_date'] = date
            all_dfs.append(df_day)
        
        if not all_dfs:
            raise ValueError("No daily data loaded. Check data path and date range.")
        
        df = pd.concat(all_dfs, ignore_index=True)

        basic_df: Optional[pd.DataFrame] = None

        def _get_basic() -> pd.DataFrame:
            nonlocal basic_df
            if basic_df is None:
                basic_df = pd.read_csv(self.basic_path)
            return basic_df

        # 过滤ST股（stock_st 文件合并列表；偏保守，等价于剔除「曾出现在 ST 列表」的代码）
        if exclude_st:
            st_files = [f for f in os.listdir(self.st_dir) if f.endswith('.csv')]
            st_codes = set()
            for st_file in st_files:
                st_df = pd.read_csv(os.path.join(self.st_dir, st_file))
                st_codes.update(st_df['ts_code'].unique())
            df = df[~df['ts_code'].isin(st_codes)]
        
        # 过滤北交所（市场类型为北交所的股票）
        if exclude_bj:
            basic = _get_basic()
            bj_codes = basic[basic['market'] == '北交所']['ts_code'].unique()
            df = df[~df['ts_code'].isin(bj_codes)]
        
        # 限定股票池
        if stock_pool == 'hs300':
            components_dict = self._load_hs300_components()
            if not components_dict:
                raise ValueError("No HS300 component data loaded.")
    
            # 获取所有交易日期（df 中的 trade_date 列，格式为 YYYYMMDD）
            all_dates = sorted(df['trade_date'].unique())
            comp_dates = sorted(components_dict.keys())
    
            # 构建每个日期应使用的成分股日期（向前填充）
            date_to_comp_date = {}
            comp_idx = 0
            for d in all_dates:
                # 找到第一个大于等于 d 的成分股日期
                while comp_idx < len(comp_dates) and comp_dates[comp_idx] < d:
                    comp_idx += 1
                # 如果所有成分股日期都小于 d，则使用最后一个
                if comp_idx >= len(comp_dates):
                    use_date = comp_dates[-1]
                else:
                    # 如果 comp_dates[comp_idx] == d，直接使用
                    if comp_dates[comp_idx] == d:
                        use_date = comp_dates[comp_idx]
                    else:  # comp_dates[comp_idx] > d，使用前一个（如果存在）
                        if comp_idx > 0:
                            use_date = comp_dates[comp_idx - 1]
                        else:
                            # 没有更早的成分股日期，跳过该日（或使用最早的）
                            use_date = None
                if use_date is not None:
                    date_to_comp_date[d] = use_date
    
            # 构建成分股 DataFrame
            comp_list = []
            for date, comp_date in date_to_comp_date.items():
                for code in components_dict[comp_date]:
                    comp_list.append({'trade_date': date, 'ts_code': code})
            comp_df = pd.DataFrame(comp_list)
    
            # 内连接过滤
            original_len = len(df)
            df = df.merge(comp_df, on=['trade_date', 'ts_code'], how='inner')
            self._vprint(f"Filtered to HS300 constituents: {original_len} -> {len(df)} rows")
        elif stock_pool == 'cyb':
            basic = _get_basic()
            pool_codes = basic[basic['market'] == '创业板']['ts_code'].unique()
            df = df[df['ts_code'].isin(pool_codes)]
        elif stock_pool == 'kcb':
            basic = _get_basic()
            pool_codes = basic[basic['market'] == '科创板']['ts_code'].unique()
            df = df[df['ts_code'].isin(pool_codes)]
        elif stock_pool == 'all':
            # 作业 PDF：除 ST、北交所外的所有 A 股上市股票（约 5000）；用 basic 中非北交所板块界定上市范围
            basic = _get_basic()
            a_share_codes = set(basic.loc[basic['market'] != '北交所', 'ts_code'].astype(str))
            before = len(df)
            df = df[df['ts_code'].astype(str).isin(a_share_codes)]
            self._vprint(
                "Stock pool: all A-shares (作业比赛范围: basic 主板/创业板/科创板, ∩ daily 行; "
                f"ST/北交所已按配置过滤) rows {before} -> {len(df)}, n_codes_basic={len(a_share_codes)}"
            )
        else:
            raise ValueError(
                f"Unknown stock_pool={stock_pool!r}; expected 'all', 'hs300', 'cyb', or 'kcb'."
            )

        # 日历 end_date 只是上限；缺失的 daily/*.csv 会被跳过，故「实际最后交易日」常早于日历末日
        if len(df) > 0:
            loaded_days = sorted(df["trade_date"].astype(str).unique())
            self._vprint(
                f"[daily] 日历区间内开市日约 {len(trade_dates)} 天；本地实际读到 {len(loaded_days)} 天；"
                f"面板最后交易日: {loaded_days[-1]}（缺 CSV 的日期不会进入面板）"
            )

        # 设置多重索引
        df = df.set_index(['trade_date', 'ts_code']).sort_index()
        self.df = df
        self._vprint(f"Data loaded: {df.index.levshape[0]} dates, {df.index.levshape[1]} stocks")
        return df
    
    # ==================== 2. 特征工程 ====================
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加常见技术指标（MACD, RSI, 成交量移动平均等）
        """
        def _macd(close, fast=12, slow=26, signal=9):
            exp1 = close.ewm(span=fast, adjust=False).mean()
            exp2 = close.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist

        def _rsi(close, window=14):
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df = df.copy()
        grouped = df.groupby('ts_code')
    
        # 移动平均线（可以用 transform）
        df['ma5'] = grouped['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['ma10'] = grouped['close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
        df['ma20'] = grouped['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        df['vol_ma5'] = grouped['vol'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['rsi14'] = grouped['close'].transform(lambda x: _rsi(x, 14))
    
        def compute_macd(grp):
            macd, macd_signal, macd_hist = _macd(grp)
            return pd.DataFrame({'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist})
    
        macd_df = grouped['close'].apply(compute_macd)
        # reset_index 以便合并，注意原索引会变成列
        macd_df = macd_df.reset_index(level=0, drop=True)  # 丢弃原来的股票索引，保留原 df 索引
        df = df.join(macd_df)
    
        # 波动率
        df['volatility'] = grouped['pct_chg'].transform(lambda x: x.rolling(20, min_periods=5).std())
        # 成交量比率
        df['volume_ratio'] = df['vol'] / df['vol_ma5']
    
        return df
    
    def select_features(self, 
                        base_features: Optional[List[str]] = None,
                        add_ta: bool = True,
                        use_all_daily_columns: bool = False) -> List[str]:
        """
        确定最终特征列
        Args:
            base_features: 基础量价特征；若 use_all_daily_columns=True 则忽略此项
            add_ta: 是否添加技术指标（在基础列之上追加）
            use_all_daily_columns: True 时保留 load_data 后所有数值型日线字段（如 open…vwap 全列）
        Returns:
            特征列名列表
        """
        if self.df is None:
            raise ValueError("Load data first.")

        if use_all_daily_columns:
            base_features = _numeric_feature_columns(self.df)
            if 'close' not in base_features:
                raise ValueError("数据中缺少 close 列，无法构造标签。")
        elif base_features is None:
            base_features = ['open', 'high', 'low', 'close', 'vol', 'pct_chg', 'vwap']
        
        miss = [c for c in base_features if c not in self.df.columns]
        if miss:
            raise ValueError(f"以下特征列不存在于数据中: {miss}")

        self.df = self.df[base_features].copy()  # 先只保留基础列
        
        if add_ta:
            self.df = self.add_technical_indicators(self.df)
            # 自动获取所有数值列作为特征（排除标签用列）
            all_cols = self.df.columns.tolist()
            # 移除明显不是特征的列（如已存在的标签列，但此时还未构造）
            self.feature_cols = [c for c in all_cols if c not in ['ts_code', 'trade_date']]
        else:
            self.feature_cols = base_features
        
        # 处理无限值/缺失值（先不做填充，留给后续步骤）
        return self.feature_cols
    
    # ==================== 3. 构造标签 ====================
    def construct_labels(self, horizon: int = 1, label_type: str = 'return'):
        """
        构造未来 horizon 日的收益率标签
        Args:
            horizon: 预测未来几天收益，例如1表示T+1收益
            label_type: 'return' 或 'direction' (二分类)
        Returns:
            添加 label_col 的 DataFrame
        """
        if self.df is None:
            raise ValueError("Load data first.")
        
        # 计算未来 horizon 日收益率: (未来收盘价 - 当前收盘价) / 当前收盘价
        # 按股票分组，shift(-horizon) 取未来收盘价
        df = self.df.copy()
        df['future_close'] = df.groupby('ts_code')['close'].shift(-horizon)
        df['label_return'] = (df['future_close'] - df['close']) / df['close']
        
        if label_type == 'direction':
            # 二分类: 1表示上涨，0表示下跌或平
            df['label'] = (df['label_return'] > 0).astype(int)
            self.label_col = 'label'
        else:
            # 回归任务：直接使用收益率
            self.label_col = 'label_return'
        
        # 删除未来信息辅助列
        df.drop('future_close', axis=1, inplace=True)
        self.df = df
        self._vprint(f"Label constructed: {self.label_col} (horizon={horizon})")
        return self.df
    
    # ==================== 4. 滑动窗口与样本构造 ====================
    def create_sequences(self,
                         window_len: int = 20,
                         horizon: int = 1,
                         train_date_range: Tuple[str, str] = ('2019-01-01', '2024-12-31'),
                         val_date_range: Tuple[str, str] = ('2025-01-01', '2025-12-31'),
                         allow_empty_val: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        为每只股票生成滑动窗口样本，并按时间划分训练/验证集
        Args:
            window_len: 输入序列长度（天数）
            horizon: 预测步长（必须与construct_labels中的horizon一致）
            train_date_range: 训练集时间范围 (start, end)
            val_date_range: 验证集时间范围 (start, end)
            min_samples_per_stock: 每只股票至少产生多少个样本，否则丢弃该股票
            allow_empty_val: True 时允许验证集为空（用于「仅用全历史监督样本训练 + 末日推理」流程）
        Returns:
            X_train, y_train, X_val, y_val (numpy arrays)
        """
        if self.label_col not in self.df.columns:
            raise ValueError("Labels not constructed. Call construct_labels() first.")
        
        # 确保特征列存在，去除 label 列和索引列
        feature_data = self.df[self.feature_cols].copy()
        # 处理缺失值：仅在每只股票时间序列内前向填充（避免 bfill 引入未来信息）
        feature_data = feature_data.groupby('ts_code').transform(lambda x: x.ffill())
        # 删除仍然含有 NaN 的行（一般是股票上市初期或退市边缘）
        valid_idx = feature_data.notna().all(axis=1)
        self.df = self.df[valid_idx]
        feature_data = feature_data[valid_idx]
        
        # 按日期和股票重新整理
        df_clean = self.df[[self.label_col]].copy()
        df_clean = df_clean.loc[valid_idx]
        df_clean = pd.concat([df_clean, feature_data], axis=1)
        
        # 按股票分组生成序列（向量化滑窗，避免 Python 内层逐 i 循环）
        X_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []
        date_chunks: List[np.ndarray] = []
        label_end_chunks: List[np.ndarray] = []
        stock_chunks: List[np.ndarray] = []
        
        grouped = df_clean.groupby('ts_code')
        for stock, grp in tqdm(grouped, desc="Creating sequences", disable=not self.verbose):
            grp = grp.sort_index(level='trade_date')  # 按时间升序
            dates = grp.index.get_level_values('trade_date').unique()
            values = grp[self.feature_cols].values
            labels = grp[self.label_col].values
            
            # 标签在行 j 上对应未来 horizon 日收益，需 j <= len(dates)-1-horizon，故
            # i + window_len <= len(dates) - 1 - horizon  →  i < len(dates) - window_len - horizon
            max_i = len(dates) - window_len - horizon
            if max_i <= 0:
                continue
            # windows.shape = (len(dates)-window_len+1, window_len, feat_dim)
            windows = np.lib.stride_tricks.sliding_window_view(
                values, window_shape=window_len, axis=0
            )
            # sliding_window_view(axis=0) 返回 (n, feat_dim, window_len)，转为 (n, window_len, feat_dim)
            windows = np.swapaxes(windows, 1, 2)
            n_use = max_i
            X_stock = windows[:n_use].astype(np.float32, copy=False)
            idx = np.arange(window_len, window_len + n_use, dtype=np.int64)
            y_stock = labels[idx].astype(np.float32, copy=False)
            d_stock = np.asarray(dates[idx], dtype=object)
            d_end_stock = np.asarray(dates[idx + horizon], dtype=object)
            s_stock = np.full(shape=(n_use,), fill_value=str(stock), dtype=object)

            X_chunks.append(X_stock)
            y_chunks.append(y_stock)
            date_chunks.append(d_stock)
            label_end_chunks.append(d_end_stock)
            stock_chunks.append(s_stock)

        if len(X_chunks) == 0:
            raise ValueError("No sequences generated. Check window_len/horizon and data length.")
        
        # 转换为数组
        X = np.concatenate(X_chunks, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(y_chunks, axis=0).astype(np.float32, copy=False).reshape(-1, 1)
        dates_arr = np.concatenate(date_chunks, axis=0)
        label_end_arr = np.concatenate(label_end_chunks, axis=0)
        stocks_arr = np.concatenate(stock_chunks, axis=0)
        
        # 按时间划分（禁止打乱）
        # 将传入的日期范围字符串转换为 YYYYMMDD 格式
        def to_yyyymmdd(d):
            if isinstance(d, str) and '-' in d:
                return d.replace('-', '')
            return str(d)

        def norm_anchor(d) -> str:
            """样本锚定日统一为 8 位数字字符串，避免与 mask 边界比较时 dtype 不一致。"""
            s = to_yyyymmdd(str(d))
            if len(s) >= 8:
                s = s[:8]
            if len(s) != 8 or not s.isdigit():
                raise ValueError(f"无法解析样本锚定日: {d!r}")
            return s

        train_start = to_yyyymmdd(train_date_range[0])
        train_end   = to_yyyymmdd(train_date_range[1])
        val_start   = to_yyyymmdd(val_date_range[0])
        val_end     = to_yyyymmdd(val_date_range[1])

        dates_norm = np.array([norm_anchor(d) for d in dates_arr])
        label_end_norm = np.array([norm_anchor(d) for d in label_end_arr])

        train_mask_raw = (dates_norm >= train_start) & (dates_norm <= train_end)
        # 防止 horizon 标签跨入验证期（purge）：训练样本标签结束日必须严格早于 val_start。
        train_mask = train_mask_raw & (label_end_norm < val_start)
        val_mask   = (dates_norm >= val_start) & (dates_norm <= val_end)

        dropped_by_purge = int(train_mask_raw.sum() - train_mask.sum())
        if dropped_by_purge > 0:
            self._vprint(
                f"[purge] drop train samples crossing val_start={val_start}: {dropped_by_purge}"
            )
        
        X_train, y_train = X[train_mask], y[train_mask]

        if val_mask.sum() == 0:
            uniq = sorted(set(dates_norm.tolist()))
            latest = uniq[-1] if uniq else "（无）"
            if not allow_empty_val:
                raise ValueError(
                    "验证集样本数为 0。"
                    f" 请求的 val_range=({val_date_range[0]}, {val_date_range[1]}) → [{val_start}, {val_end}]。"
                    f"\n当前 horizon={horizon} 时，样本锚定日 T 需要未来 {horizon} 个交易日的收盘价计算标签，"
                    "因此面板中**实际会出现的最晚锚定日**通常比 `daily/` **全局最后一个交易日**早。"
                    f"\n本次数据中，构造出的最晚锚定日为 **{latest}**。"
                    "\n若你想以某日 D 作为验证/导出 `pred_score` 的 trade_date，请确保 "
                    "`--load-end` / `daily/` 已覆盖至 **D 之后至少 horizon 个交易日** "
                    "（例如 horizon=1 时需存在 **D 的下一交易日** 的日线 CSV）。"
                    "\n若是「末日推理」训练流程，请使用 train.py 的 `--workflow predict-next`。"
                )
            X_val = np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32)
            y_val = np.empty((0, 1), dtype=np.float32)
            self.val_dates = np.array([], dtype=str)
            self.val_stocks = np.array([], dtype=str)
        else:
            X_val, y_val = X[val_mask], y[val_mask]
            self.val_dates = dates_norm[val_mask]
            self.val_stocks = stocks_arr[val_mask]

        self._vprint(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

        # 保存样本对应的日期和股票，便于回测
        self.train_dates = dates_norm[train_mask]
        self.train_stocks = stocks_arr[train_mask]
        self.train_label_end_dates = label_end_norm[train_mask]
        self.val_label_end_dates = label_end_norm[val_mask] if val_mask.sum() > 0 else np.array([], dtype=str)

        return X_train, y_train, X_val, y_val
    
    def build_inference_X_at_anchor(self, anchor_date: str, window_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        在面板已有截至 anchor 日（含）的行情时，构造「锚定日为 anchor」的输入矩阵，
        **不要求** anchor 日之后仍有日线（无需 T+1 收盘价即可算标签）。
        窗口划分与 create_sequences 一致：特征为 dates[j-window_len:j]，导出时的 trade_date 记为 anchor。
        """
        if self.label_col not in self.df.columns:
            raise ValueError("Labels not constructed. Call construct_labels() first.")
        anchor_s = str(anchor_date).replace("-", "")[:8]
        if len(anchor_s) != 8 or not anchor_s.isdigit():
            raise ValueError(f"无效的推理锚定日: {anchor_date!r}，应为 YYYYMMDD 或 YYYY-MM-DD")

        feature_data = self.df[self.feature_cols].copy()
        feature_data = feature_data.groupby("ts_code").transform(lambda x: x.ffill())
        valid_idx = feature_data.notna().all(axis=1)
        df_sub = self.df.loc[valid_idx]
        feature_data = feature_data.loc[valid_idx]
        df_clean = df_sub[[self.label_col]].copy()
        df_clean = pd.concat([df_clean, feature_data], axis=1)

        def norm_day(d) -> str:
            s = str(d).replace("-", "")[:8]
            if len(s) != 8 or not s.isdigit():
                raise ValueError(f"无法解析交易日: {d!r}")
            return s

        X_list: List[np.ndarray] = []
        codes: List[str] = []

        grouped = df_clean.groupby("ts_code")
        for stock, grp in grouped:
            grp = grp.sort_index(level="trade_date")
            dates = grp.index.get_level_values("trade_date").unique()
            values = grp[self.feature_cols].values
            j = None
            for idx in range(len(dates)):
                if norm_day(dates[idx]) == anchor_s:
                    j = idx
                    break
            if j is None:
                continue
            if j < window_len:
                continue
            X_seq = values[j - window_len : j].astype(np.float32)
            if not np.isfinite(X_seq).all():
                continue
            X_list.append(X_seq)
            codes.append(str(stock))

        if not X_list:
            raise ValueError(
                f"推理锚定日 {anchor_s} 无可用样本：请确认该日在面板内、且每只股票历史长度 ≥ window_len={window_len}。"
            )
        return np.stack(X_list, axis=0), np.asarray(codes, dtype=str)

    def build_inference_X_for_next_trade(self, asof_date: str, window_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        构造用于「下一交易日」决策的输入矩阵：
        - asof_date 为最新可用行情日（通常是盘后）
        - 输入窗口**包含** asof_date 当天，即 [asof-window_len+1, asof]
        """
        if self.label_col not in self.df.columns:
            raise ValueError("Labels not constructed. Call construct_labels() first.")
        asof_s = str(asof_date).replace("-", "")[:8]
        if len(asof_s) != 8 or not asof_s.isdigit():
            raise ValueError(f"无效 asof_date: {asof_date!r}，应为 YYYYMMDD 或 YYYY-MM-DD")

        feature_data = self.df[self.feature_cols].copy()
        feature_data = feature_data.groupby("ts_code").transform(lambda x: x.ffill())
        valid_idx = feature_data.notna().all(axis=1)
        df_sub = self.df.loc[valid_idx]
        feature_data = feature_data.loc[valid_idx]
        df_clean = df_sub[[self.label_col]].copy()
        df_clean = pd.concat([df_clean, feature_data], axis=1)

        def norm_day(d) -> str:
            s = str(d).replace("-", "")[:8]
            if len(s) != 8 or not s.isdigit():
                raise ValueError(f"无法解析交易日: {d!r}")
            return s

        X_list: List[np.ndarray] = []
        codes: List[str] = []
        grouped = df_clean.groupby("ts_code")
        for stock, grp in grouped:
            grp = grp.sort_index(level="trade_date")
            dates = grp.index.get_level_values("trade_date").unique()
            values = grp[self.feature_cols].values
            j = None
            for idx in range(len(dates)):
                if norm_day(dates[idx]) == asof_s:
                    j = idx
                    break
            if j is None:
                continue
            if j < (window_len - 1):
                continue
            X_seq = values[j - window_len + 1 : j + 1].astype(np.float32)
            if not np.isfinite(X_seq).all():
                continue
            X_list.append(X_seq)
            codes.append(str(stock))

        if not X_list:
            raise ValueError(
                f"推理 asof_date={asof_s} 无可用样本：请确认该日在面板内、且每只股票历史长度 ≥ window_len={window_len}。"
            )
        return np.stack(X_list, axis=0), np.asarray(codes, dtype=str)

    # ==================== 5. 标准化（避免未来信息） ====================
    def fit_standardize(self, X_train: np.ndarray, X_val: np.ndarray = None):
        """
        仅用 **训练集锚定样本**拟合标准化器（将 (N,T,F) 展平为训练时刻上的截面后 fit）。
        **不得**在未按时间分割的全量(train+验证)或未截断的未来数据上估计均值与方差，否则等价于混入未来截面信息。

        验证集 / 末日推理仅用训练阶段 fit 完的 scaler.transform。

        Args:
            X_train: 训练集 (N, window_len, feat_dim)
            X_val: 验证集 (M, window_len, feat_dim)
        Returns:
            标准化后的 X_train_scaled, X_val_scaled
        """
        # 将三维数组展开为 (N * window_len, feat_dim) 进行拟合
        N, T, F = X_train.shape
        train_flat = X_train.reshape(-1, F)
        self.scaler = StandardScaler()
        self.scaler.fit(train_flat)
        
        # 转换训练集
        X_train_scaled = self.scaler.transform(train_flat).reshape(N, T, F)
        X_val_scaled = None
        if X_val is not None:
            if X_val.shape[0] == 0:
                X_val_scaled = np.empty((0, T, F), dtype=np.float32)
            else:
                M, T2, F2 = X_val.shape
                val_flat = X_val.reshape(-1, F2)
                X_val_scaled = self.scaler.transform(val_flat).reshape(M, T2, F2)
        
        return X_train_scaled, X_val_scaled
    
    # ==================== 6. 完整流程（一站式调用） ====================
    def run_pipeline(self,
                     start_date='2022-01-01',
                     end_date='2025-12-31',
                     stock_pool='all',
                     window_len=60,
                     horizon=3,
                     train_range=('2022-01-01', '2024-12-31'),
                     val_range=('2025-01-01', '2025-12-31'),
                     add_ta: bool = True,
                     use_all_daily_columns: bool = False,
                     allow_empty_val: bool = False,
                     use_data_moneyflow_metric_index: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        执行完整数据处理流程
        Returns:
            X_train_scaled, y_train, X_val_scaled, y_val

        ``use_data_moneyflow_metric_index``：读入 ``moneyflow/metric/market`` 并入面板（本文件内 ``merge_moneyflow_metric_market_into_panel``）。
        训练数据范围与 ``../data/README.md`` 一致：日线、交易日历、basic、ST 过滤、側车三表与指数；不包含新闻或外部长表。
        """
        self.load_data(start_date, end_date, stock_pool, exclude_st=True, exclude_bj=True)

        if use_data_moneyflow_metric_index:
            self.df = merge_moneyflow_metric_market_into_panel(
                self.df, self.data_root, verbose=self.verbose
            )

        self.select_features(add_ta=add_ta, use_all_daily_columns=use_all_daily_columns)
        # 3. 构造标签
        self.construct_labels(horizon=horizon, label_type='return')
        # 4. 生成序列
        X_train, y_train, X_val, y_val = self.create_sequences(
            window_len=window_len,
            horizon=horizon,
            train_date_range=train_range,
            val_date_range=val_range,
            allow_empty_val=allow_empty_val,
        )
        # 5. 标准化
        X_train_scaled, X_val_scaled = self.fit_standardize(X_train, X_val)
        
        # 保存一些元数据供外部使用
        self.input_shape = (window_len, len(self.feature_cols))
        return X_train_scaled, y_train, X_val_scaled, y_val


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="全市场（默认）或子池 + 日线数据处理示例")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DL_DATA_ROOT", ""),
        help="数据根目录（含 daily/, index_weight/, basic.csv 等）",
    )
    args = parser.parse_args()
    args.data_root = resolve_data_root(args.data_root)

    processor = DataProcessor(args.data_root)
    X_train, y_train, X_val, y_val = processor.run_pipeline(
        start_date="2022-01-01",
        end_date="2025-12-31",
        stock_pool="all",
        window_len=60,
        horizon=3,
        train_range=("2022-01-01", "2024-12-31"),
        val_range=("2025-01-01", "2025-12-31"),
        add_ta=False,
        use_all_daily_columns=True,
    )
    print("feature_cols:", processor.feature_cols)
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
