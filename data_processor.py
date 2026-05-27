import os
import pandas as pd
import numpy as np
import bisect

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class DataProcessor:

    def __init__(self, data_root):

        self.data_root = data_root

        self.daily_dir = os.path.join(
            data_root,
            'daily'
        )

        self.basic_path = os.path.join(
            data_root,
            'basic.csv'
        )

        self.trade_cal_path = os.path.join(
            data_root,
            'trade_cal.csv'
        )

        self.st_dir = os.path.join(
            data_root,
            'stock_st'
        )

        # 新增数据路径（文件夹形式）
        self.fundamental_dir = os.path.join(data_root, 'metric')
        self.moneyflow_dir = os.path.join(data_root, 'moneyflow')

        self.df = None
        self.df_fundamental = None
        self.df_moneyflow = None

        self.feature_cols = None

        self.label_cols = None

        self.scaler = None

    # =========================================================
    # HS300 成分股
    # =========================================================
    def _load_hs300_components(self):

        index_weight_dir = os.path.join(
            self.data_root,
            'index_weight'
        )

        all_files = [
            f for f in os.listdir(index_weight_dir)
            if '_000300.SH.csv' in f
        ]

        components = {}

        for fname in all_files:

            filepath = os.path.join(
                index_weight_dir,
                fname
            )

            df = pd.read_csv(filepath)

            code_col = (
                'con_code'
                if 'con_code' in df.columns
                else 'ts_code'
            )

            df['trade_date'] = df[
                'trade_date'
            ].astype(str)

            for date, group in df.groupby(
                'trade_date'
            ):
                # 处理日期格式：确保是8位日期格式
                # 成分股文件中的日期可能是6位（年月）或8位（年月日）
                if len(date) == 6:
                    # 如果是6位年月格式，补全为当月最后一天
                    # 这里简单处理为当月第一天，后续通过bisect查找最近日期
                    date = date + '01'
                
                if date not in components:
                    components[date] = set()

                components[date].update(
                    group[code_col].unique()
                )

        print(
            f"Loaded HS300 components for "
            f"{len(components)} dates"
        )

        return components

    # =========================================================
    # 加载资金流向数据
    # =========================================================
    def load_moneyflow_data(self):
        """加载资金流向数据（文件夹形式）"""
        if not os.path.exists(self.moneyflow_dir):
            print(f"警告：未找到资金流向数据目录 {self.moneyflow_dir}")
            self.df_moneyflow = None
            return

        all_files = [f for f in os.listdir(self.moneyflow_dir) if f.endswith('.csv')]
        if not all_files:
            print("警告：资金流向目录为空")
            self.df_moneyflow = None
            return

        # 读取所有资金流向文件并合并
        dfs = []
        for fname in tqdm(all_files, desc="Loading moneyflow data"):
            filepath = os.path.join(self.moneyflow_dir, fname)
            df_day = pd.read_csv(filepath)
            dfs.append(df_day)

        self.df_moneyflow = pd.concat(dfs, ignore_index=True)
        self.df_moneyflow['trade_date'] = self.df_moneyflow['trade_date'].astype(str)
        self.df_moneyflow = self.df_moneyflow.sort_values(['ts_code', 'trade_date'])

        print(f"加载资金流向数据: {len(self.df_moneyflow)} 条记录")

    # =========================================================
    # 加载基本面数据
    # =========================================================
    def load_fundamental_data(self):
        """加载基本面数据（文件夹形式）"""
        if not os.path.exists(self.fundamental_dir):
            print(f"警告：未找到基本面数据目录 {self.fundamental_dir}")
            self.df_fundamental = None
            return

        all_files = [f for f in os.listdir(self.fundamental_dir) if f.endswith('.csv')]
        if not all_files:
            print("警告：基本面目录为空")
            self.df_fundamental = None
            return

        # 读取所有基本面文件并合并
        dfs = []
        for fname in tqdm(all_files, desc="Loading fundamental data"):
            filepath = os.path.join(self.fundamental_dir, fname)
            df_day = pd.read_csv(filepath)
            dfs.append(df_day)

        self.df_fundamental = pd.concat(dfs, ignore_index=True)
        self.df_fundamental['trade_date'] = self.df_fundamental['trade_date'].astype(str)
        if 'ann_date' in self.df_fundamental.columns:
            self.df_fundamental['ann_date'] = self.df_fundamental['ann_date'].astype(str)
        self.df_fundamental = self.df_fundamental.sort_values(['ts_code', 'trade_date'])

        print(f"加载基本面数据: {len(self.df_fundamental)} 条记录")

    # =========================================================
    # 加载数据
    # =========================================================
    def load_data(
        self,
        start_date='2024-06-01',
        end_date='2025-06-30',
        stock_pool='hs300',
        exclude_st=True,
        exclude_bj=True
    ):

        cal = pd.read_csv(
            self.trade_cal_path
        )

        cal['cal_date'] = cal[
            'cal_date'
        ].astype(str)

        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')

        cal = cal[
            (cal['is_open'] == 1)
            &
            (cal['cal_date'] >= start_str)
            &
            (cal['cal_date'] <= end_str)
        ]

        trade_dates = sorted(
            cal['cal_date'].unique()
        )

        all_dfs = []

        for date in tqdm(
            trade_dates,
            desc="Loading daily data"
        ):

            file_path = os.path.join(
                self.daily_dir,
                f"{date}.csv"
            )

            if not os.path.exists(file_path):
                continue

            df_day = pd.read_csv(file_path)

            df_day['trade_date'] = date

            all_dfs.append(df_day)

        df = pd.concat(
            all_dfs,
            ignore_index=True
        )

        # =====================================================
        # ST
        # =====================================================
        if exclude_st:

            st_files = [
                f for f in os.listdir(self.st_dir)
                if f.endswith('.csv')
            ]

            st_codes = set()

            for st_file in st_files:

                st_df = pd.read_csv(
                    os.path.join(
                        self.st_dir,
                        st_file
                    )
                )

                st_codes.update(
                    st_df['ts_code'].unique()
                )

            df = df[
                ~df['ts_code'].isin(st_codes)
            ]

        # =====================================================
        # 北交所
        # =====================================================
        if exclude_bj:

            basic = pd.read_csv(
                self.basic_path
            )

            bj_codes = basic[
                basic['market'] == '北交所'
            ]['ts_code'].unique()

            df = df[
                ~df['ts_code'].isin(bj_codes)
            ]

        # =====================================================
        # 加载行业代码（用于后续行业中性化）
        # =====================================================
        basic = pd.read_csv(self.basic_path)
        self.industry_map = dict(zip(basic['ts_code'], basic.get('industry_code', 'unknown')))
        df['industry_code'] = df['ts_code'].map(self.industry_map).fillna('unknown')

        # =====================================================
        # 过滤极端交易日：停牌、一字涨停/跌停
        # =====================================================
        df = self._filter_extreme_days(df)

        # =====================================================
        # 股票池
        # =====================================================
        if stock_pool == 'hs300':
            components = self._load_hs300_components()
            
            # 获取所有成分股日期并排序
            component_dates = sorted(components.keys())
            
            def filter_fn(row):
                date = row['trade_date']
                code = row['ts_code']
                
                # 如果当天有成分股数据，直接使用
                if date in components:
                    return code in components[date]
                
                # 否则找到最近的成分股日期（向前查找）
                idx = bisect.bisect_right(component_dates, date) - 1
                if idx >= 0:
                    nearest_date = component_dates[idx]
                    return code in components[nearest_date]
                
                return False

            mask = df.apply(filter_fn, axis=1)

            print(
                f"Filtered HS300: "
                f"{len(df)} -> {mask.sum()}"
            )

            df = df[mask]

        self.df = df.set_index(
            ['trade_date', 'ts_code']
        ).sort_index()

        print(
            f"Data loaded: "
            f"{self.df.index.levshape[0]} dates, "
            f"{self.df.index.levshape[1]} stocks"
        )

        return self.df

    # =========================================================
    # 技术指标与特征工程
    # =========================================================
    def add_technical_indicators(self, df):

        grouped = df.groupby('ts_code', group_keys=False)

        # ========== 基础移动平均线 ==========
        df['ma5'] = grouped['close'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['ma10'] = grouped['close'].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
        df['ma20'] = grouped['close'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        df['ma60'] = grouped['close'].transform(
            lambda x: x.rolling(60, min_periods=1).mean()
        )

        # ========== MACD指标 ==========
        # 计算EMA12和EMA26
        df['ema12'] = grouped['close'].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        df['ema26'] = grouped['close'].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
        # DIF = EMA12 - EMA26
        df['dif'] = df['ema12'] - df['ema26']
        # DEA = DIF的9日EMA
        df['dea'] = grouped['dif'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        # MACD = 2 * (DIF - DEA)
        df['macd'] = 2 * (df['dif'] - df['dea'])

        # ========== RSI指标（14日） ==========
        df['pct_change'] = grouped['close'].transform(
            lambda x: x.pct_change()
        )
        df['gain'] = df['pct_change'].apply(lambda x: max(x, 0) if not pd.isna(x) else 0)
        df['loss'] = df['pct_change'].apply(lambda x: abs(min(x, 0)) if not pd.isna(x) else 0)
        
        avg_gain = grouped['gain'].transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )
        avg_loss = grouped['loss'].transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )
        
        df['rsi'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

        # ========== 动量指标 ==========
        df['momentum_5'] = grouped['close'].transform(
            lambda x: x.pct_change(5)
        )
        df['momentum_10'] = grouped['close'].transform(
            lambda x: x.pct_change(10)
        )
        df['momentum_20'] = grouped['close'].transform(
            lambda x: x.pct_change(20)
        )
        df['momentum_60'] = grouped['close'].transform(
            lambda x: x.pct_change(60)
        )

        # ========== 成交量相关 ==========
        df['vol_ma5'] = grouped['vol'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['vol_ma10'] = grouped['vol'].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
        df['vol_ma20'] = grouped['vol'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        df['vol_ratio_5'] = df['vol'] / (df['vol_ma5'] + 1e-10)
        df['vol_ratio_10'] = df['vol'] / (df['vol_ma10'] + 1e-10)
        df['vol_ratio_20'] = df['vol'] / (df['vol_ma20'] + 1e-10)

        # ========== 量价背离指标（5日均价与5日均量的相关性） ==========
        # 先分别计算价格和成交量的变化率
        df['price_ret_5'] = grouped['close'].transform(lambda x: x.pct_change(5))
        df['vol_ret_5'] = grouped['vol'].transform(lambda x: x.pct_change(5))
        
        # 然后计算滚动相关性
        def calc_corr(group):
            return group['price_ret_5'].rolling(5, min_periods=3).corr(group['vol_ret_5'])
        
        df['price_volume_corr'] = grouped.apply(calc_corr).values
        
        # 清理临时列
        df = df.drop(['price_ret_5', 'vol_ret_5'], axis=1)

        # ========== 波动率（20日标准差） ==========
        df['volatility_20'] = grouped['pct_chg'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )

        # ========== 量价配合特征：累计收益率与成交量均值的比值 ==========
        # 5日量价配合度：5日累计收益率 / 5日均量（捕捉动量与成交量的背离）
        df['vol_price_ratio_5'] = df['momentum_5'] / (df['vol_ma5'] + 1e-10)
        # 10日量价配合度：10日累计收益率 / 10日均量
        df['vol_price_ratio_10'] = df['momentum_10'] / (df['vol_ma10'] + 1e-10)

        # ========== 滚动波动率增强特征 ==========
        # 5日滚动波动率
        df['volatility_5'] = grouped['pct_chg'].transform(
            lambda x: x.rolling(5, min_periods=3).std()
        )

        # 清理临时列
        df = df.drop(['pct_change', 'gain', 'loss'], axis=1)

        return df

    # =========================================================
    # 过滤极端交易日（停牌、一字涨停/跌停）
    # =========================================================
    def _filter_extreme_days(self, df):
        """
        过滤极端交易日样本：
        1. 停牌股票（volume == 0 或存在缺失值）
        2. 一字涨停/跌停（open == close == high == low 且涨跌幅显著）
        
        注意：这里只是标记/过滤单日数据，不破坏其他正常样本的30天连续序列构建
        """
        n_before = len(df)
        
        df = df.copy()
        
        # 标记停牌：volume == 0 或 open/high/low/close 存在缺失
        suspended = (
            (df['vol'] == 0) | 
            (df[['open', 'high', 'low', 'close']].isna().any(axis=1))
        )
        
        # 标记一字涨跌停：开盘价=收盘价=最高价=最低价 且 涨跌幅绝对值 >= 9.5%
        # A股涨跌停幅度通常为10%（或20%），使用9.5%作为阈值以捕捉边界情况
        limit_up = (
            (df['open'] == df['close']) & 
            (df['close'] == df['high']) & 
            (df['low'] == df['high']) & 
            (df['pct_chg'] >= 9.5)
        )
        
        limit_down = (
            (df['open'] == df['close']) & 
            (df['close'] == df['high']) & 
            (df['low'] == df['high']) & 
            (df['pct_chg'] <= -9.5)
        )
        
        # 合并所有过滤条件
        extreme_mask = suspended | limit_up | limit_down
        
        n_suspended = suspended.sum()
        n_limit_up = limit_up.sum()
        n_limit_down = limit_down.sum()
        n_filtered = extreme_mask.sum()
        
        # 打印过滤统计
        print(f"极端交易日过滤:")
        print(f"  - 停牌样本: {n_suspended}")
        print(f"  - 涨停样本: {n_limit_up}")
        print(f"  - 跌停样本: {n_limit_down}")
        print(f"  - 总计过滤: {n_filtered} / {n_before} ({100*n_filtered/n_before:.2f}%)")
        
        # 过滤掉极端样本
        df = df[~extreme_mask]
        
        return df

    # =========================================================
    # 资金流特征工程
    # =========================================================
    def add_moneyflow_features(self):
        self.df['super_large_net'] = self.df['super_large_net'].fillna(0)
        self.df['large_net'] = self.df['large_net'].fillna(0)
        self.df['medium_net'] = self.df['medium_net'].fillna(0)
        self.df['small_net'] = self.df['small_net'].fillna(0)
        self.df['total_turnover'] = self.df['total_turnover'].fillna(0)

        self.df['mf_main_force_ratio'] = (self.df['super_large_net'] + self.df['large_net']) / (self.df['total_turnover'] + 1e-10)
        self.df['mf_retail_ratio'] = (self.df['medium_net'] + self.df['small_net']) / (self.df['total_turnover'] + 1e-10)

        self.feature_cols = self.feature_cols + ['mf_main_force_ratio', 'mf_retail_ratio']

    # =========================================================
    # 截面Z-Score标准化（Cross-sectional Normalization）
    # =========================================================
    def cross_sectional_normalize(self, df, industry_neutral=False):
        """
        对所有输入特征在截面上进行Z-Score标准化
        
        参数:
            df: 输入数据
            industry_neutral: 是否对基本面因子进行行业中性化处理
                             如果为True，将对PE、PB、ROE等因子按行业分组做Z-Score
        """
        feature_cols = self.feature_cols
        
        if industry_neutral and 'industry_code' in df.columns:
            # 分离基本面因子和其他因子
            fundamental_factors = ['pe_ttm', 'pb', 'roe', 'ps_ttm', 'pcf_ncf', 'pcf_ocf']
            fundamental_cols = [col for col in fundamental_factors if col in feature_cols]
            other_cols = [col for col in feature_cols if col not in fundamental_cols]
            
            print(f"行业中性化处理:")
            print(f"  - 基本面因子 ({len(fundamental_cols)}个): {fundamental_cols}")
            print(f"  - 其他因子 ({len(other_cols)}个): {other_cols}")
            
            # 1. 对其他因子进行全市场截面Z-Score
            if other_cols:
                def normalize_other(group):
                    for col in other_cols:
                        if col in group.columns:
                            mean_val = group[col].mean()
                            std_val = group[col].std()
                            group[col] = (group[col] - mean_val) / (std_val + 1e-10)
                    return group
                
                df = df.groupby(level='trade_date', group_keys=False).apply(normalize_other)
            
            # 2. 对基本面因子进行行业中性化Z-Score
            if fundamental_cols:
                def neutralize_fundamental(group):
                    for col in fundamental_cols:
                        if col in group.columns:
                            # 按行业分组计算Z-Score
                            industry_means = group.groupby('industry_code')[col].transform('mean')
                            industry_stds = group.groupby('industry_code')[col].transform('std')
                            group[col] = (group[col] - industry_means) / (industry_stds + 1e-10)
                    return group
                
                df = df.groupby(level='trade_date', group_keys=False).apply(neutralize_fundamental)
                print(f"  - 基本面因子已完成行业中性化")
        else:
            # 标准全市场截面Z-Score
            def normalize_group(group):
                for col in feature_cols:
                    if col in group.columns:
                        mean_val = group[col].mean()
                        std_val = group[col].std()
                        group[col] = (group[col] - mean_val) / (std_val + 1e-10)
                return group
            
            df = df.groupby(level='trade_date', group_keys=False).apply(normalize_group)
        
        print("截面Z-Score标准化完成")
        return df

    # =========================================================
    # 特征选择
    # =========================================================
    def select_features(self):
        """选择并构建所有特征"""

        base_features = [
            'open',
            'high',
            'low',
            'close',
            'vol',
            'pct_chg',
            'vwap'
        ]

        self.df = self.df[
            base_features
        ].copy()

        # 添加技术指标
        self.df = self.add_technical_indicators(self.df)

        # ========== 计算超额收益率（个股收益率减去截面中位数） ==========
        self.df['excess_return'] = self.df.groupby(level='trade_date', group_keys=False)['pct_chg'].apply(
            lambda x: x - x.median()
        )

        # 定义最终特征列（去除标签相关列）
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'vol', 'pct_chg', 'vwap',
            'ma5', 'ma10', 'ma20', 'ma60',
            'ema12', 'ema26', 'dif', 'dea', 'macd',
            'rsi',
            'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
            'vol_ma5', 'vol_ma10', 'vol_ma20',
            'vol_ratio_5', 'vol_ratio_10', 'vol_ratio_20',
            'price_volume_corr',
            'volatility_5', 'volatility_20',
            'vol_price_ratio_5', 'vol_price_ratio_10',
            'excess_return'
        ]

        # 填充缺失值
        self.df = self.df.fillna(0)

        print(
            f"Features selected: "
            f"{len(self.feature_cols)} dimensions"
        )

        return self.feature_cols

    # =========================================================
    # 标签构建 - 三个维度的变化率
    # =========================================================
    def construct_labels(
        self,
        horizon=1
    ):
        """
        构建三个维度的标签：
        - y_op: 次日开盘价变化率 (次日开盘价 - 当日收盘价) / 当日收盘价
        - y_lp: 次日最低价变化率 (次日最低价 - 当日收盘价) / 当日收盘价
        - y_hp: 次日最高价变化率 (次日最高价 - 当日收盘价) / 当日收盘价
        """

        df = self.df.copy()

        # 获取次日数据
        def shift_by_stock(group, n):
            return group.shift(-n)

        df['next_open'] = df.groupby('ts_code')['open'].transform(
            lambda x: shift_by_stock(x, horizon)
        )
        df['next_low'] = df.groupby('ts_code')['low'].transform(
            lambda x: shift_by_stock(x, horizon)
        )
        df['next_high'] = df.groupby('ts_code')['high'].transform(
            lambda x: shift_by_stock(x, horizon)
        )

        # 计算三个维度的变化率
        df['label_op'] = (df['next_open'] - df['close']) / (df['close'] + 1e-10)
        df['label_lp'] = (df['next_low'] - df['close']) / (df['close'] + 1e-10)
        df['label_hp'] = (df['next_high'] - df['close']) / (df['close'] + 1e-10)

        self.label_cols = ['label_op', 'label_lp', 'label_hp']

        # 清理临时列
        df = df.drop(['next_open', 'next_low', 'next_high'], axis=1)

        self.df = df

        print(
            f"Labels constructed: "
            f"{self.label_cols}"
        )

    # =========================================================
    # 滑动窗口 - 支持多目标标签
    # =========================================================
    def create_sequences(
        self,
        window_len=30,
        horizon=1,
        train_date_range=(
            '2024-06-01',
            '2025-03-31'
        ),
        val_date_range=(
            '2025-04-01',
            '2025-06-30'
        )
    ):

        feature_cols = self.feature_cols
        label_cols = self.label_cols

        df = self.df.copy()

        df = df.sort_index()

        # =====================================================
        # 预处理日期范围（去掉 -）
        # =====================================================
        train_start = train_date_range[0].replace('-', '')
        train_end = train_date_range[1].replace('-', '')
        val_start = val_date_range[0].replace('-', '')
        val_end = val_date_range[1].replace('-', '')

        n_features = len(feature_cols)

        # 第一遍：计数
        train_count = 0
        val_count = 0
        grouped = df.groupby('ts_code')

        for code, group in tqdm(grouped, desc="Counting samples"):
            group = group.sort_index()
            dates_arr = group.index.get_level_values('trade_date').values
            T = len(group)
            for i in range(T - window_len):
                date = dates_arr[i + window_len - 1]
                if train_start <= date <= train_end:
                    train_count += 1
                elif val_start <= date <= val_end:
                    val_count += 1

        print(f"Train samples: {train_count}")
        print(f"Val samples: {val_count}")

        import tempfile
        tmpdir = os.path.join(self.data_root, '.lstm_cache')
        os.makedirs(tmpdir, exist_ok=True)
        print(f"[INFO] memmap 缓存目录: {tmpdir}")

        def _make_memmap(shape, dtype, suffix):
            path = os.path.join(tmpdir, f'lstm_{suffix}_{os.getpid()}.dat')
            return np.memmap(path, dtype=dtype, mode='w+', shape=shape), path

        X_train, _  = _make_memmap((train_count, window_len, n_features), np.float32, 'X_train') if train_count > 0 else (np.array([], dtype=np.float32).reshape(0, window_len, n_features), None)
        y_op_train, _  = _make_memmap((train_count, 1), np.float32, 'yop_train') if train_count > 0 else (np.array([], dtype=np.float32).reshape(0, 1), None)
        y_lp_train, _  = _make_memmap((train_count, 1), np.float32, 'ylp_train') if train_count > 0 else (np.array([], dtype=np.float32).reshape(0, 1), None)
        y_hp_train, _  = _make_memmap((train_count, 1), np.float32, 'yhp_train') if train_count > 0 else (np.array([], dtype=np.float32).reshape(0, 1), None)
        dates_train_mem, dates_train_path = _make_memmap((train_count,), dtype='U10', suffix='dates_train') if train_count > 0 else (np.array([], dtype='U10'), None)
        stocks_train_mem, stocks_train_path = _make_memmap((train_count,), dtype='U10', suffix='stocks_train') if train_count > 0 else (np.array([], dtype='U10'), None)
        close_train_mem, close_train_path = _make_memmap((train_count, 1), np.float32, 'close_train') if train_count > 0 else (np.array([], dtype=np.float32).reshape(0, 1), None)

        X_val, _  = _make_memmap((val_count, window_len, n_features), np.float32, 'X_val') if val_count > 0 else (np.array([], dtype=np.float32).reshape(0, window_len, n_features), None)
        y_op_val, _  = _make_memmap((val_count, 1), np.float32, 'yop_val') if val_count > 0 else (np.array([], dtype=np.float32).reshape(0, 1), None)
        y_lp_val, _  = _make_memmap((val_count, 1), np.float32, 'ylp_val') if val_count > 0 else (np.array([], dtype=np.float32).reshape(0, 1), None)
        y_hp_val, _  = _make_memmap((val_count, 1), np.float32, 'yhp_val') if val_count > 0 else (np.array([], dtype=np.float32).reshape(0, 1), None)
        dates_val_mem, dates_val_path = _make_memmap((val_count,), dtype='U10', suffix='dates_val') if val_count > 0 else (np.array([], dtype='U10'), None)
        stocks_val_mem, stocks_val_path = _make_memmap((val_count,), dtype='U10', suffix='stocks_val') if val_count > 0 else (np.array([], dtype='U10'), None)
        close_val_mem, close_val_path  = _make_memmap((val_count, 1), np.float32, 'close_val') if val_count > 0 else (np.array([], dtype=np.float32).reshape(0, 1), None)

        # 第二遍：写入
        train_idx = 0
        val_idx = 0

        for code, group in tqdm(grouped, desc="Writing sequences"):
            group = group.sort_index()
            features = group[feature_cols].values
            y_op = group['label_op'].values
            y_lp = group['label_lp'].values
            y_hp = group['label_hp'].values
            dates_arr = group.index.get_level_values('trade_date').values
            current_closes = group['close'].values
            T = len(group)

            for i in range(T - window_len):
                x = features[i:i + window_len]
                op = y_op[i + window_len - 1]
                lp = y_lp[i + window_len - 1]
                hp = y_hp[i + window_len - 1]
                date = dates_arr[i + window_len - 1]
                current_close = current_closes[i + window_len - 1]

                if np.isnan(op) or np.isnan(lp) or np.isnan(hp):
                    continue
                if np.isnan(x).any():
                    continue

                if train_start <= date <= train_end:
                    X_train[train_idx] = x
                    y_op_train[train_idx] = op
                    y_lp_train[train_idx] = lp
                    y_hp_train[train_idx] = hp
                    dates_train_mem[train_idx] = str(date)
                    stocks_train_mem[train_idx] = str(code)
                    close_train_mem[train_idx] = current_close
                    train_idx += 1
                elif val_start <= date <= val_end:
                    X_val[val_idx] = x
                    y_op_val[val_idx] = op
                    y_lp_val[val_idx] = lp
                    y_hp_val[val_idx] = hp
                    dates_val_mem[val_idx] = str(date)
                    stocks_val_mem[val_idx] = str(code)
                    close_val_mem[val_idx] = current_close
                    val_idx += 1

        memo = {
            'dates_train_path': dates_train_path,
            'stocks_train_path': stocks_train_path,
            'close_train_path': close_train_path,
            'dates_val_path': dates_val_path,
            'stocks_val_path': stocks_val_path,
            'close_val_path': close_val_path,
            'train_count': train_count,
            'val_count': val_count,
        }
        self._memmap_meta = memo

        dates_train = np.array(dates_train_mem, dtype=str) if train_count > 0 else np.array([], dtype=str)
        stocks_train = np.array(stocks_train_mem, dtype=str) if train_count > 0 else np.array([], dtype=str)
        current_close_train = np.array(close_train_mem, dtype=np.float32).reshape(-1, 1) if train_count > 0 else np.array([], dtype=np.float32).reshape(0, 1)
        dates_val = np.array(dates_val_mem, dtype=str) if val_count > 0 else np.array([], dtype=str)
        stocks_val = np.array(stocks_val_mem, dtype=str) if val_count > 0 else np.array([], dtype=str)
        current_close_val = np.array(close_val_mem, dtype=np.float32).reshape(-1, 1) if val_count > 0 else np.array([], dtype=np.float32).reshape(0, 1)

        return (
            X_train, y_op_train, y_lp_train, y_hp_train,
            dates_train, stocks_train, current_close_train,
            X_val, y_op_val, y_lp_val, y_hp_val,
            dates_val, stocks_val, current_close_val
        )

    # =========================================================
    # 标准化
    # =========================================================
    def fit_standardize(
        self,
        X_train,
        X_val
    ):
        if len(X_train) == 0 and len(X_val) == 0:
            return X_train, X_val

        if len(X_train) == 0:
            fit_data = X_val
        else:
            fit_data = X_train

        N_fit, T_fit, F_fit = fit_data.shape

        self.scaler = StandardScaler()

        sample_ratio = min(1.0, 50000 / max(1, N_fit))
        n_sample = max(1000, int(N_fit * sample_ratio))
        n_sample = min(n_sample, N_fit)

        np.random.seed(42)
        sample_indices = np.random.choice(N_fit, n_sample, replace=False)

        sample_flat = fit_data[sample_indices].reshape(-1, F_fit)
        self.scaler.fit(sample_flat)
        del sample_flat

        chunk_size = 2000

        def _transform_inplace(X, name):
            N, T, F = X.shape
            print(f"标准化{name}: {X.shape}（分批处理，每批 {chunk_size} 样本）")
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                chunk_flat = X[start:end].reshape(-1, F)
                chunk_flat[:] = self.scaler.transform(chunk_flat)
                X[start:end] = chunk_flat.reshape(end - start, T, F)
            return X

        if len(X_train) > 0:
            X_train = _transform_inplace(X_train, "训练集")

        if len(X_val) > 0:
            X_val = _transform_inplace(X_val, "验证集")
        else:
            X_val = np.array([])

        return X_train, X_val

    # =========================================================
    # pipeline - 支持多目标标签
    # =========================================================
    def run_pipeline(
        self,
        start_date='2024-06-01',
        end_date='2025-06-30',
        stock_pool='hs300',
        window_len=30,
        horizon=1,
        train_range=(
            '2024-06-01',
            '2025-03-31'
        ),
        val_range=(
            '2025-04-01',
            '2025-06-30'
        ),
        industry_neutral=False,
        use_fundamental=False,
        use_moneyflow=False
    ):

        self.load_data(
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool
        )

        # 根据开关加载基本面和资金流数据
        if use_fundamental:
            self.load_fundamental_data()
            if self.df_fundamental is not None:
                print(f"基本面数据: {len(self.df_fundamental)} 条记录")

        if use_moneyflow:
            self.load_moneyflow_data()
            if self.df_moneyflow is not None:
                print(f"资金流数据: {len(self.df_moneyflow)} 条记录")

        self.select_features()

        self.construct_labels(
            horizon=horizon
        )

        # =====================================================
        # 合并基本面和资金流数据（仅当启用时）
        # =====================================================
        merged = self.df.copy()

        extra_feature_cols = []

        if use_fundamental and self.df_fundamental is not None:
            fund_cols = ['trade_date', 'ts_code', 'pe_ttm', 'pb', 'roe']
            available_cols = [c for c in fund_cols if c in self.df_fundamental.columns]
            if available_cols:
                value_cols = [c for c in available_cols if c not in ['trade_date', 'ts_code']]
                merged = merged.merge(
                    self.df_fundamental[available_cols],
                    on=['trade_date', 'ts_code'],
                    how='left'
                )
                extra_feature_cols.extend(value_cols)
                print(f"合并基本面特征: {len(value_cols)} 个")

        if use_moneyflow and self.df_moneyflow is not None:
            mf_cols = ['trade_date', 'ts_code', 'super_large_net', 'large_net', 'medium_net', 'small_net', 'total_turnover']
            available_cols = [c for c in mf_cols if c in self.df_moneyflow.columns]
            if available_cols:
                value_cols = [c for c in available_cols if c not in ['trade_date', 'ts_code']]
                merged = merged.merge(
                    self.df_moneyflow[available_cols],
                    on=['trade_date', 'ts_code'],
                    how='left'
                )
                extra_feature_cols.extend(value_cols)
                print(f"合并资金流特征: {len(value_cols)} 个")

        if use_moneyflow and 'super_large_net' in merged.columns:
            self.add_moneyflow_features()
            print("已添加资金流特征")

        if extra_feature_cols:
            for col in extra_feature_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(merged[col].median() if merged[col].notna().any() else 0)
            self.feature_cols = self.feature_cols + extra_feature_cols
            print(f"更新特征维度: {len(self.feature_cols)} 维（含 {len(extra_feature_cols)} 个额外特征）")

        self.df = merged

        # 确保 trade_date 和 ts_code 设置为索引（用于截面标准化）
        if 'trade_date' in self.df.columns:
            self.df = self.df.set_index(['trade_date', 'ts_code'])

        # 截面Z-Score标准化（支持行业中性化）
        self.df = self.cross_sectional_normalize(self.df, industry_neutral=industry_neutral)

        # 多目标标签：OP、LP、HP，以及日期、股票代码、收盘价
        X_train, y_op_train, y_lp_train, y_hp_train, dates_train, stocks_train, current_close_train, \
        X_val, y_op_val, y_lp_val, y_hp_val, dates_val, stocks_val, current_close_val = (
            self.create_sequences(
                window_len=window_len,
                horizon=horizon,
                train_date_range=train_range,
                val_date_range=val_range
            )
        )

        X_train, X_val = (
            self.fit_standardize(
                X_train,
                X_val
            )
        )

        return (
            X_train,
            y_op_train,
            y_lp_train,
            y_hp_train,
            dates_train,
            stocks_train,
            current_close_train,
            X_val,
            y_op_val,
            y_lp_val,
            y_hp_val,
            dates_val,
            stocks_val,
            current_close_val
        )
