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

        self.df = None

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
    # 截面Z-Score标准化（Cross-sectional Normalization）
    # =========================================================
    def cross_sectional_normalize(self, df):
        """对所有输入特征在截面上进行Z-Score标准化"""
        feature_cols = self.feature_cols
        
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

        # =====================================================
        # 按股票分组
        # =====================================================
        grouped = df.groupby('ts_code')

        # 【内存优化】直接分开存储训练集和验证集
        X_train_list = []
        y_op_train_list = []
        y_lp_train_list = []
        y_hp_train_list = []
        dates_train_list = []
        stocks_train_list = []
        current_close_train_list = []

        X_val_list = []
        y_op_val_list = []
        y_lp_val_list = []
        y_hp_val_list = []
        dates_val_list = []
        stocks_val_list = []
        current_close_val_list = []

        for code, group in tqdm(
            grouped,
            desc="Creating sequences"
        ):

            group = group.sort_index()

            # 提取特征矩阵 (T, F)
            features = group[feature_cols].values

            # 提取标签
            y_op = group['label_op'].values
            y_lp = group['label_lp'].values
            y_hp = group['label_hp'].values

            # 提取元数据
            dates = group.index.get_level_values('trade_date').values
            current_closes = group['close'].values

            # 滑动窗口
            T = len(group)
            for i in range(T - window_len):
                # 输入窗口
                x = features[i:i+window_len]

                # 标签（三个维度）
                op = y_op[i+window_len-1]
                lp = y_lp[i+window_len-1]
                hp = y_hp[i+window_len-1]

                # 元数据
                date = dates[i+window_len-1]

                current_close = current_closes[i+window_len-1]

                # 检查标签是否有效
                if np.isnan(op) or np.isnan(lp) or np.isnan(hp):
                    continue

                # 检查输入是否有效
                if np.isnan(x).any():
                    continue

                # 【内存优化】在生成时就按日期范围分割
                if train_start <= date <= train_end:
                    X_train_list.append(x)
                    y_op_train_list.append(op)
                    y_lp_train_list.append(lp)
                    y_hp_train_list.append(hp)
                    dates_train_list.append(date)
                    stocks_train_list.append(code)
                    current_close_train_list.append(current_close)
                elif val_start <= date <= val_end:
                    X_val_list.append(x)
                    y_op_val_list.append(op)
                    y_lp_val_list.append(lp)
                    y_hp_val_list.append(hp)
                    dates_val_list.append(date)
                    stocks_val_list.append(code)
                    current_close_val_list.append(current_close)

        # =====================================================
        # 转换为numpy数组（仅转换需要的数据）
        # =====================================================
        X_train = np.array(X_train_list, dtype=np.float32)
        y_op_train = np.array(y_op_train_list, dtype=np.float32).reshape(-1, 1)
        y_lp_train = np.array(y_lp_train_list, dtype=np.float32).reshape(-1, 1)
        y_hp_train = np.array(y_hp_train_list, dtype=np.float32).reshape(-1, 1)
        dates_train = np.array(dates_train_list, dtype=str)
        stocks_train = np.array(stocks_train_list, dtype=str)
        current_close_train = np.array(current_close_train_list, dtype=np.float32).reshape(-1, 1)

        X_val = np.array(X_val_list, dtype=np.float32)
        y_op_val = np.array(y_op_val_list, dtype=np.float32).reshape(-1, 1)
        y_lp_val = np.array(y_lp_val_list, dtype=np.float32).reshape(-1, 1)
        y_hp_val = np.array(y_hp_val_list, dtype=np.float32).reshape(-1, 1)
        dates_val = np.array(dates_val_list, dtype=str)
        stocks_val = np.array(stocks_val_list, dtype=str)
        current_close_val = np.array(current_close_val_list, dtype=np.float32).reshape(-1, 1)

        print(
            f"Train samples: {len(X_train)}"
        )
        print(
            f"Val samples: {len(X_val)}"
        )

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
        """
        对输入特征进行时间序列上的标准化
        (在截面上已做过Z-Score，这里对时间维度做标准化)
        
        使用批量处理提高效率
        """

        N_train, T_train, F_train = X_train.shape
        N_val, T_val, F_val = X_val.shape

        assert F_train == F_val

        # =====================================================
        # 采样拟合 scaler（避免全量数据）
        # =====================================================
        self.scaler = StandardScaler()

        sample_ratio = min(1.0, 50000 / N_train)
        n_sample = max(10000, int(N_train * sample_ratio))

        np.random.seed(42)
        sample_indices = np.random.choice(N_train, n_sample, replace=False)

        sample_flat = X_train[sample_indices].reshape(-1, F_train)
        self.scaler.fit(sample_flat)
        del sample_flat

        # =====================================================
        # 批量处理训练集：一次性处理所有样本
        # =====================================================
        print(f"标准化训练集: {X_train.shape}")
        X_train_flat = X_train.reshape(-1, F_train)
        X_train_flat[:] = self.scaler.transform(X_train_flat)
        X_train = X_train_flat.reshape(N_train, T_train, F_train)

        # =====================================================
        # 批量处理验证集
        # =====================================================
        if len(X_val) > 0:
            print(f"标准化验证集: {X_val.shape}")
            X_val_flat = X_val.reshape(-1, F_val)
            X_val_flat[:] = self.scaler.transform(X_val_flat)
            X_val = X_val_flat.reshape(N_val, T_val, F_val)
        else:
            X_val = np.array([])

        return (
            X_train,
            X_val
        )

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
        )
    ):

        self.load_data(
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool
        )

        self.select_features()

        self.construct_labels(
            horizon=horizon
        )

        # 截面Z-Score标准化
        self.df = self.cross_sectional_normalize(self.df)

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
