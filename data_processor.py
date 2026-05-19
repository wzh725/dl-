"""
深度学习大作业 - 数据处理模块
功能：
1. 读取daily截面数据，合并为面板数据
2. 过滤ST股、北交所股票（可配置）
3. 缺失值处理与特征标准化（避免未来信息）
4. 构造n日收益率标签
5. 构造滑动窗口样本（过去L天特征 → 未来n日收益）
6. 按时间划分训练集和验证集（禁止随机打乱）
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict


class DataProcessor:
    def __init__(self, data_root: str):
        """
        Args:
            data_root: 数据根目录，应包含 daily/, basic.csv, trade_cal.csv, stock_st/ 等
        """
        self.data_root = data_root
        self.daily_dir = os.path.join(data_root, 'daily')
        self.basic_path = os.path.join(data_root, 'basic.csv')
        self.trade_cal_path = os.path.join(data_root, 'trade_cal.csv')
        self.st_dir = os.path.join(data_root, 'stock_st')
        
        self.df = None          # 合并后的面板数据 (date, ts_code, features)
        self.feature_cols = None
        self.label_col = None
        self.scaler = None      # 训练集拟合的标准化器
    
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
                print(f"Warning: {fname} missing required columns, skipping.")
                continue
            # 将 trade_date 转换为字符串 YYYYMMDD
            df['trade_date'] = df['trade_date'].astype(str)
            # 按 trade_date 分组，每组股票代码集合
            for date, group in df.groupby('trade_date'):
                if date not in components:
                    components[date] = set()
                components[date].update(group[code_col].unique())

        print(f"Loaded HS300 components for {len(components)} distinct dates.")
        return components
        
    # ==================== 1. 数据加载与过滤 ====================
    def load_data(self, 
                  start_date: str = '2022-01-01',
                  end_date: str = '2025-12-31',
                  stock_pool: str = 'all',           # 'all', 'hs300', 'cyb', 'kcb'
                  exclude_st: bool = True,
                  exclude_bj: bool = True) -> pd.DataFrame:
        """
        加载日频数据并过滤股票池
        Returns:
            DataFrame: MultiIndex (trade_date, ts_code) 包含所有量价特征
        """
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
        for date in tqdm(trade_dates, desc="Loading daily data"):
            file_path = os.path.join(self.daily_dir, f"{date}.csv")
            if not os.path.exists(file_path):
                continue
            df_day = pd.read_csv(file_path)
            df_day['trade_date'] = date
            all_dfs.append(df_day)
        
        if not all_dfs:
            raise ValueError("No daily data loaded. Check data path and date range.")
        
        df = pd.concat(all_dfs, ignore_index=True)
        
        # 过滤ST股（根据每日ST列表）
        if exclude_st:
            st_files = [f for f in os.listdir(self.st_dir) if f.endswith('.csv')]
            st_codes = set()
            for st_file in st_files:
                st_df = pd.read_csv(os.path.join(self.st_dir, st_file))
                st_codes.update(st_df['ts_code'].unique())
            df = df[~df['ts_code'].isin(st_codes)]
        
        # 过滤北交所（市场类型为北交所的股票）
        if exclude_bj:
            basic = pd.read_csv(self.basic_path)
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
            print(f"Filtered to HS300 constituents: {original_len} -> {len(df)} rows")
        elif stock_pool == 'cyb':
            basic = pd.read_csv(self.basic_path)
            pool_codes = basic[basic['market'] == '创业板']['ts_code'].unique()
            df = df[df['ts_code'].isin(pool_codes)]
        elif stock_pool == 'kcb':
            basic = pd.read_csv(self.basic_path)
            pool_codes = basic[basic['market'] == '科创板']['ts_code'].unique()
            df = df[df['ts_code'].isin(pool_codes)]
        # else 'all' 不做额外筛选
        
        # 设置多重索引
        df = df.set_index(['trade_date', 'ts_code']).sort_index()
        self.df = df
        print(f"Data loaded: {df.index.levshape[0]} dates, {df.index.levshape[1]} stocks")
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
                        base_features: List[str] = None,
                        add_ta: bool = True) -> List[str]:
        """
        确定最终特征列
        Args:
            base_features: 基础量价特征，默认为 ['open','high','low','close','vol','pct_chg','vwap']
            add_ta: 是否添加技术指标
        Returns:
            特征列名列表
        """
        if base_features is None:
            base_features = ['open', 'high', 'low', 'close', 'vol', 'pct_chg', 'vwap']
        
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
        print(f"Label constructed: {self.label_col} (horizon={horizon})")
        return self.df
    
    # ==================== 4. 滑动窗口与样本构造 ====================
    def create_sequences(self,
                         window_len: int = 20,
                         horizon: int = 1,
                         train_date_range: Tuple[str, str] = ('2019-01-01', '2024-12-31'),
                         val_date_range: Tuple[str, str] = ('2025-01-01', '2025-12-31')) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        为每只股票生成滑动窗口样本，并按时间划分训练/验证集
        Args:
            window_len: 输入序列长度（天数）
            horizon: 预测步长（必须与construct_labels中的horizon一致）
            train_date_range: 训练集时间范围 (start, end)
            val_date_range: 验证集时间范围 (start, end)
            min_samples_per_stock: 每只股票至少产生多少个样本，否则丢弃该股票
        Returns:
            X_train, y_train, X_val, y_val (numpy arrays)
        """
        if self.label_col not in self.df.columns:
            raise ValueError("Labels not constructed. Call construct_labels() first.")
        
        # 确保特征列存在，去除 label 列和索引列
        feature_data = self.df[self.feature_cols].copy()
        # 处理缺失值：前向填充（每个股票内部）
        feature_data = feature_data.groupby('ts_code').transform(lambda x: x.ffill().bfill())
        # 删除仍然含有 NaN 的行（一般是股票上市初期或退市边缘）
        valid_idx = feature_data.notna().all(axis=1)
        self.df = self.df[valid_idx]
        feature_data = feature_data[valid_idx]
        
        # 按日期和股票重新整理
        df_clean = self.df[[self.label_col]].copy()
        df_clean = df_clean.loc[valid_idx]
        df_clean = pd.concat([df_clean, feature_data], axis=1)
        
        # 按股票分组生成序列
        X_list, y_list = [], []
        date_info = []  # 存储样本对应的日期（用于回测）
        stock_info = []
        
        grouped = df_clean.groupby('ts_code')
        for stock, grp in tqdm(grouped, desc="Creating sequences"):
            grp = grp.sort_index(level='trade_date')  # 按时间升序
            dates = grp.index.get_level_values('trade_date').unique()
            values = grp[self.feature_cols].values
            labels = grp[self.label_col].values
            
            # 滑动窗口
            for i in range(len(dates) - window_len - horizon + 1):
                # 特征：过去 window_len 天的特征
                X_seq = values[i:i+window_len, :]
                # 标签：第 i+window_len 天后的 horizon 日收益（对应日期为 dates[i+window_len]）
                y_label = labels[i+window_len]   # 因为标签已对齐到第 i+window_len 天的未来收益
                # 样本日期为窗口的最后一天（即做出预测并可以交易的那一天）
                sample_date = dates[i+window_len]
                
                X_list.append(X_seq)
                y_list.append(y_label)
                date_info.append(sample_date)
                stock_info.append(stock)
        
        if len(X_list) == 0:
            raise ValueError("No sequences generated. Check window_len/horizon and data length.")
        
        # 转换为数组
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
        dates_arr = np.array(date_info)
        stocks_arr = np.array(stock_info)
        
        # 按时间划分（禁止打乱）
        # 将传入的日期范围字符串转换为 YYYYMMDD 格式
        def to_yyyymmdd(d):
            if isinstance(d, str) and '-' in d:
                return d.replace('-', '')
            return str(d)

        train_start = to_yyyymmdd(train_date_range[0])
        train_end   = to_yyyymmdd(train_date_range[1])
        val_start   = to_yyyymmdd(val_date_range[0])
        val_end     = to_yyyymmdd(val_date_range[1])

        train_mask = (dates_arr >= train_start) & (dates_arr <= train_end)
        val_mask   = (dates_arr >= val_start) & (dates_arr <= val_end)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        # 保存样本对应的日期和股票，便于回测
        self.train_dates = dates_arr[train_mask]
        self.train_stocks = stocks_arr[train_mask]
        self.val_dates = dates_arr[val_mask]
        self.val_stocks = stocks_arr[val_mask]
        
        return X_train, y_train, X_val, y_val
    
    # ==================== 5. 标准化（避免未来信息） ====================
    def fit_standardize(self, X_train: np.ndarray, X_val: np.ndarray = None):
        """
        使用训练集拟合标准化器，并转换训练集和验证集。
        对每个特征维度独立标准化，同时对时间序列的每个时刻使用相同的scaler。
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
            M, T2, F2 = X_val.shape
            val_flat = X_val.reshape(-1, F2)
            X_val_scaled = self.scaler.transform(val_flat).reshape(M, T2, F2)
        
        return X_train_scaled, X_val_scaled
    
    # ==================== 6. 完整流程（一站式调用） ====================
    def run_pipeline(self,
                     start_date='2022-01-01',
                     end_date='2025-12-31',
                     stock_pool='hs300',
                     window_len=20,
                     horizon=1,
                     train_range=('2022-01-01', '2024-12-31'),
                     val_range=('2025-01-01', '2025-12-31'),
                     add_ta=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        执行完整数据处理流程
        Returns:
            X_train_scaled, y_train, X_val_scaled, y_val
        """
        # 1. 加载并过滤
        self.load_data(start_date, end_date, stock_pool, exclude_st=True, exclude_bj=True)
        # 2. 特征选择与构造
        self.select_features(add_ta=add_ta)
        # 3. 构造标签
        self.construct_labels(horizon=horizon, label_type='return')
        # 4. 生成序列
        X_train, y_train, X_val, y_val = self.create_sequences(
            window_len=window_len,
            horizon=horizon,
            train_date_range=train_range,
            val_date_range=val_range
        )
        # 5. 标准化
        X_train_scaled, X_val_scaled = self.fit_standardize(X_train, X_val)
        
        # 保存一些元数据供外部使用
        self.input_shape = (window_len, len(self.feature_cols))
        return X_train_scaled, y_train, X_val_scaled, y_val


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置路径（请根据实际数据存放位置修改）
    DATA_ROOT = "D:\科大云盘\文件\科大云盘\大作业\A股数据"   # 替换为实际路径
    
    processor = DataProcessor(DATA_ROOT)
    X_train, y_train, X_val, y_val = processor.run_pipeline(
        start_date='2022-01-01',
        end_date='2025-12-31',
        stock_pool='hs300',      # 可选 'all', 'hs300', 'cyb', 'kcb'
        window_len=20,           # 过去20天
        horizon=1,               # 预测次日收益
        train_range=('2022-01-01', '2024-12-31'),
        val_range=('2025-01-01', '2025-12-31'),
        add_ta=True
    )
