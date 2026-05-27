import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from model import AssociatedResidualNet


class DailyTrader:
    """每日预测引擎：加载最新数据 -> 特征工程 -> 模型推理 -> 输出得分排名"""

    def __init__(self, model_dir, data_root):
        self.model_dir = model_dir
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        self.meta = None
        self.model = None
        self.feature_cols = None
        self.stock_name_map = {}

    # =========================================================
    # 加载模型、Scaler、元数据
    # =========================================================
    def load_artifacts(self):
        """从模型目录加载模型权重 + StandardScaler + 模型元数据"""
        model_path = os.path.join(self.model_dir, 'best_model.pth')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        meta_path = os.path.join(self.model_dir, 'model_meta.pkl')

        for fpath, fname in [(model_path, '模型'), (scaler_path, 'Scaler'), (meta_path, '元数据')]:
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"{fname}文件不存在: {fpath}\n"
                    "请先运行 train.py 训练模型，train.py 已更新为自动保存 scaler.pkl 和 model_meta.pkl"
                )

        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.model = AssociatedResidualNet(
            input_dim=self.meta['input_dim'],
            hidden_dim=self.meta['hidden_dim'],
            num_layers=self.meta['num_layers'],
            dropout=self.meta['dropout'],
            fc_dropout=self.meta['fc_dropout'],
            bidirectional=self.meta['bidirectional']
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        print(f"[INFO] 模型加载成功")
        print(f"  - hidden_dim={self.meta['hidden_dim']}, num_layers={self.meta['num_layers']}")
        print(f"  - dropout={self.meta['dropout']}, fc_dropout={self.meta['fc_dropout']}")
        print(f"  - 基本面={'是' if self.meta.get('use_fundamental') else '否'}")
        print(f"  - 资金流={'是' if self.meta.get('use_moneyflow') else '否'}")

    # =========================================================
    # 加载股票名称映射
    # =========================================================
    def load_stock_names(self):
        basic_path = os.path.join(self.data_root, 'basic.csv')
        if os.path.exists(basic_path):
            basic = pd.read_csv(basic_path)
            self.stock_name_map = dict(zip(basic['ts_code'], basic.get('name', basic['ts_code'])))

    # =========================================================
    # 加载最近 N 天的日线数据并做预处理
    # =========================================================
    def load_and_prepare_data(self, lookback_days=45):
        """
        加载数据并做与训练时完全一致的特征工程。

        使用 DataProcessor 的方法确保一致性：
          1. load_data() -> 加载日线、过滤ST、北交所、停牌/涨停/跌停
          2. select_features() -> 提取基础列
          3. add_technical_indicators() -> 计算技术指标
          4. 加载基本面/资金流并合并
        """
        latest_date = self._find_latest_trade_date()
        print(f"[INFO] 数据目录最新交易日: {latest_date}")

        # 计算起始日期（使用交易日历）
        cal_path = os.path.join(self.data_root, 'trade_cal.csv')
        cal = pd.read_csv(cal_path)
        cal['cal_date'] = cal['cal_date'].astype(str)
        cal = cal[cal['is_open'] == 1].sort_values('cal_date')
        all_dates = cal['cal_date'].unique().tolist()

        try:
            latest_idx = all_dates.index(latest_date)
        except ValueError:
            print(f"[WARN] 交易日期 {latest_date} 不在交易日历中，使用最近的交易日")
            from bisect import bisect_right
            latest_idx = bisect_right(all_dates, latest_date) - 1
            if latest_idx < 0:
                raise RuntimeError(f"找不到 {latest_date} 附近的交易日")
            latest_date = all_dates[latest_idx]
            print(f"[INFO] 使用最近交易日: {latest_date}")

        start_idx = max(0, latest_idx - lookback_days)
        needed_dates = all_dates[start_idx:latest_idx + 1]
        data_start = needed_dates[0]
        data_end = needed_dates[-1]

        # 格式化为 YYYY-MM-DD 供 DataProcessor 使用
        data_start_fmt = f"{data_start[:4]}-{data_start[4:6]}-{data_start[6:8]}"
        data_end_fmt = f"{data_end[:4]}-{data_end[4:6]}-{data_end[6:8]}"
        print(f"[INFO] 加载数据范围: {data_start_fmt} ~ {data_end_fmt} ({len(needed_dates)}个交易日)")

        # 使用 DataProcessor 加载并过滤数据
        processor = DataProcessor(data_root=self.data_root)
        processor.load_data(
            start_date=data_start_fmt,
            end_date=data_end_fmt,
            stock_pool='all',
            exclude_st=True,
            exclude_bj=True
        )

        # 提取基础列
        base_features = ['open', 'high', 'low', 'close', 'vol', 'pct_chg', 'vwap']
        collected_cols = base_features + ['industry_code']
        available = [c for c in collected_cols if c in processor.df.columns]
        df = processor.df[available].copy()

        # 移除索引（方便后续分组操作）
        df = df.reset_index()

        # 技术指标（使用 DataProcessor 的方法，需要先设置 processor.df）
        old_df = processor.df
        processor.df = df
        df = processor.add_technical_indicators(df)
        processor.df = old_df

        # 超额收益率
        df['excess_return'] = df.groupby('trade_date', group_keys=False)['pct_chg'].apply(
            lambda x: x - x.median()
        )

        # ---- 基本面 / 资金流 ----
        use_fund = self.meta.get('use_fundamental', False)
        use_mf = self.meta.get('use_moneyflow', False)

        extra_feature_cols = []

        if use_fund:
            processor.load_fundamental_data()
            if processor.df_fundamental is not None:
                fund_cols = ['trade_date', 'ts_code', 'pe_ttm', 'pb', 'roe']
                avail = [c for c in fund_cols if c in processor.df_fundamental.columns]
                val_cols = [c for c in avail if c not in ['trade_date', 'ts_code']]
                if val_cols:
                    df = df.merge(
                        processor.df_fundamental[avail], on=['trade_date', 'ts_code'], how='left'
                    )
                    for col in val_cols:
                        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
                    extra_feature_cols.extend(val_cols)
                    print(f"[INFO] 合并基本面: {val_cols}")

        if use_mf:
            processor.load_moneyflow_data()
            if processor.df_moneyflow is not None:
                mf_cols = ['trade_date', 'ts_code', 'super_large_net', 'large_net',
                           'medium_net', 'small_net', 'total_turnover']
                avail = [c for c in mf_cols if c in processor.df_moneyflow.columns]
                val_cols = [c for c in avail if c not in ['trade_date', 'ts_code']]
                if val_cols:
                    df = df.merge(
                        processor.df_moneyflow[avail], on=['trade_date', 'ts_code'], how='left'
                    )
                    for col in val_cols:
                        df[col] = df[col].fillna(0)
                    extra_feature_cols.extend(val_cols)
                    print(f"[INFO] 合并资金流: {val_cols}")

            if use_mf and 'super_large_net' in df.columns:
                for col in ['super_large_net', 'large_net', 'medium_net', 'small_net', 'total_turnover']:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
                df['mf_main_force_ratio'] = (df['super_large_net'] + df['large_net']) / (df['total_turnover'] + 1e-10)
                df['mf_retail_ratio'] = (df['medium_net'] + df['small_net']) / (df['total_turnover'] + 1e-10)
                extra_feature_cols.extend(['mf_main_force_ratio', 'mf_retail_ratio'])
                print("[INFO] 添加资金流派生特征: mf_main_force_ratio, mf_retail_ratio")

        # 定义特征列（与 DataProcessor.select_features 完全一致）
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
        self.feature_cols += [c for c in extra_feature_cols if c in df.columns]
        print(f"[INFO] 特征维度: {len(self.feature_cols)}")

        df = df.fillna(0)

        return df, latest_date

    # =========================================================
    # 截面 Z-Score 标准化
    # =========================================================
    def cross_sectional_normalize(self, df):
        """与 DataProcessor.cross_sectional_normalize 一致的截面标准化"""
        df = df.copy()
        for col in self.feature_cols:
            if col in df.columns:
                mean_val = df.groupby('trade_date')[col].transform('mean')
                std_val = df.groupby('trade_date')[col].transform('std')
                df[col] = (df[col] - mean_val) / (std_val + 1e-10)
        return df

    # =========================================================
    # 构建预测序列
    # =========================================================
    def build_prediction_sequences(self, df, target_date):
        """
        为目标日期的每只股票，提取最近 seq_len 天的特征序列。

        返回 records 列表，每项包含:
          {ts_code, close, sequence: (seq_len, n_features)}
        """
        seq_len = self.meta['seq_len']
        records = []

        for code, group in tqdm(df.groupby('ts_code'), desc="构建预测序列"):
            group = group.sort_values('trade_date')
            dates_arr = group['trade_date'].values
            features = group[self.feature_cols].values
            closes = group['close'].values

            target_idx = None
            for i, d in enumerate(dates_arr):
                if str(d) == str(target_date):
                    target_idx = i
                    break

            if target_idx is None or target_idx < seq_len - 1:
                continue

            seq = features[target_idx - seq_len + 1:target_idx + 1]
            if np.isnan(seq).any():
                continue

            records.append({
                'ts_code': code,
                'close': closes[target_idx],
                'sequence': seq.astype(np.float32)
            })

        print(f"[INFO] 为目标日期 {target_date} 构建了 {len(records)} 只股票的序列")
        return records

    # =========================================================
    # 标准化预测序列
    # =========================================================
    def standardize_sequences(self, records):
        """使用训练时保存的 StandardScaler 对序列做时间维标准化"""
        if not records:
            return records

        n_features = len(self.feature_cols)
        for r in records:
            flat = r['sequence'].reshape(-1, n_features)
            flat = self.scaler.transform(flat)
            r['sequence'] = flat.reshape(r['sequence'].shape)
        return records

    # =========================================================
    # 模型推理
    # =========================================================
    def predict(self, records, batch_size=256):
        """
        分批推理，得分 = (pred_hp - pred_op) / label_scale
        """
        if not records:
            return pd.DataFrame()

        X = np.array([r['sequence'] for r in records], dtype=np.float32)
        label_scale = self.meta['label_scale']
        results = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                x_batch = torch.from_numpy(X[start:end]).float().to(self.device)
                pred_op, _, pred_hp = self.model(x_batch)
                pred_op = pred_op.cpu().numpy().flatten()
                pred_hp = pred_hp.cpu().numpy().flatten()
                for j in range(len(pred_op)):
                    results.append({
                        'ts_code': records[start + j]['ts_code'],
                        'close': records[start + j]['close'],
                        'pred_op': pred_op[j] / label_scale,
                        'pred_hp': pred_hp[j] / label_scale,
                        'score': (pred_hp[j] - pred_op[j]) / label_scale
                    })

        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values('score', ascending=False).reset_index(drop=True)
        df_result['rank'] = range(1, len(df_result) + 1)
        df_result['ths_code'] = df_result['ts_code'].apply(lambda x: x.split('.')[0])
        df_result['name'] = df_result['ts_code'].map(self.stock_name_map).fillna(
            df_result['ts_code'].apply(lambda x: x.split('.')[0])
        )
        return df_result

    # =========================================================
    # 辅助：找最新交易日
    # =========================================================
    def _find_latest_trade_date(self):
        daily_dir = os.path.join(self.data_root, 'daily')
        files = [f.replace('.csv', '') for f in os.listdir(daily_dir) if f.endswith('.csv')]
        files = sorted([f for f in files if f.isdigit() and len(f) == 8])
        if not files:
            raise RuntimeError(f"未在 {daily_dir} 找到日期数据文件")
        return files[-1]

    # =========================================================
    # 主流程
    # =========================================================
    def run(self, output_path=None):
        self.load_artifacts()
        self.load_stock_names()

        df, target_date = self.load_and_prepare_data(lookback_days=45)
        df = self.cross_sectional_normalize(df)

        records = self.build_prediction_sequences(df, target_date)
        records = self.standardize_sequences(records)

        result_df = self.predict(records)

        if output_path is None:
            output_path = f'daily_predictions_{target_date}.csv'

        columns = ['rank', 'ts_code', 'ths_code', 'name', 'score',
                   'pred_op', 'pred_hp', 'close']
        result_df[columns].to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\n[INFO] 预测结果已保存到: {output_path}")
        print(f"[INFO] 共 {len(result_df)} 只股票, 得分范围: [{result_df['score'].min():.6f}, {result_df['score'].max():.6f}]")
        print(f"\n{'='*65}")
        print("TOP 20 推荐股票（得分从高到低）")
        print(f"{'='*65}")
        top_cols = ['rank', 'ts_code', 'name', 'score', 'pred_op', 'pred_hp', 'close']
        print(result_df[top_cols].head(20).to_string(index=False))

        return output_path, result_df


def parse_args():
    parser = argparse.ArgumentParser(description='每日 LSTM 股票预测')
    parser.add_argument('--model_dir', type=str, default='./saved_models/production_v1',
                        help='模型目录（包含 best_model.pth, scaler.pkl, model_meta.pkl）')
    parser.add_argument('--data_path', type=str, default='D:/zhw/A股数据',
                        help='A股数据根目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 CSV 路径，不指定则自动命名 daily_predictions_YYYYMMDD.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("每日 LSTM 股票预测引擎")
    print("=" * 65)

    try:
        trader = DailyTrader(model_dir=args.model_dir, data_root=args.data_path)
        trader.run(output_path=args.output)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\n[提示] 模型目录应包含以下文件:")
        print("  - best_model.pth   （模型权重）")
        print("  - scaler.pkl       （StandardScaler，由 train.py 自动保存）")
        print("  - model_meta.pkl   （模型元数据，由 train.py 自动保存）")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
