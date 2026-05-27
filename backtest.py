import argparse
import numpy as np
import pandas as pd
import torch

def main():
    parser = argparse.ArgumentParser(description='股票LSTM收益率预测模型回测')
    # 数据参数
    parser.add_argument('--model_path', type=str, default='./saved_models/best_model.pth',
                        help='训练好的模型路径')
    parser.add_argument('--data_path', type=str, default='D:/zhw/A股数据',
                        help='A股数据路径')
    parser.add_argument('--stock_pool', type=str, default='hs300',
                        help='股票池：hs300 或 all')
    # 日期范围参数
    parser.add_argument('--start_date', type=str, default='20240601',
                        help='数据起始日期（YYYYMMDD格式）')
    parser.add_argument('--end_date', type=str, default='20250630',
                        help='数据结束日期（YYYYMMDD格式）')
    parser.add_argument('--train_start', type=str, default='20240601',
                        help='训练集起始日期（YYYYMMDD格式）')
    parser.add_argument('--train_end', type=str, default='20250331',
                        help='训练集结束日期（YYYYMMDD格式）')
    parser.add_argument('--val_start', type=str, default='20250401',
                        help='验证集起始日期（YYYYMMDD格式）')
    parser.add_argument('--val_end', type=str, default='20250630',
                        help='验证集结束日期（YYYYMMDD格式）')
    # 回测参数
    parser.add_argument('--n', type=int, default=10,
                        help='持仓数量（5-30）', choices=range(5, 31))
    parser.add_argument('--k', type=int, default=2,
                        help='每日调仓数量（1-5）', choices=range(1, 6))
    parser.add_argument('--seq_len', type=int, default=30,
                        help='时间窗口长度（必须与训练时一致）')
    parser.add_argument('--transaction_cost', type=float, default=0.001,
                        help='交易成本率')
    # 模型结构参数（必须与训练时一致）
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='LSTM隐藏层维度（必须与训练时一致）')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='LSTM层数（必须与训练时一致）')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout概率（必须与训练时一致）')
    parser.add_argument('--fc_dropout', type=float, default=0.5,
                        help='输出层Dropout概率（必须与训练时一致）')
    parser.add_argument('--bidirectional', action='store_true',
                        help='是否使用双向LSTM（必须与训练时一致）')
    parser.add_argument('--label_scale', type=float, default=100.0,
                        help='标签缩放因子（必须与训练时一致）')
    parser.add_argument('--use_fundamental', action='store_true', default=False,
                        help='是否使用基本面数据（必须与训练时一致）')
    parser.add_argument('--use_moneyflow', action='store_true', default=False,
                        help='是否使用资金流数据（必须与训练时一致）')
    args = parser.parse_args()

    print("="*60)
    print("股票LSTM收益率预测模型回测系统")
    print("="*60)
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.data_path}")
    print(f"股票池: {args.stock_pool}")
    print(f"数据范围: {args.start_date} ~ {args.end_date}")
    print(f"训练集: {args.train_start} ~ {args.train_end}")
    print(f"验证集: {args.val_start} ~ {args.val_end}")
    print(f"持仓数量n: {args.n}")
    print(f"每日调仓数量k: {args.k}")
    print(f"时间窗口: {args.seq_len}")
    print(f"交易成本率: {args.transaction_cost*100:.2f}%")
    print(f"LSTM隐藏层维度: {args.hidden_dim}")
    print(f"LSTM层数: {args.num_layers}")
    print(f"Dropout概率: {args.dropout}")
    print(f"FC Dropout概率: {args.fc_dropout}")
    print(f"是否双向LSTM: {'是' if args.bidirectional else '否'}")
    print(f"标签缩放因子: {args.label_scale}")
    print(f"使用基本面数据: {'是' if args.use_fundamental else '否'}")
    print(f"使用资金流数据: {'是' if args.use_moneyflow else '否'}")
    print("="*60)

    # 加载数据
    print("\n[步骤1] 加载数据...")
    try:
        from data_processor import DataProcessor

        data_start = min(args.train_start, args.val_start)
        data_end = max(args.train_end, args.val_end)
        print(f"[INFO] 自动推导数据加载范围: {data_start} ~ {data_end}")
        print(f"[INFO] 训练区间: {args.train_start} ~ {args.train_end}")
        print(f"[INFO] 验证区间: {args.val_start} ~ {args.val_end}")

        processor = DataProcessor(data_root=args.data_path)
        X_train, y_op_train, y_lp_train, y_hp_train, dates_train, stocks_train, current_close_train, \
        X_val, y_op_val, y_lp_val, y_hp_val, dates_val, stocks_val, current_close_val = processor.run_pipeline(
            start_date=data_start,
            end_date=data_end,
            stock_pool=args.stock_pool,
            window_len=args.seq_len,
            horizon=1,
            train_range=(args.train_start, args.train_end),
            val_range=(args.val_start, args.val_end),
            use_fundamental=args.use_fundamental,
            use_moneyflow=args.use_moneyflow
        )
        
        print(f"训练集样本数: {len(X_train)}")
        print(f"验证集样本数: {len(X_val)}")
        input_dim = X_val.shape[-1] if len(X_val) > 0 else (X_train.shape[-1] if len(X_train) > 0 else 33)
        print(f"特征维度: {input_dim}")
        
    except Exception as e:
        print(f"[ERROR] 数据加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 加载模型并预测
    print("\n[步骤2] 加载模型并预测...")
    try:
        from strategy import TradingStrategy
        
        # 使用命令行传入的模型参数，确保与训练时一致
        strategy = TradingStrategy(
            model_path=args.model_path,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            fc_dropout=args.fc_dropout,
            bidirectional=args.bidirectional,
            label_scale=args.label_scale
        )
        
        success = strategy.load_model(input_dim=input_dim)
        if not success:
            print("[ERROR] 模型加载失败")
            print("[提示] 请检查以下参数是否与训练时一致：")
            print(f"       - hidden_dim: 当前设置为 {args.hidden_dim}")
            print(f"       - num_layers: 当前设置为 {args.num_layers}")
            print(f"       - dropout: 当前设置为 {args.dropout}")
            print(f"       - fc_dropout: 当前设置为 {args.fc_dropout}")
            print(f"       - bidirectional: 当前设置为 {args.bidirectional}")
            print(f"       - seq_len: 当前设置为 {args.seq_len}")
            print(f"       - label_scale: 当前设置为 {args.label_scale}")
            print(f"       - use_fundamental: 当前设置为 {args.use_fundamental}")
            print(f"       - use_moneyflow: 当前设置为 {args.use_moneyflow}")
            print("[提示] 如果您使用了不同的参数训练模型，请使用以下命令格式：")
            print("       python backtest.py --model_path ./saved_models/best_model.pth ")
            print("                          --hidden_dim 128 --num_layers 2 --dropout 0.3 --bidirectional --label_scale 100.0")
            return
        
        # 对验证集进行预测
        scores, predictions = strategy.predict_scores(X_val)
        if scores is None:
            print("[ERROR] 预测失败")
            return
        
        print(f"预测收益率范围: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"预测收益率均值: {scores.mean():.4f}")
        
    except Exception as e:
        print(f"[ERROR] 模型预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 运行回测
    print("\n[步骤3] 运行回测...")
    try:
        from strategy import BackTester
        
        backtester = BackTester(
            n=args.n,
            k=args.k,
            transaction_cost=args.transaction_cost
        )
        
        # 准备回测数据
        # 使用实际开盘价变化率作为标签，当前收盘价作为买入成本
        metrics = backtester.run_backtest(
            dates=dates_val,
            stocks=stocks_val,
            scores=scores,
            actual_op=y_op_val.flatten(),
            current_close=current_close_val.flatten()
        )
        
        # 打印回测结果
        backtester.print_metrics(metrics)
        
        # 保存回测结果
        print("\n[步骤4] 保存回测结果...")
        results_df = pd.DataFrame({
            '日期': dates_val,
            '股票代码': stocks_val,
            '预测得分': scores,
            '预测op': predictions['op'],
            '预测lp': predictions['lp'],
            '预测hp': predictions['hp'],
            '实际op': y_op_val.flatten(),
            '实际lp': y_lp_val.flatten(),
            '实际hp': y_hp_val.flatten(),
            '当前收盘价': current_close_val.flatten()
        })
        
        results_df.to_csv('./backtest_results.csv', index=False, encoding='utf-8-sig')
        print("回测详情已保存到: ./backtest_results.csv")
        
        # 保存每日组合价值
        portfolio_df = pd.DataFrame({
            '日期': dates_val[:len(backtester.daily_portfolio_values)],
            '组合价值': backtester.daily_portfolio_values,
            '每日收益': [0] + backtester.daily_returns
        })
        portfolio_df.to_csv('./portfolio_values.csv', index=False, encoding='utf-8-sig')
        print("组合价值曲线已保存到: ./portfolio_values.csv")
        
        # 保存指标汇总
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('./backtest_metrics.csv', index=False, encoding='utf-8-sig')
        print("回测指标已保存到: ./backtest_metrics.csv")
        
    except Exception as e:
        print(f"[ERROR] 回测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("回测完成！")
    print("="*60)

if __name__ == "__main__":
    main()
