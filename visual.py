import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from model import AssociatedNet
from data_processor import DataProcessor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化模型预测结果')
    
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径(.pth文件)')
    parser.add_argument('--data_path', type=str, default='D:/zhw/A股数据', help='数据目录路径')
    parser.add_argument('--save_dir', type=str, default='./visual_results', help='可视化结果保存目录')
    
    # 数据参数
    parser.add_argument('--seq_len', type=int, default=30, help='时间序列长度')
    parser.add_argument('--horizon', type=int, default=1, help='预测步长')
    
    # 模型参数（需要与训练时一致）
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout概率')
    
    # 数据范围
    parser.add_argument('--start_date', type=str, default='2024-06-01', help='数据起始日期')
    parser.add_argument('--end_date', type=str, default='2025-06-30', help='数据结束日期')
    parser.add_argument('--val_start', type=str, default='2025-04-01', help='验证集起始日期')
    parser.add_argument('--val_end', type=str, default='2025-06-30', help='验证集结束日期')
    parser.add_argument('--stock_pool', type=str, default='hs300', help='股票池')
    
    return parser.parse_args()


def load_model(args, input_dim):
    """加载训练好的模型"""
    model = AssociatedNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    
    print(f"模型加载成功: {args.model_path}")
    return model


def load_val_data(args):
    """加载验证数据"""
    processor = DataProcessor(data_root=args.data_path)
    
    # 运行数据处理流程，只获取验证集
    X_train, _, _, _, X_val, y_op_val, y_lp_val, y_hp_val = processor.run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        stock_pool=args.stock_pool,
        window_len=args.seq_len,
        horizon=args.horizon,
        train_range=(args.start_date, args.val_start),
        val_range=(args.val_start, args.val_end)
    )
    
    return X_val, y_op_val, y_lp_val, y_hp_val, processor


def predict(model, X_val):
    """使用模型进行预测"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_val).float()
        op_pred, lp_pred, hp_pred = model(X_tensor)
    
    return op_pred.numpy(), lp_pred.numpy(), hp_pred.numpy()


def normalize_labels(y_op, y_lp, y_hp):
    """
    对标签进行标准化处理，使其与特征数据尺度一致
    使用Z-score标准化
    """
    # 合并三个标签计算统计量
    all_labels = np.concatenate([y_op, y_lp, y_hp])
    
    mean = np.mean(all_labels)
    std = np.std(all_labels)
    
    # 标准化
    y_op_norm = (y_op - mean) / (std + 1e-8)
    y_lp_norm = (y_lp - mean) / (std + 1e-8)
    y_hp_norm = (y_hp - mean) / (std + 1e-8)
    
    return y_op_norm, y_lp_norm, y_hp_norm, mean, std


def denormalize_predictions(y_pred, mean, std):
    """将预测值反标准化回原始尺度"""
    return y_pred * std + mean


def plot_predictions(y_true, y_pred, label_name, save_path):
    """绘制预测值与真实值的对比图"""
    plt.figure(figsize=(12, 6))
    
    # 绘制真实值和预测值
    plt.plot(y_true[:200], label=f'{label_name} True', color='blue', alpha=0.7, linewidth=2)
    plt.plot(y_pred[:200], label=f'{label_name} Predicted', color='red', alpha=0.7, linewidth=2)
    
    plt.title(f'{label_name} Prediction vs True', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Normalized Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_scatter(y_true, y_pred, label_name, save_path):
    """绘制散点图（真实值 vs 预测值）"""
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal Line', linewidth=2)
    
    plt.title(f'{label_name} True vs Predicted', fontsize=14)
    plt.xlabel('True Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_histogram(y_true, y_pred, label_name, save_path):
    """绘制预测误差直方图"""
    errors = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    
    plt.title(f'{label_name} Prediction Error Distribution', fontsize=14)
    plt.xlabel('Error (Predicted - True)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.text(0.95, 0.95, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}', 
             ha='right', va='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_price_range(y_op_true, y_lp_true, y_hp_true, y_op_pred, y_lp_pred, y_hp_pred, save_path):
    """绘制价格区间预测图（开盘价、最低价、最高价）"""
    plt.figure(figsize=(12, 6))
    
    # 真实值
    plt.plot(y_hp_true[:100], label='HP True', color='red', linestyle='--', linewidth=2)
    plt.plot(y_op_true[:100], label='OP True', color='blue', linestyle='--', linewidth=2)
    plt.plot(y_lp_true[:100], label='LP True', color='green', linestyle='--', linewidth=2)
    
    # 预测值
    plt.plot(y_hp_pred[:100], label='HP Predicted', color='red', alpha=0.7, linewidth=1.5)
    plt.plot(y_op_pred[:100], label='OP Predicted', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(y_lp_pred[:100], label='LP Predicted', color='green', alpha=0.7, linewidth=1.5)
    
    plt.title('Price Range Prediction Comparison', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Normalized Price', fontsize=12)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def print_metrics(op_metrics, lp_metrics, hp_metrics):
    """打印评估指标"""
    print("\n" + "="*50)
    print("Model Evaluation Metrics")
    print("="*50)
    
    print("\nOpen Price (OP):")
    print(f"  MSE:  {op_metrics['MSE']:.6f}")
    print(f"  MAE:  {op_metrics['MAE']:.6f}")
    print(f"  RMSE: {op_metrics['RMSE']:.6f}")
    print(f"  MAPE: {op_metrics['MAPE']:.4f}%")
    
    print("\nLow Price (LP):")
    print(f"  MSE:  {lp_metrics['MSE']:.6f}")
    print(f"  MAE:  {lp_metrics['MAE']:.6f}")
    print(f"  RMSE: {lp_metrics['RMSE']:.6f}")
    print(f"  MAPE: {lp_metrics['MAPE']:.4f}%")
    
    print("\nHigh Price (HP):")
    print(f"  MSE:  {hp_metrics['MSE']:.6f}")
    print(f"  MAE:  {hp_metrics['MAE']:.6f}")
    print(f"  RMSE: {hp_metrics['RMSE']:.6f}")
    print(f"  MAPE: {hp_metrics['MAPE']:.4f}%")
    
    print("\n" + "="*50)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载验证数据
    print("Loading validation data...")
    X_val, y_op_val, y_lp_val, y_hp_val, processor = load_val_data(args)
    
    # 对标签进行标准化处理，使其与特征数据尺度一致
    print("Normalizing labels...")
    y_op_norm, y_lp_norm, y_hp_norm, mean, std = normalize_labels(y_op_val, y_lp_val, y_hp_val)
    
    # 获取输入特征维度
    input_dim = X_val.shape[-1]
    
    # 加载模型
    model = load_model(args, input_dim)
    
    # 进行预测
    print("Making predictions...")
    op_pred_raw, lp_pred_raw, hp_pred_raw = predict(model, X_val)
    
    # 将预测值标准化（因为模型输出与标准化后的特征尺度一致）
    op_pred = (op_pred_raw - mean) / (std + 1e-8)
    lp_pred = (lp_pred_raw - mean) / (std + 1e-8)
    hp_pred = (hp_pred_raw - mean) / (std + 1e-8)
    
    # 计算评估指标（使用标准化后的值）
    op_metrics = calculate_metrics(y_op_norm, op_pred)
    lp_metrics = calculate_metrics(y_lp_norm, lp_pred)
    hp_metrics = calculate_metrics(y_hp_norm, hp_pred)
    
    # 打印评估指标
    print_metrics(op_metrics, lp_metrics, hp_metrics)
    
    # 生成可视化图表
    print("\nGenerating visualization charts...")
    
    # 开盘价可视化
    plot_predictions(y_op_norm, op_pred, 'Open Price (OP)', os.path.join(args.save_dir, 'op_prediction.png'))
    plot_scatter(y_op_norm, op_pred, 'Open Price (OP)', os.path.join(args.save_dir, 'op_scatter.png'))
    plot_histogram(y_op_norm, op_pred, 'Open Price (OP)', os.path.join(args.save_dir, 'op_error.png'))
    
    # 最低价可视化
    plot_predictions(y_lp_norm, lp_pred, 'Low Price (LP)', os.path.join(args.save_dir, 'lp_prediction.png'))
    plot_scatter(y_lp_norm, lp_pred, 'Low Price (LP)', os.path.join(args.save_dir, 'lp_scatter.png'))
    plot_histogram(y_lp_norm, lp_pred, 'Low Price (LP)', os.path.join(args.save_dir, 'lp_error.png'))
    
    # 最高价可视化
    plot_predictions(y_hp_norm, hp_pred, 'High Price (HP)', os.path.join(args.save_dir, 'hp_prediction.png'))
    plot_scatter(y_hp_norm, hp_pred, 'High Price (HP)', os.path.join(args.save_dir, 'hp_scatter.png'))
    plot_histogram(y_hp_norm, hp_pred, 'High Price (HP)', os.path.join(args.save_dir, 'hp_error.png'))
    
    # 价格区间综合对比
    plot_price_range(y_op_norm, y_lp_norm, y_hp_norm, op_pred, lp_pred, hp_pred, 
                     os.path.join(args.save_dir, 'price_range.png'))
    
    print(f"\nAll visualization results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
