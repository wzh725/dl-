import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from data_processor import DataProcessor
from model import AssociatedResidualNet, JointDirectionalLoss, ListNetLoss


class CompositeLoss(nn.Module):
    """
    复合损失函数：回归损失 + 排序损失

    Total_Loss = Loss_regression + lambda * Loss_ranking

    其中：
    - Loss_regression: JointDirectionalLoss（三个分支的方向回归损失）
    - Loss_ranking: ListNetLoss（截面排序损失）
    - lambda: 排序损失权重（默认1.0）
    """

    def __init__(self, regression_loss, ranking_loss, ranking_lambda=1.0):
        super(CompositeLoss, self).__init__()
        self.regression_loss = regression_loss
        self.ranking_loss = ranking_loss
        self.ranking_lambda = ranking_lambda

    def forward(self, pred_op, pred_lp, pred_hp, true_op, true_lp, true_hp):
        """
        计算复合损失
        """
        # 回归损失
        loss_regression = self.regression_loss(pred_op, pred_lp, pred_hp, true_op, true_lp, true_hp)

        # 排序损失
        loss_ranking = self.ranking_loss(pred_op, pred_hp, true_op, true_hp)

        # 总损失
        total_loss = loss_regression + self.ranking_lambda * loss_ranking

        return total_loss


def parse_args():
    parser = argparse.ArgumentParser(description='训练多值关联残差神经网络（AssociatedResidualNet）')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='D:/zhw/A股数据', help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--seq_len', type=int, default=30, help='时间序列长度/窗口大小')
    parser.add_argument('--horizon', type=int, default=1, help='预测步长')

    # 模型参数
    parser.add_argument('--input_dim', type=int, default=34, help='原始输入特征维度')
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout概率')
    parser.add_argument('--bidirectional', action='store_true', help='是否使用双向LSTM')

    # 训练参数
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--adamw', action='store_true', help='是否使用AdamW优化器')
    parser.add_argument('--penalty_weight', type=float, default=1.5, help='方向性惩罚权重')
    parser.add_argument('--ranking_lambda', type=float, default=1.0, help='ListNet排序损失权重')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--label_scale', type=float, default=100.0, help='标签缩放因子')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='模型保存目录')

    # 数据范围参数
    parser.add_argument('--start_date', type=str, default='2024-06-01', help='数据起始日期')
    parser.add_argument('--end_date', type=str, default='2025-06-30', help='数据结束日期')
    parser.add_argument('--train_start', type=str, default='2024-06-01', help='训练集起始日期')
    parser.add_argument('--train_end', type=str, default='2025-03-31', help='训练集结束日期')
    parser.add_argument('--val_start', type=str, default='2025-04-01', help='验证集起始日期')
    parser.add_argument('--val_end', type=str, default='2025-06-30', help='验证集结束日期')

    # 股票池
    parser.add_argument('--stock_pool', type=str, default='hs300', help='股票池: hs300/all')

    return parser.parse_args()


def load_real_data(args):
    """从真实数据加载训练/验证数据"""
    print(f"加载真实数据...")
    print(f"数据路径: {args.data_path}")
    print(f"股票池: {args.stock_pool}")

    # 创建数据处理器
    processor = DataProcessor(data_root=args.data_path)

    # 运行数据处理流程
    X_train, y_op_train, y_lp_train, y_hp_train, _, _, _, \
        X_val, y_op_val, y_lp_val, y_hp_val, _, _, _ = processor.run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        stock_pool=args.stock_pool,
        window_len=args.seq_len,
        horizon=args.horizon,
        train_range=(args.train_start, args.train_end),
        val_range=(args.val_start, args.val_end)
    )

    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")
    print(f"特征维度: {X_train.shape[-1]}")

    # 更新实际的特征维度
    args.input_dim = X_train.shape[-1]

    return X_train, y_op_train, y_lp_train, y_hp_train, X_val, y_op_val, y_lp_val, y_hp_val


def create_dataloaders(args, X_train, y_op_train, y_lp_train, y_hp_train, X_val, y_op_val, y_lp_val, y_hp_val):
    """创建训练和验证的DataLoader"""
    # 训练集
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_op_train).float(),
        torch.from_numpy(y_lp_train).float(),
        torch.from_numpy(y_hp_train).float()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 验证集
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_op_val).float(),
        torch.from_numpy(y_lp_val).float(),
        torch.from_numpy(y_hp_val).float()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", position=0, leave=True)

    for batch_idx, batch in pbar:
        x, y_op, y_lp, y_hp = batch

        # 将数据移到设备上
        x = x.to(device)
        y_op = y_op.to(device)
        y_lp = y_lp.to(device)
        y_hp = y_hp.to(device)

        # 前向传播
        pred_op, pred_lp, pred_hp = model(x)

        # 计算复合损失
        loss = criterion(pred_op, pred_lp, pred_hp, y_op, y_lp, y_hp)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 累加损失
        total_loss += loss.item() * x.size(0)

        # 更新进度条描述
        pbar.set_description(f"Training [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # 计算平均损失
    n_samples = len(train_loader.dataset)
    avg_loss = total_loss / n_samples

    return avg_loss


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", position=0, leave=True)

    for batch_idx, batch in pbar:
        x, y_op, y_lp, y_hp = batch

        # 将数据移到设备上
        x = x.to(device)
        y_op = y_op.to(device)
        y_lp = y_lp.to(device)
        y_hp = y_hp.to(device)

        # 前向传播
        pred_op, pred_lp, pred_hp = model(x)

        # 计算损失
        loss = criterion(pred_op, pred_lp, pred_hp, y_op, y_lp, y_hp)

        # 累加损失
        total_loss += loss.item() * x.size(0)

        # 更新进度条描述
        pbar.set_description(f"Validating [{batch_idx+1}/{len(val_loader)}] Loss: {loss.item():.4f}")

    # 计算平均损失
    n_samples = len(val_loader.dataset)
    avg_loss = total_loss / n_samples

    return avg_loss


def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    print(f"训练参数: {args}")
    print(f"使用设备: {args.device}")

    # 加载真实数据
    X_train, y_op_train, y_lp_train, y_hp_train, X_val, y_op_val, y_lp_val, y_hp_val = load_real_data(args)

    # 创建DataLoader
    train_loader, val_loader = create_dataloaders(
        args, X_train, y_op_train, y_lp_train, y_hp_train,
        X_val, y_op_val, y_lp_val, y_hp_val
    )

    # 创建模型
    model = AssociatedResidualNet(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    ).to(args.device)

    # 打印模型结构
    print(f"\n模型结构:\n{model}")

    # 配置损失函数：回归损失 + 排序损失
    regression_loss = JointDirectionalLoss(penalty_weight=args.penalty_weight, label_scale=args.label_scale)
    ranking_loss = ListNetLoss(label_scale=args.label_scale)
    criterion = CompositeLoss(regression_loss, ranking_loss, ranking_lambda=args.ranking_lambda)
    print(f"使用复合损失函数：回归损失 + {args.ranking_lambda} * 排序损失")
    print(f"标签缩放因子: {args.label_scale}")

    # 配置优化器
    if args.adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print("使用 AdamW 优化器")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print("使用 Adam 优化器")

    # 配置学习率调度器（余弦退火）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print("使用余弦退火学习率调度器")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    best_val_loss = float('inf')
    early_stopping_count = 0
    early_stopping_patience = args.early_stopping_patience
    print(f"\n开始训练，早停耐心值: {early_stopping_patience}")
    print("="*60)

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)

        # 验证
        val_loss = validate(model, val_loader, criterion, args.device)

        # 更新学习率
        scheduler.step()

        # 打印日志
        print(f"训练损失: {train_loss:.6f}")
        print(f"验证损失: {val_loss:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_count = 0
            model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"保存最佳模型: {model_path}")
        else:
            early_stopping_count += 1
            print(f"早停计数: {early_stopping_count}/{early_stopping_patience}")

            if early_stopping_count >= early_stopping_patience:
                print("触发早停，停止训练")
                break

    print("\n" + "="*60)
    print(f"训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
