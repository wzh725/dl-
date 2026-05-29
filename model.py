import torch
import torch.nn as nn


class ResidualLSTMBlock(nn.Module):
    """
    残差LSTM模块：
    - LSTM -> Dropout -> LayerNorm -> 残差连接
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.3,
        bidirectional=False
    ):
        super(ResidualLSTMBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # LayerNorm
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # 如果输入维度不等于输出维度，需要投影
        if input_dim != lstm_output_dim:
            self.projection = nn.Linear(input_dim, lstm_output_dim)
        else:
            self.projection = None

    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            out: (batch, seq_len, lstm_output_dim)
        """
        # LSTM输出
        lstm_out, _ = self.lstm(x)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 残差连接
        if self.projection is not None:
            residual = self.projection(x)
        else:
            residual = x

        # LayerNorm
        out = self.layer_norm(lstm_out + residual)

        return out


class AssociatedResidualNet(nn.Module):
    """
    多值关联残差神经网络（AssociatedResidualNet）
    架构：
    - 分支1（OP分支）：输入原始特征，输出pred_op
    - 分支2（LP分支）：输入[原始特征 + pred_op级联]，输出pred_lp
    - 分支3（HP分支）：输入[原始特征 + pred_op + pred_lp级联]，输出pred_hp
    每个分支使用ResidualLSTMBlock实现
    """

    def __init__(
        self,
        input_dim=30,
        hidden_dim=64,
        num_layers=3,
        dropout=0.3,
        bidirectional=False
    ):
        super(AssociatedResidualNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # LSTM输出维度
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # ========== 分支1：OP分支 ==========
        # 使用多个ResidualLSTMBlock
        self.op_blocks = nn.ModuleList()
        # 第一层：输入原始特征
        self.op_blocks.append(
            ResidualLSTMBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=dropout,
                bidirectional=bidirectional
            )
        )
        # 后续层：输入等于上一层输出
        for _ in range(num_layers - 1):
            self.op_blocks.append(
                ResidualLSTMBlock(
                    input_dim=lstm_out_dim,
                    hidden_dim=hidden_dim,
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=bidirectional
                )
            )
        # OP输出层
        self.op_fc = nn.Linear(lstm_out_dim, 1)

        # ========== 分支2：LP分支 ==========
        # 输入维度 = 原始特征 + OP预测输出
        lp_input_dim = input_dim + 1
        self.lp_blocks = nn.ModuleList()
        self.lp_blocks.append(
            ResidualLSTMBlock(
                input_dim=lp_input_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=dropout,
                bidirectional=bidirectional
            )
        )
        for _ in range(num_layers - 1):
            self.lp_blocks.append(
                ResidualLSTMBlock(
                    input_dim=lstm_out_dim,
                    hidden_dim=hidden_dim,
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=bidirectional
                )
            )
        # LP输出层
        self.lp_fc = nn.Linear(lstm_out_dim, 1)

        # ========== 分支3：HP分支 ==========
        # 输入维度 = 原始特征 + OP预测输出 + LP预测输出
        hp_input_dim = input_dim + 2
        self.hp_blocks = nn.ModuleList()
        self.hp_blocks.append(
            ResidualLSTMBlock(
                input_dim=hp_input_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=dropout,
                bidirectional=bidirectional
            )
        )
        for _ in range(num_layers - 1):
            self.hp_blocks.append(
                ResidualLSTMBlock(
                    input_dim=lstm_out_dim,
                    hidden_dim=hidden_dim,
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=bidirectional
                )
            )
        # HP输出层
        self.hp_fc = nn.Linear(lstm_out_dim, 1)

    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch, seq_len, input_dim) 原始特征
        Returns:
            pred_op: (batch, 1) 预测的开盘价变化率
            pred_lp: (batch, 1) 预测的最低价变化率
            pred_hp: (batch, 1) 预测的最高价变化率
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # ========== 分支1：OP分支 ==========
        op_out = x
        for block in self.op_blocks:
            op_out = block(op_out)
        # 取最后一个时间步
        op_last = op_out[:, -1, :]
        pred_op = self.op_fc(op_last)

        # ========== 分支2：LP分支 ==========
        # 构建LP输入：原始特征 + OP预测（在时间维度上广播）
        # pred_op: (batch, 1) -> (batch, seq_len, 1)
        pred_op_expanded = pred_op.unsqueeze(1).expand(-1, seq_len, 1)
        lp_input = torch.cat([x, pred_op_expanded], dim=-1)

        lp_out = lp_input
        for block in self.lp_blocks:
            lp_out = block(lp_out)
        lp_last = lp_out[:, -1, :]
        pred_lp = self.lp_fc(lp_last)

        # ========== 分支3：HP分支 ==========
        # 构建HP输入：原始特征 + OP预测 + LP预测
        pred_lp_expanded = pred_lp.unsqueeze(1).expand(-1, seq_len, 1)
        hp_input = torch.cat([x, pred_op_expanded, pred_lp_expanded], dim=-1)

        hp_out = hp_input
        for block in self.hp_blocks:
            hp_out = block(hp_out)
        hp_last = hp_out[:, -1, :]
        pred_hp = self.hp_fc(hp_last)

        return pred_op, pred_lp, pred_hp


class ListNetLoss(nn.Module):
    """
    ListNet损失函数：用于优化截面排序
    核心思想：将同一截面内的"预测得分序列"与"真实得分序列"分别进行Top-N Softmax转换，
    然后计算它们之间的交叉熵损失

    公式：
    - 预测得分：pred_score = pred_hp - pred_op
    - 真实得分：true_score = true_hp - true_op (缩放后)
    - P(true) = softmax(true_score) / sum(softmax(true_score))
    - P(pred) = softmax(pred_score) / sum(softmax(pred_score))
    - Loss = -sum(P(true) * log(P(pred)))
    """

    def __init__(self, top_k=None, label_scale=100.0):
        """
        Args:
            top_k: 截取前k个样本进行计算，None表示使用全截面
            label_scale: 标签缩放因子，用于匹配模型输出范围
        """
        super(ListNetLoss, self).__init__()
        self.top_k = top_k
        self.label_scale = label_scale
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred_op, pred_hp, true_op, true_hp):
        """
        计算ListNet排序损失
        Args:
            pred_op: (batch, 1) 预测开盘价变化率
            pred_hp: (batch, 1) 预测最高价变化率
            true_op: (batch, 1) 真实开盘价变化率
            true_hp: (batch, 1) 真实最高价变化率
        Returns:
            loss: 标量损失
        """
        # 计算预测得分和真实得分
        pred_score = pred_hp.squeeze(-1) - pred_op.squeeze(-1)  # (batch,)
        true_score = (true_hp.squeeze(-1) - true_op.squeeze(-1)) * self.label_scale  # (batch,) 缩放

        # 获取有效样本（排除NaN）
        valid_mask = ~(torch.isnan(pred_score) | torch.isnan(true_score))
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_op.device)

        pred_score = pred_score[valid_mask]
        true_score = true_score[valid_mask]

        # 如果样本数小于2，无法计算有效的排序损失
        if len(pred_score) < 2:
            return torch.tensor(0.0, device=pred_op.device)

        # Top-K选择
        if self.top_k is not None and self.top_k < len(pred_score):
            # 选择得分最高的top_k个样本
            _, top_indices = torch.topk(true_score, min(self.top_k, len(true_score)))
            pred_score = pred_score[top_indices]
            true_score = true_score[top_indices]

        # Softmax转换
        pred_probs = torch.softmax(pred_score, dim=0)
        true_probs = torch.softmax(true_score, dim=0)

        # 交叉熵损失
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10))

        return loss


class JointDirectionalLoss(nn.Module):
    """
    多任务联合损失 + 方向性惩罚
    - 联合损失：(Loss_op + Loss_lp + Loss_hp) / 3.0
    - 方向性惩罚：如果预测的y_hp或y_op的正负符号与实际相反，将该样本的MSE乘以惩罚权重（如1.5）

    注意：标签值是变化率（约0.01级别），需要对标签进行缩放以匹配模型输出范围
    """

    def __init__(self, penalty_weight=1.5, label_scale=100.0):
        super(JointDirectionalLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.penalty_weight = penalty_weight
        self.label_scale = label_scale

    def forward(self, pred_op, pred_lp, pred_hp, true_op, true_lp, true_hp):
        """
        计算多任务联合损失
        Args:
            pred_op, pred_lp, pred_hp: (batch, 1) 预测值
            true_op, true_lp, true_hp: (batch, 1) 真实值
        Returns:
            total_loss: 标量损失
        """
        # 对标签进行缩放，使其与模型输出范围匹配
        # 标签是变化率（约0.01），缩放100倍后约为1.0
        true_op_scaled = true_op * self.label_scale
        true_lp_scaled = true_lp * self.label_scale
        true_hp_scaled = true_hp * self.label_scale

        # 计算每个样本的MSE（使用缩放后的标签）
        loss_op = self.mse(pred_op, true_op_scaled)
        loss_lp = self.mse(pred_lp, true_lp_scaled)
        loss_hp = self.mse(pred_hp, true_hp_scaled)

        # 计算方向性惩罚：检查OP和HP的预测方向是否正确
        # 使用原始标签（未缩放）来判断方向
        op_direction_correct = (torch.sign(pred_op) == torch.sign(true_op)).float()
        hp_direction_correct = (torch.sign(pred_hp) == torch.sign(true_hp)).float()

        # 构建惩罚系数：如果OP或HP方向错误，施加惩罚
        # 两个方向都正确：penalty = 1.0
        # 任意一个错误：penalty = penalty_weight
        penalty = torch.ones_like(loss_op)
        penalty[(op_direction_correct < 0.5) | (hp_direction_correct < 0.5)] = self.penalty_weight

        # 应用惩罚
        loss_op_penalized = loss_op * penalty
        loss_lp_penalized = loss_lp * penalty
        loss_hp_penalized = loss_hp * penalty

        # 平均每个样本的损失（按任务平均）
        mean_loss_op = loss_op_penalized.mean()
        mean_loss_lp = loss_lp_penalized.mean()
        mean_loss_hp = loss_hp_penalized.mean()

        # 多任务联合损失
        total_loss = (mean_loss_op + mean_loss_lp + mean_loss_hp) / 3.0

        return total_loss


# =========================================================
# 运行示例：使用虚拟数据进行前向传播和Loss计算
# =========================================================
if __name__ == "__main__":
    # 超参数设置
    input_dim = 30
    hidden_dim = 64
    num_layers = 3
    dropout = 0.3
    bidirectional = True
    batch_size = 32
    seq_len = 30

    # 创建模型实例
    model = AssociatedResidualNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )

    # 创建虚拟输入数据 (batch_size, seq_len, input_dim)
    x_dummy = torch.randn(batch_size, seq_len, input_dim)

    # 创建虚拟目标标签
    op_target = torch.randn(batch_size, 1) * 0.05
    lp_target = torch.randn(batch_size, 1) * 0.05
    hp_target = torch.randn(batch_size, 1) * 0.05

    # 前向传播
    pred_op, pred_lp, pred_hp = model(x_dummy)

    # 计算损失
    criterion = JointDirectionalLoss(penalty_weight=1.5)
    loss = criterion(pred_op, pred_lp, pred_hp, op_target, lp_target, hp_target)

    # 输出结果
    print(f"模型结构:\n{model}")
    print(f"\n输入形状: {x_dummy.shape}")
    print(f"预测输出形状: {pred_op.shape}, {pred_lp.shape}, {pred_hp.shape}")
    print(f"\n多任务联合损失: {loss.item():.6f}")

    # 展示优化器绑定
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 演示训练步骤
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print("\n优化器已成功绑定并执行反向传播")
