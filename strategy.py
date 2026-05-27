import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from collections import defaultdict

from model import AssociatedResidualNet


class TradingStrategy:
    """
    交易策略类：基于多值关联残差神经网络预测并计算得分
    """

    def __init__(
        self,
        model_path,
        hidden_dim=64,
        num_layers=3,
        dropout=0.3,
        fc_dropout=0.5,
        bidirectional=False,
        label_scale=100.0
    ):
        self.model_path = model_path
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.bidirectional = bidirectional
        self.label_scale = label_scale
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None

    def load_model(self, input_dim=30):
        try:
            self.model = AssociatedResidualNet(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                fc_dropout=self.fc_dropout,
                bidirectional=self.bidirectional
            ).to(self.device)

            checkpoint = torch.load(self.model_path, map_location=self.device)
            # 兼容两种保存格式：直接保存state_dict或保存包含model_state_dict的字典
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"[INFO] 模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def predict_scores(self, X, batch_size=128):
        """
        预测并计算得分（支持批处理，解决内存不足问题）
        得分公式：score = pred_hp - pred_op（预期做多空间）
        【修复】原公式 (pred_hp - pred_op) / (pred_op + eps) 存在分母接近0或负数时得分爆炸问题
        【修复】添加标签缩放因子的逆缩放，确保预测值与实际值尺度匹配

        Args:
            X: 特征数据 (N, seq_len, input_dim)
            batch_size: 批处理大小

        Returns:
            scores: 收益得分 (N,)
            predictions: 包含所有预测值的字典（已逆缩放）
        """
        if self.model is None:
            print("[ERROR] 模型未加载，请先调用 load_model()")
            return None, None

        try:
            self.model.eval()

            n_samples = X.shape[0]
            pred_op_list = []
            pred_lp_list = []
            pred_hp_list = []

            num_batches = (n_samples + batch_size - 1) // batch_size
            print(f"[INFO] 开始分批预测，共 {num_batches} 批，每批 {batch_size} 样本")
            print(f"[INFO] 使用标签缩放因子: {self.label_scale}（用于逆缩放预测值）")

            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_samples)
                X_batch = X[start:end]

                with torch.no_grad():
                    X_tensor = torch.from_numpy(X_batch).float().to(self.device)
                    pred_op, pred_lp, pred_hp = self.model(X_tensor)

                # 将结果移到CPU并保存
                pred_op_list.append(pred_op.cpu().numpy())
                pred_lp_list.append(pred_lp.cpu().numpy())
                pred_hp_list.append(pred_hp.cpu().numpy())

                if (i + 1) % max(1, num_batches // 10) == 0:
                    print(f"[INFO] 已完成 {i+1}/{num_batches} 批预测")

            # 合并所有批次的结果
            pred_op_np = np.concatenate(pred_op_list, axis=0).flatten()
            pred_lp_np = np.concatenate(pred_lp_list, axis=0).flatten()
            pred_hp_np = np.concatenate(pred_hp_list, axis=0).flatten()

            # 【关键修复】逆缩放预测值，还原为实际变化率
            # 训练时标签被放大了 label_scale 倍，预测时需要缩小相同倍数
            pred_op_np = pred_op_np / self.label_scale
            pred_lp_np = pred_lp_np / self.label_scale
            pred_hp_np = pred_hp_np / self.label_scale

            # 计算得分：score = pred_hp - pred_op（预期开盘后做多空间）
            # 由于两个值都被逆缩放，得分也会相应缩放，不影响排序
            scores = pred_hp_np - pred_op_np

            predictions = {
                'op': pred_op_np,
                'lp': pred_lp_np,
                'hp': pred_hp_np
            }

            print(f"[INFO] 预测完成，共 {n_samples} 个样本")
            print(f"[INFO] 预测op范围: [{pred_op_np.min():.6f}, {pred_op_np.max():.6f}]")
            print(f"[INFO] 预测hp范围: [{pred_hp_np.min():.6f}, {pred_hp_np.max():.6f}]")
            print(f"[INFO] 得分范围: [{scores.min():.6f}, {scores.max():.6f}]")
            return scores, predictions
        except Exception as e:
            print(f"[ERROR] 预测失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None


class BackTester:
    """
    回测类：模拟每日交易，计算绩效指标
    新增指标：分组收益（Top10% vs Bottom10%）、胜率、收益分布统计
    """

    def __init__(self, n=10, k=2, transaction_cost=0.001):
        """
        初始化回测器

        Args:
            n: 持仓数量
            k: 每日调仓数量
            transaction_cost: 交易成本率
        """
        self.n = n
        self.k = k
        self.transaction_cost = transaction_cost

        # 回测结果
        self.portfolio = {}
        self.daily_returns = []
        self.daily_portfolio_values = []
        self.daily_positions = []

        # IC计算相关
        self.daily_ics = []
        self.predicted_scores = []
        self.actual_returns = []

    def calculate_group_returns(self, scores, returns, n_groups=10):
        """
        计算分组收益（Top10% vs Bottom10%）

        Args:
            scores: 预测得分
            returns: 实际收益
            n_groups: 分组数量

        Returns:
            dict: 包含各分组收益的字典
        """
        sorted_indices = np.argsort(scores)
        group_size = len(scores) // n_groups

        if group_size == 0:
            return {'top_group_return': 0.0, 'bottom_group_return': 0.0, 'spread': 0.0}

        # 计算各组平均收益
        group_returns = {}
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < n_groups - 1 else len(scores)
            group_stocks = sorted_indices[start_idx:end_idx]
            group_return = returns[group_stocks].mean()
            group_returns[f'group_{i+1}'] = group_return

        # 计算Top和Bottom组
        top_indices = sorted_indices[-group_size:]
        bottom_indices = sorted_indices[:group_size]

        top_return = returns[top_indices].mean()
        bottom_return = returns[bottom_indices].mean()
        spread = top_return - bottom_return

        group_returns['top_group_return'] = top_return
        group_returns['bottom_group_return'] = bottom_return
        group_returns['spread'] = spread

        return group_returns

    def calculate_win_rate(self, scores, returns):
        """
        计算胜率（预测方向正确的比例）

        Args:
            scores: 预测得分
            returns: 实际收益

        Returns:
            float: 胜率
        """
        # 方向定义：得分>0预测涨，实际收益>0为涨
        pred_up = scores > 0
        actual_up = returns > 0

        # 正确的情况：预测涨且实际涨，或预测跌且实际跌
        correct = (pred_up & actual_up) | (~pred_up & ~actual_up)

        win_rate = correct.mean()
        return win_rate

    def calculate_return_stats(self, returns):
        """
        计算收益分布统计

        Args:
            returns: 收益序列

        Returns:
            dict: 包含各统计指标的字典
        """
        if len(returns) == 0:
            return {}

        stats = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'skew': pd.Series(returns).skew(),
            'kurt': pd.Series(returns).kurtosis(),
            'min': np.min(returns),
            'max': np.max(returns),
            'median': np.median(returns)
        }
        return stats

    def run_backtest(self, dates, stocks, scores, actual_op, current_close):
        """
        运行完整回测

        Args:
            dates: 日期数组
            stocks: 股票代码数组
            scores: 预测得分数组
            actual_op: 实际开盘价变化率（用于计算次日收益）
            current_close: 当前收盘价（用于计算买入成本）

        Returns:
            dict: 回测指标
        """
        # 按日期分组数据
        date_groups = defaultdict(list)
        for i, date in enumerate(dates):
            date_groups[date].append({
                'stock': stocks[i],
                'score': scores[i],
                'actual_op': actual_op[i],
                'current_close': current_close[i]
            })

        # 按日期排序
        sorted_dates = sorted(date_groups.keys())
        print(f"[INFO] 回测日期范围: {sorted_dates[0]} 至 {sorted_dates[-1]}")
        print(f"[INFO] 回测天数: {len(sorted_dates)}")

        # 用于计算整体IC的数据收集
        all_scores = []
        all_actual_returns = []

        initial_capital = 1000000.0
        current_capital = initial_capital
        portfolio_value = initial_capital

        # 第一天建仓
        first_date = sorted_dates[0]
        first_day_stocks = date_groups[first_date]

        # 按得分排序，选择前n只
        first_day_stocks.sort(key=lambda x: x['score'], reverse=True)
        selected_stocks = first_day_stocks[:self.n]

        # 等权建仓
        self.portfolio = {}
        weight = 1.0 / self.n
        for item in selected_stocks:
            stock = item['stock']
            close_price = item['current_close']
            self.portfolio[stock] = {
                'weight': weight,
                'cost_price': close_price,
                'quantity': (current_capital * weight) / close_price
            }
            current_capital -= self.portfolio[stock]['quantity'] * close_price

        self.daily_portfolio_values.append(portfolio_value)
        self.daily_positions.append(list(self.portfolio.keys()))

        # 收集IC数据
        all_scores.extend([x['score'] for x in first_day_stocks])
        all_actual_returns.extend([x['actual_op'] for x in first_day_stocks])

        # 计算第一天的分组收益和胜率
        day_scores = np.array([x['score'] for x in first_day_stocks])
        day_returns = np.array([x['actual_op'] for x in first_day_stocks])
        group_returns = self.calculate_group_returns(day_scores, day_returns)
        win_rate = self.calculate_win_rate(day_scores, day_returns)
        daily_return_stats = self.calculate_return_stats(day_returns)

        print(f"\n[INFO] 第一天 - 分组收益: Top10% = {group_returns['top_group_return']:.4f}, "
              f"Bottom10% = {group_returns['bottom_group_return']:.4f}, Spread = {group_returns['spread']:.4f}")
        print(f"[INFO] 第一天 - 胜率: {win_rate:.2%}")

        # 逐日回测
        for t, date in enumerate(sorted_dates[1:], 1):
            prev_date = sorted_dates[t - 1]
            day_stocks = date_groups[date]
            prev_day_stocks = date_groups[prev_date]

            # 计算前一天持仓的当日收益
            daily_return = 0.0
            for stock in list(self.portfolio.keys()):
                # 查找前一天该股票的信息
                prev_item = next((x for x in prev_day_stocks if x['stock'] == stock), None)
                if prev_item is None:
                    continue

                # 使用实际开盘价变化率计算收益
                # 假设买入价格是前一天收盘价，卖出是第二天开盘价
                stock_return = prev_item['actual_op']
                position_value = self.portfolio[stock]['quantity'] * prev_item['current_close'] * (1 + stock_return)
                daily_return += (position_value - self.portfolio[stock]['quantity'] * prev_item['current_close'])

                # 更新持仓信息
                self.portfolio[stock]['cost_price'] = prev_item['current_close'] * (1 + stock_return)

            # 更新组合价值
            portfolio_value += daily_return
            self.daily_portfolio_values.append(portfolio_value)

            # 记录每日收益率（相对初始资金）
            daily_rate = daily_return / (self.daily_portfolio_values[t - 1])
            self.daily_returns.append(daily_rate)

            # 调仓逻辑：保留前n-k只，新增k只
            # 按得分排序所有股票
            day_stocks.sort(key=lambda x: x['score'], reverse=True)

            # 保留当前持仓中得分最高的n-k只
            current_positions = list(self.portfolio.keys())
            current_with_scores = []
            for stock in current_positions:
                stock_item = next((x for x in day_stocks if x['stock'] == stock), None)
                if stock_item is not None:
                    current_with_scores.append({
                        'stock': stock,
                        'score': stock_item['score'],
                        'current_close': stock_item['current_close']
                    })

            current_with_scores.sort(key=lambda x: x['score'], reverse=True)
            keep_stocks = [x['stock'] for x in current_with_scores[:max(self.n - self.k, 0)]]

            # 选择候选新股（不在keep_stocks中的高分股票）
            candidate_stocks = []
            for item in day_stocks:
                stock = item['stock']
                if stock not in keep_stocks and stock not in self.portfolio:
                    candidate_stocks.append(item)
                    if len(candidate_stocks) >= self.k:
                        break

            # 构建新组合
            new_portfolio = {}
            total_value = portfolio_value

            # 先卖出不在新组合中的持仓
            for stock in list(self.portfolio.keys()):
                if stock not in keep_stocks + [x['stock'] for x in candidate_stocks]:
                    # 查找当日该股票的收盘价
                    day_item = next((x for x in day_stocks if x['stock'] == stock), None)
                    if day_item is None:
                        continue

                    sell_price = day_item['current_close'] * (1 - self.transaction_cost)
                    quantity = self.portfolio[stock]['quantity']
                    proceeds = quantity * sell_price
                    current_capital += proceeds
                    del self.portfolio[stock]

            # 计算等权配置
            new_n = len(keep_stocks) + len(candidate_stocks)
            if new_n > 0:
                weight = 1.0 / new_n
                target_value = total_value * weight

                # 处理保留股票
                for stock in keep_stocks:
                    # 查找当日该股票的收盘价
                    day_item = next((x for x in day_stocks if x['stock'] == stock), None)
                    if day_item is None:
                        continue

                    close_price = day_item['current_close']
                    current_value = self.portfolio[stock]['quantity'] * close_price

                    # 调整仓位
                    if current_value > target_value * (1 + self.transaction_cost):
                        # 减仓
                        sell_value = current_value - target_value
                        sell_quantity = sell_value / close_price
                        self.portfolio[stock]['quantity'] -= sell_quantity
                        current_capital += sell_quantity * close_price * (1 - self.transaction_cost)
                    elif current_value < target_value * (1 - self.transaction_cost):
                        # 加仓
                        buy_value = target_value - current_value
                        buy_quantity = buy_value / close_price
                        if current_capital >= buy_value * (1 + self.transaction_cost):
                            self.portfolio[stock]['quantity'] += buy_quantity
                            current_capital -= buy_value * (1 + self.transaction_cost)

                    new_portfolio[stock] = {
                        'weight': weight,
                        'cost_price': close_price,
                        'quantity': self.portfolio[stock]['quantity']
                    }

                # 处理新买入股票
                for item in candidate_stocks:
                    stock = item['stock']
                    close_price = item['current_close']

                    buy_value = target_value
                    buy_quantity = buy_value / close_price
                    if current_capital >= buy_value * (1 + self.transaction_cost):
                        new_portfolio[stock] = {
                            'weight': weight,
                            'cost_price': close_price,
                            'quantity': buy_quantity
                        }
                        current_capital -= buy_value * (1 + self.transaction_cost)

            self.portfolio = new_portfolio
            self.daily_positions.append(list(self.portfolio.keys()))

            # 收集IC数据
            all_scores.extend([x['score'] for x in day_stocks])
            all_actual_returns.extend([x['actual_op'] for x in day_stocks])

        # 计算整体指标
        self.predicted_scores = np.array(all_scores)
        self.actual_returns = np.array(all_actual_returns)

        # 计算IC
        if len(self.predicted_scores) > 1:
            try:
                ic, _ = pearsonr(self.predicted_scores, self.actual_returns)
            except:
                ic = 0.0
        else:
            ic = 0.0

        # 计算分组收益和胜率（整体）
        overall_group_returns = self.calculate_group_returns(
            self.predicted_scores, self.actual_returns
        )
        overall_win_rate = self.calculate_win_rate(
            self.predicted_scores, self.actual_returns
        )
        overall_return_stats = self.calculate_return_stats(self.actual_returns)

        # 计算收益率指标
        if len(self.daily_returns) > 0:
            total_return = (self.daily_portfolio_values[-1] - initial_capital) / initial_capital
            daily_mean = np.mean(self.daily_returns)
            daily_std = np.std(self.daily_returns)
            if daily_std > 0:
                sharpe_ratio = daily_mean / daily_std * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # 计算最大回撤
            portfolio_values = np.array(self.daily_portfolio_values)
            cummax = np.maximum.accumulate(portfolio_values)
            drawdown = (cummax - portfolio_values) / cummax
            max_drawdown = np.max(drawdown)
        else:
            total_return = 0.0
            daily_mean = 0.0
            daily_std = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0

        metrics = {
            'total_return': total_return,
            'daily_mean': daily_mean,
            'daily_std': daily_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'ic': ic,
            'top_group_return': overall_group_returns['top_group_return'],
            'bottom_group_return': overall_group_returns['bottom_group_return'],
            'spread': overall_group_returns['spread'],
            'win_rate': overall_win_rate,
            'return_stats': overall_return_stats
        }

        return metrics

    def print_metrics(self, metrics):
        """打印回测指标"""
        print("\n" + "="*60)
        print("回测指标")
        print("="*60)
        print(f"累计收益: {metrics['total_return']:.2%}")
        print(f"日均收益: {metrics['daily_mean']:.4%}")
        print(f"收益波动: {metrics['daily_std']:.4%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"IC值: {metrics['ic']:.4f}")
        print("-"*60)
        print("分组收益:")
        print(f"Top10%: {metrics['top_group_return']:.4%}")
        print(f"Bottom10%: {metrics['bottom_group_return']:.4%}")
        print(f"Spread: {metrics['spread']:.4%}")
        print("-"*60)
        print(f"胜率: {metrics['win_rate']:.2%}")
        print("-"*60)
        print("收益分布统计:")
        print(f"均值: {metrics['return_stats']['mean']:.4%}")
        print(f"标准差: {metrics['return_stats']['std']:.4%}")
        print(f"偏度: {metrics['return_stats']['skew']:.4f}")
        print(f"峰度: {metrics['return_stats']['kurt']:.4f}")
        print("="*60)
