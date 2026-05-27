import os
import sys
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TradePlanGenerator:
    """
    交易计划生成器：
    1. 读取每日预测排名 CSV
    2. 读取当前持仓状态 JSON
    3. 对比前 N 名与当前持仓 -> 生成买入/卖出清单
    4. 计算每只股票的买卖数量（满仓等权，100 股整数倍）
    5. 输出 trade_plan CSV
    6. 更新持仓状态 JSON
    """

    def __init__(self, predictions_path, portfolio_path='portfolio_state.json',
                 n_hold=10, k_rebalance=2, capital=None):
        self.predictions_path = predictions_path
        self.portfolio_path = portfolio_path
        self.n_hold = n_hold
        self.k_rebalance = k_rebalance
        self.capital_override = capital
        self.df_pred = None
        self.portfolio = None
        self.trade_date = None

    # =========================================================
    # 加载数据
    # =========================================================
    def load(self):
        if not os.path.exists(self.predictions_path):
            raise FileNotFoundError(f"预测文件不存在: {self.predictions_path}")
        self.df_pred = pd.read_csv(self.predictions_path)
        self.trade_date = str(self.df_pred.iloc[0].get('date', 'unknown'))
        # 每日预测文件名为 daily_predictions_YYYYMMDD.csv, 提取日期
        basename = os.path.basename(self.predictions_path)
        if basename.startswith('daily_predictions_') and basename.endswith('.csv'):
            self.trade_date = basename[len('daily_predictions_'):-4]
        print(f"[INFO] 预测文件: {self.predictions_path}, 日期: {self.trade_date}")
        print(f"[INFO] 候选股票数: {len(self.df_pred)}")

        if os.path.exists(self.portfolio_path):
            with open(self.portfolio_path, 'r', encoding='utf-8') as f:
                self.portfolio = json.load(f)
            print(f"[INFO] 持仓文件: {self.portfolio_path}")
        else:
            self.portfolio = {
                "init_capital": 1000000,
                "current_capital": 1000000,
                "total_value": 1000000,
                "positions": {},
                "last_run_date": ""
            }
            print("[INFO] 未找到持仓文件，使用空仓初始化")

        if self.capital_override is not None:
            self.portfolio['total_value'] = self.capital_override
            if not self.portfolio.get('positions'):
                self.portfolio['current_capital'] = self.capital_override
        return self

    # =========================================================
    # 计算调仓方案
    # =========================================================
    def compute_rebalance(self):
        """
        选出得分前 N 的股票 -> 卖出不在前 N 的持仓 -> 买入未持有的前 N 股票
        每日最多调仓 K 只
        """
        top_n = self.df_pred.head(self.n_hold)
        top_codes = set(top_n['ts_code'].tolist())
        top_info = {}
        for _, row in top_n.iterrows():
            top_info[row['ts_code']] = {
                'score': row['score'],
                'close': row['close'],
                'name': row.get('name', row['ts_code'].split('.')[0]),
                'ths_code': row.get('ths_code', row['ts_code'].split('.')[0]),
            }

        current_positions = self.portfolio.get('positions', {})
        current_codes = set(current_positions.keys())

        codes_to_sell = current_codes - top_codes
        codes_to_buy = top_codes - current_codes

        total_value = self.portfolio.get('total_value',
                        self.portfolio.get('init_capital', 1000000))
        per_stock_budget = total_value / self.n_hold

        sell_plan = []
        for i, code in enumerate(sorted(codes_to_sell, key=lambda c:
            current_positions[c].get('quantity', 0) * current_positions[c].get('cost_price', 0),
            reverse=True)):
            if i >= self.k_rebalance:
                break
            pos = current_positions[code]
            qty = int(pos.get('quantity', 0))
            if qty <= 0:
                continue
            sell_plan.append({
                '操作': '卖出',
                'ts_code': code,
                'ths_code': code.split('.')[0],
                '名称': self._get_name(code),
                '价格': round(pos.get('cost_price', 0), 2),
                '数量': qty,
                '金额': round(qty * pos.get('cost_price', 0), 2),
                '得分排名': '-',
                '备注': '排名跌出前{}'.format(self.n_hold)
            })

        buy_plan = []
        for i, code in enumerate(sorted(codes_to_buy, key=lambda c:
            top_info[c]['score'], reverse=True)):
            if i >= self.k_rebalance:
                break
            info = top_info[code]
            close_price = info['close']
            if close_price <= 0:
                continue
            qty = int(per_stock_budget / close_price / 100) * 100
            if qty == 0:
                qty = 100
            buy_plan.append({
                '操作': '买入',
                'ts_code': code,
                'ths_code': info['ths_code'],
                '名称': info['name'],
                '价格': round(close_price, 2),
                '数量': qty,
                '金额': round(qty * close_price, 2),
                '得分排名': str(top_n[top_n['ts_code'] == code].index[0] + 1)
                    if code in top_n['ts_code'].values else '-',
                '备注': '等权配置 ~{:.0f}元'.format(per_stock_budget)
            })

        print(f"\n[INFO] 当前持仓: {len(current_codes)} 只")
        print(f"[INFO] 目标持仓: {self.n_hold} 只 (前{self.n_hold}名)")
        print(f"[INFO] 每只预算: {per_stock_budget/10000:.1f}万")
        print(f"[INFO] 卖出: {len(sell_plan)} 只")
        print(f"[INFO] 买入: {len(buy_plan)} 只")

        return sell_plan, buy_plan

    # =========================================================
    # 保存交易计划 CSV
    # =========================================================
    def save_trade_plan(self, sell_plan, buy_plan, output_dir='./trade_plans'):
        os.makedirs(output_dir, exist_ok=True)

        all_rows = sell_plan + buy_plan
        if not all_rows:
            print("\n[INFO] 无需调仓，所有持仓股票仍在前{}名".format(self.n_hold))
            plan_path = os.path.join(output_dir, f'trade_plan_{self.trade_date}.csv')
            pd.DataFrame(columns=['操作', 'ts_code', 'ths_code', '名称', '价格', '数量', '金额', '得分排名', '备注']
                        ).to_csv(plan_path, index=False, encoding='utf-8-sig')
            print(f"[INFO] 空交易计划已保存: {plan_path}")
            return plan_path

        columns = ['操作', 'ts_code', 'ths_code', '名称', '价格', '数量', '金额', '得分排名', '备注']
        plan_df = pd.DataFrame(all_rows, columns=columns)
        plan_path = os.path.join(output_dir, f'trade_plan_{self.trade_date}.csv')
        plan_df.to_csv(plan_path, index=False, encoding='utf-8-sig')
        print(f"\n[INFO] 交易计划已保存: {plan_path}")

        return plan_path

    # =========================================================
    # 打印交易计划摘要
    # =========================================================
    def print_summary(self, sell_plan, buy_plan):
        all_rows = sell_plan + buy_plan
        if not all_rows:
            print(f"\n{'='*60}")
            print("交易计划: 无需调仓")
            print(f"{'='*60}")
            return

        print(f"\n{'='*60}")
        print(f"交易计划  {self.trade_date}")
        print(f"{'='*60}")
        print(f"本金: {self.portfolio.get('init_capital', 1000000):,} 元")
        print(f"持仓数: {self.n_hold} 只 | 每日调仓上限: {self.k_rebalance} 只")
        print(f"{'='*60}")

        if sell_plan:
            print(f"\n  【卖出】({len(sell_plan)} 只)")
            print(f"  {'-'*50}")
            for p in sell_plan:
                print(f"  {p['ths_code']} {p['名称']:<8s}  {p['数量']:>6d}股  ~{p['金额']:>10,.0f}元  {p['备注']}")

        if buy_plan:
            print(f"\n  【买入】({len(buy_plan)} 只)")
            print(f"  {'-'*50}")
            for p in buy_plan:
                print(f"  {p['ths_code']} {p['名称']:<8s}  {p['数量']:>6d}股  ~{p['金额']:>10,.0f}元  {p['备注']}")

        total_buy = sum(p['金额'] for p in buy_plan)
        total_sell = sum(p['金额'] for p in sell_plan)
        print(f"\n  买入合计: {total_buy:>10,.0f} 元")
        print(f"  卖出合计: {total_sell:>10,.0f} 元")
        print(f"{'='*60}")

    # =========================================================
    # 更新持仓状态（预估）
    # =========================================================
    def update_portfolio_state(self, sell_plan, buy_plan):
        """
        根据交易计划更新持仓 JSON。
        注意：此为预估更新，实际成交价可能有偏差。
        """
        positions = self.portfolio.get('positions', {})

        for p in sell_plan:
            code = p['ts_code']
            if code in positions:
                del positions[code]

        for p in buy_plan:
            code = p['ts_code']
            positions[code] = {
                'quantity': p['数量'],
                'cost_price': p['价格'],
                'buy_date': self.trade_date
            }

        self.portfolio['positions'] = positions
        self.portfolio['last_run_date'] = self.trade_date

        with open(self.portfolio_path, 'w', encoding='utf-8') as f:
            json.dump(self.portfolio, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 持仓状态已更新: {self.portfolio_path}")
        print(f"[INFO] 当前持仓: {len(positions)} 只")

    # =========================================================
    # 辅助：获取股票名称
    # =========================================================
    def _get_name(self, ts_code):
        row = self.df_pred[self.df_pred['ts_code'] == ts_code]
        if not row.empty and 'name' in row.columns:
            name = row.iloc[0]['name']
            if pd.notna(name) and str(name) != 'nan':
                return str(name)
        return ts_code.split('.')[0]

    # =========================================================
    # 主流程
    # =========================================================
    def run(self, output_dir='./trade_plans'):
        self.load()
        sell_plan, buy_plan = self.compute_rebalance()
        self.save_trade_plan(sell_plan, buy_plan, output_dir)
        self.print_summary(sell_plan, buy_plan)
        self.update_portfolio_state(sell_plan, buy_plan)


def parse_args():
    parser = argparse.ArgumentParser(description='生成每日交易计划')
    parser.add_argument('--predictions', type=str, required=True,
                        help='每日预测 CSV 路径（由 daily_trader.py 生成）')
    parser.add_argument('--portfolio', type=str, default='portfolio_state.json',
                        help='持仓状态 JSON 路径')
    parser.add_argument('--n', type=int, default=10,
                        help='持仓数量（默认 10）')
    parser.add_argument('--k', type=int, default=2,
                        help='每日最大调仓数量（默认 2）')
    parser.add_argument('--capital', type=float, default=None,
                        help='本金，不指定则使用持仓文件中的 total_value')
    parser.add_argument('--output_dir', type=str, default='./trade_plans',
                        help='交易计划 CSV 输出目录')
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("交易计划生成器")
    print("=" * 60)

    try:
        gen = TradePlanGenerator(
            predictions_path=args.predictions,
            portfolio_path=args.portfolio,
            n_hold=args.n,
            k_rebalance=args.k,
            capital=args.capital
        )
        gen.run(output_dir=args.output_dir)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\n[提示] 请先运行每日预测:")
        print("  python daily_trader.py --model_dir ./saved_models/production_v1")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
