#!/usr/bin/env python3
"""
Backtesting Framework Example
============================
This example demonstrates how to backtest scanning strategies to evaluate
their historical performance.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ib_stock_scanner import ScannerConfig, StockScanner
from ib_stock_scanner.backtesting import Backtester, Strategy
from ib_stock_scanner.criteria import (
    MomentumCriteria,
    PriceCriteria,
    TechnicalCriteria,
    VolatilityCriteria,
    VolumeCriteria,
)


class MomentumBreakoutStrategy(Strategy):
    """Example momentum breakout strategy"""

    def __init__(self):
        super().__init__(name="Momentum Breakout")

        # Strategy parameters
        self.holding_period = 5  # days
        self.stop_loss = 0.03    # 3%
        self.take_profit = 0.10  # 10%

        # Scan criteria
        self.criteria = [
            PriceCriteria(min_price=10.0, max_price=200.0),
            VolumeCriteria(min_avg_volume=1_000_000),
            MomentumCriteria(
                rsi_min=60,
                rsi_max=80,
                macd_signal="bullish"
            ),
            VolatilityCriteria(min_volatility=0.02, max_volatility=0.05),
            TechnicalCriteria(
                indicator="SMA",
                period=20,
                condition="above"
            )
        ]

    def generate_signals(self, scanner, date):
        """Generate buy signals for given date"""

        # Run scan with historical data up to date
        candidates = scanner.scan_historical(
            criteria=self.criteria,
            date=date,
            universe="US_STOCKS",
            max_results=10
        )

        signals = []
        for stock in candidates:
            # Additional filters
            hist_data = scanner.get_historical_data(
                symbol=stock.symbol,
                end_date=date,
                period="1M"
            )

            if hist_data is None or len(hist_data) < 20:
                continue

            # Check for breakout pattern
            if self.is_breakout(hist_data):
                signals.append({
                    'symbol': stock.symbol,
                    'date': date,
                    'price': stock.price,
                    'confidence': self.calculate_confidence(hist_data)
                })

        return signals

    def is_breakout(self, data):
        """Check if stock is breaking out"""
        # Simple breakout: price above 20-day high with volume
        current_price = data['close'].iloc[-1]
        high_20d = data['high'].iloc[-20:].max()
        avg_volume = data['volume'].iloc[-20:].mean()
        current_volume = data['volume'].iloc[-1]

        return (current_price > high_20d * 0.99 and
                current_volume > avg_volume * 1.5)

    def calculate_confidence(self, data):
        """Calculate signal confidence"""
        # Based on multiple factors
        confidence = 0.5

        # Volume confirmation
        if data['volume'].iloc[-1] > data['volume'].mean() * 2:
            confidence += 0.2

        # Trend strength
        sma20 = data['close'].rolling(20).mean()
        if data['close'].iloc[-1] > sma20.iloc[-1] * 1.02:
            confidence += 0.15

        # Low volatility
        volatility = data['close'].pct_change().std()
        if volatility < 0.03:
            confidence += 0.15

        return min(confidence, 1.0)

def main():
    # Configure scanner
    config = ScannerConfig(
        host="127.0.0.1",
        port=7497,
        client_id=6
    )

    scanner = StockScanner(config)

    # Initialize backtester
    backtester = Backtester(
        scanner=scanner,
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now() - timedelta(days=1),
        initial_capital=100000,
        commission=0.005  # $0.005 per share
    )

    # Define strategies to test
    strategies = [
        MomentumBreakoutStrategy(),
        create_mean_reversion_strategy(),
        create_trend_following_strategy(),
        create_volatility_breakout_strategy()
    ]

    # Run backtests
    results = {}

    for strategy in strategies:
        print(f"\nBacktesting {strategy.name} strategy...")
        result = backtester.run(
            strategy=strategy,
            position_size=0.1,  # 10% of capital per position
            max_positions=5
        )
        results[strategy.name] = result

        # Print summary
        print(f"\n{strategy.name} Results:")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")

    # Compare strategies
    compare_strategies(results)

    # Detailed analysis of best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    analyze_strategy(best_strategy[0], best_strategy[1])

    # Generate report
    generate_backtest_report(results)

def create_mean_reversion_strategy():
    """Create mean reversion strategy"""

    class MeanReversionStrategy(Strategy):
        def __init__(self):
            super().__init__(name="Mean Reversion")
            self.holding_period = 3
            self.stop_loss = 0.02
            self.take_profit = 0.05

            self.criteria = [
                PriceCriteria(min_price=20.0, max_price=100.0),
                VolumeCriteria(min_avg_volume=2_000_000),
                MomentumCriteria(
                    rsi_min=20,
                    rsi_max=35,
                    macd_signal="oversold"
                ),
                TechnicalCriteria(
                    indicator="BB_LOWER",
                    period=20,
                    condition="below"
                )
            ]

        def generate_signals(self, scanner, date):
            candidates = scanner.scan_historical(
                criteria=self.criteria,
                date=date,
                universe="US_STOCKS",
                max_results=10
            )

            signals = []
            for stock in candidates:
                signals.append({
                    'symbol': stock.symbol,
                    'date': date,
                    'price': stock.price,
                    'confidence': 0.7
                })

            return signals

    return MeanReversionStrategy()

def create_trend_following_strategy():
    """Create trend following strategy"""

    class TrendFollowingStrategy(Strategy):
        def __init__(self):
            super().__init__(name="Trend Following")
            self.holding_period = 20
            self.stop_loss = 0.05
            self.take_profit = 0.20

            self.criteria = [
                PriceCriteria(min_price=10.0),
                VolumeCriteria(min_avg_volume=500_000),
                TechnicalCriteria(
                    indicator="SMA",
                    period=50,
                    condition="above"
                ),
                TechnicalCriteria(
                    indicator="SMA",
                    period=200,
                    condition="above"
                ),
                MomentumCriteria(
                    rsi_min=50,
                    rsi_max=70
                )
            ]

        def generate_signals(self, scanner, date):
            candidates = scanner.scan_historical(
                criteria=self.criteria,
                date=date,
                universe="US_STOCKS",
                max_results=10
            )

            signals = []
            for stock in candidates:
                signals.append({
                    'symbol': stock.symbol,
                    'date': date,
                    'price': stock.price,
                    'confidence': 0.8
                })

            return signals

    return TrendFollowingStrategy()

def create_volatility_breakout_strategy():
    """Create volatility breakout strategy"""

    class VolatilityBreakoutStrategy(Strategy):
        def __init__(self):
            super().__init__(name="Volatility Breakout")
            self.holding_period = 2
            self.stop_loss = 0.02
            self.take_profit = 0.04

            self.criteria = [
                PriceCriteria(min_price=5.0, max_price=50.0),
                VolumeCriteria(min_avg_volume=5_000_000),
                VolatilityCriteria(min_volatility=0.03, max_volatility=0.08),
                TechnicalCriteria(
                    indicator="ATR",
                    period=14,
                    condition="expanding"
                )
            ]

        def generate_signals(self, scanner, date):
            candidates = scanner.scan_historical(
                criteria=self.criteria,
                date=date,
                universe="US_STOCKS",
                max_results=10
            )

            signals = []
            for stock in candidates:
                signals.append({
                    'symbol': stock.symbol,
                    'date': date,
                    'price': stock.price,
                    'confidence': 0.6
                })

            return signals

    return VolatilityBreakoutStrategy()

def compare_strategies(results):
    """Compare strategy performance"""

    # Create comparison dataframe
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Strategy': name,
            'Total Return': result.total_return,
            'Annual Return': result.annual_return,
            'Sharpe Ratio': result.sharpe_ratio,
            'Sortino Ratio': result.sortino_ratio,
            'Max Drawdown': result.max_drawdown,
            'Win Rate': result.win_rate,
            'Avg Win': result.avg_win,
            'Avg Loss': result.avg_loss,
            'Profit Factor': result.profit_factor,
            'Total Trades': result.total_trades
        })

    df = pd.DataFrame(comparison_data)

    # Display comparison table
    print("\n" + "="*100)
    print("STRATEGY COMPARISON")
    print("="*100)
    print(df.to_string(index=False))

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Returns comparison
    ax = axes[0, 0]
    strategies = df['Strategy']
    returns = df['Total Return'] * 100
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax.bar(strategies, returns, color=colors)
    ax.set_title('Total Returns (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)

    # 2. Risk-adjusted returns
    ax = axes[0, 1]
    ax.scatter(df['Max Drawdown'] * 100, df['Total Return'] * 100, s=100)
    for i, strategy in enumerate(df['Strategy']):
        ax.annotate(strategy, (df['Max Drawdown'].iloc[i] * 100,
                              df['Total Return'].iloc[i] * 100))
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Risk vs Return', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Win rate vs profit factor
    ax = axes[1, 0]
    ax.scatter(df['Win Rate'] * 100, df['Profit Factor'], s=100)
    for i, strategy in enumerate(df['Strategy']):
        ax.annotate(strategy, (df['Win Rate'].iloc[i] * 100,
                              df['Profit Factor'].iloc[i]))
    ax.set_xlabel('Win Rate (%)')
    ax.set_ylabel('Profit Factor')
    ax.set_title('Win Rate vs Profit Factor', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Sharpe ratio comparison
    ax = axes[1, 1]
    ax.bar(strategies, df['Sharpe Ratio'])
    ax.set_title('Sharpe Ratios', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150)
    plt.show()

def analyze_strategy(strategy_name, result):
    """Detailed analysis of a strategy"""

    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {strategy_name}")
    print(f"{'='*80}")

    # Performance metrics
    print("\nPerformance Metrics:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Annual Return: {result.annual_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {result.calmar_ratio:.2f}")

    # Risk metrics
    print("\nRisk Metrics:")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Max Drawdown Duration: {result.max_drawdown_duration} days")
    print(f"  Daily VaR (95%): {result.daily_var:.2%}")
    print(f"  Beta: {result.beta:.2f}")

    # Trade statistics
    print("\nTrade Statistics:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Average Win: {result.avg_win:.2%}")
    print(f"  Average Loss: {result.avg_loss:.2%}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Expectancy: {result.expectancy:.2%}")

    # Create detailed visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Equity curve
    ax = axes[0, 0]
    ax.plot(result.equity_curve.index, result.equity_curve.values)
    ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)

    # 2. Drawdown chart
    ax = axes[0, 1]
    ax.fill_between(result.drawdown.index, result.drawdown.values * 100,
                    color='red', alpha=0.3)
    ax.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)

    # 3. Monthly returns heatmap
    ax = axes[1, 0]
    monthly_returns = result.monthly_returns
    sns.heatmap(monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, ax=ax)
    ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')

    # 4. Trade distribution
    ax = axes[1, 1]
    trade_returns = result.trade_returns * 100
    ax.hist(trade_returns, bins=30, alpha=0.7, color='blue')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{strategy_name.replace(" ", "_")}_analysis.png', dpi=150)
    plt.show()

    # Top winning/losing trades
    print("\nTop 5 Winning Trades:")
    for i, trade in enumerate(result.top_trades[:5], 1):
        print(f"  {i}. {trade['symbol']} - {trade['return']:.2%} "
              f"({trade['entry_date']} to {trade['exit_date']})")

    print("\nTop 5 Losing Trades:")
    for i, trade in enumerate(result.worst_trades[:5], 1):
        print(f"  {i}. {trade['symbol']} - {trade['return']:.2%} "
              f"({trade['entry_date']} to {trade['exit_date']})")

def generate_backtest_report(results):
    """Generate comprehensive backtest report"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"backtest_report_{timestamp}"

    # Create HTML report
    html_content = f"""
    <html>
    <head>
        <title>Backtest Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Stock Scanner Backtest Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Strategy Comparison</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
                <th>Total Trades</th>
            </tr>
    """

    for name, result in results.items():
        return_class = "positive" if result.total_return > 0 else "negative"
        html_content += f"""
            <tr>
                <td>{name}</td>
                <td class="{return_class}">{result.total_return:.2%}</td>
                <td>{result.sharpe_ratio:.2f}</td>
                <td class="negative">{result.max_drawdown:.2%}</td>
                <td>{result.win_rate:.2%}</td>
                <td>{result.total_trades}</td>
            </tr>
        """

    html_content += """
        </table>
        
        <h2>Strategy Details</h2>
    """

    for name, result in results.items():
        html_content += f"""
        <h3>{name}</h3>
        <ul>
            <li>Annual Return: {result.annual_return:.2%}</li>
            <li>Sortino Ratio: {result.sortino_ratio:.2f}</li>
            <li>Profit Factor: {result.profit_factor:.2f}</li>
            <li>Average Win: {result.avg_win:.2%}</li>
            <li>Average Loss: {result.avg_loss:.2%}</li>
        </ul>
        """

    html_content += """
    </body>
    </html>
    """

    # Save HTML report
    with open(f"{report_name}.html", "w") as f:
        f.write(html_content)

    print(f"\nBacktest report saved as: {report_name}.html")

    # Save detailed results to CSV
    detailed_results = []
    for name, result in results.items():
        detailed_results.append({
            'Strategy': name,
            'Total_Return': result.total_return,
            'Annual_Return': result.annual_return,
            'Sharpe_Ratio': result.sharpe_ratio,
            'Sortino_Ratio': result.sortino_ratio,
            'Max_Drawdown': result.max_drawdown,
            'Win_Rate': result.win_rate,
            'Total_Trades': result.total_trades,
            'Profit_Factor': result.profit_factor
        })

    df = pd.DataFrame(detailed_results)
    df.to_csv(f"{report_name}.csv", index=False)
    print(f"Detailed results saved as: {report_name}.csv")

if __name__ == "__main__":
    main()
