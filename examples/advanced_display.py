"""
Advanced Display Interface with Rich
Professional terminal interface with live updating tables, charts, and alerts
"""

import asyncio
from datetime import datetime
from typing import Dict

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class AdvancedDisplay:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.data = {
            'stocks': {},
            'alerts': [],
            'sentiment': {},
            'patterns': {},
            'predictions': {}
        }

    def setup_layout(self):
        """Setup the terminal layout"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=5)
        )

        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )

        self.layout["left"].split(
            Layout(name="scanner", ratio=2),
            Layout(name="alerts", ratio=1)
        )

        self.layout["right"].split(
            Layout(name="sentiment"),
            Layout(name="patterns")
        )

    def create_header(self) -> Panel:
        """Create header panel"""
        title = Text("🚀 AI-Enhanced Stock Scanner", style="bold blue")
        subtitle = Text(f"Last Update: {datetime.now().strftime('%H:%M:%S')}", style="dim")

        header_text = Text()
        header_text.append(title)
        header_text.append("\n")
        header_text.append(subtitle)

        return Panel(
            Align.center(header_text),
            style="blue",
            padding=(0, 1)
        )

    def create_scanner_table(self) -> Table:
        """Create main scanner results table"""
        table = Table(title="📈 Top Movers", title_style="bold green")

        # Add columns
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Change %", justify="right", width=10)
        table.add_column("Volume", justify="right", width=12)
        table.add_column("RVOL", justify="right", width=8)
        table.add_column("Pattern", width=12)
        table.add_column("ML Score", justify="right", width=10)
        table.add_column("Sentiment", justify="center", width=10)

        # Add data rows
        for symbol, stock_data in sorted(
            self.data['stocks'].items(),
            key=lambda x: x[1].get('change_pct', 0),
            reverse=True
        )[:15]:

            # Price and change formatting
            price = stock_data.get('price', 0)
            change_pct = stock_data.get('change_pct', 0)
            volume = stock_data.get('volume', 0)
            rvol = stock_data.get('rvol', 0)

            # Color coding
            change_color = "green" if change_pct > 0 else "red"
            rvol_color = "yellow" if rvol > 3 else "white"

            # Format values
            price_str = f"${price:.2f}"
            change_str = f"[{change_color}]{change_pct:+.1f}%[/{change_color}]"
            volume_str = f"{volume:,.0f}" if volume > 0 else "N/A"
            rvol_str = f"[{rvol_color}]{rvol:.1f}x[/{rvol_color}]"

            # Pattern and ML data
            pattern_data = self.data['patterns'].get(symbol, {})
            pattern_str = pattern_data.get('pattern', 'None')[:12]

            ml_data = self.data['predictions'].get(symbol, {})
            ml_score = ml_data.get('breakout_probability', 0)
            ml_str = f"{ml_score:.1%}" if ml_score > 0 else "N/A"
            ml_color = "green" if ml_score > 0.7 else "yellow" if ml_score > 0.5 else "white"
            ml_str = f"[{ml_color}]{ml_str}[/{ml_color}]"

            # Sentiment
            sentiment_data = self.data['sentiment'].get(symbol, {})
            sentiment_score = sentiment_data.get('sentiment_score', 0.5)
            if sentiment_score > 0.6:
                sentiment_str = "[green]😊[/green]"
            elif sentiment_score < 0.4:
                sentiment_str = "[red]😞[/red]"
            else:
                sentiment_str = "😐"

            table.add_row(
                symbol, price_str, change_str, volume_str,
                rvol_str, pattern_str, ml_str, sentiment_str
            )

        return table

    def create_alerts_panel(self) -> Panel:
        """Create alerts panel"""
        if not self.data['alerts']:
            content = Text("No active alerts", style="dim")
        else:
            content = Text()
            for i, alert in enumerate(self.data['alerts'][-5:]):  # Last 5 alerts
                symbol = alert.get('symbol', 'Unknown')
                alert_type = alert.get('type', 'ALERT')
                message = alert.get('message', 'No message')

                # Color coding by alert type
                if 'BREAKOUT' in alert_type:
                    color = "green"
                    icon = "🚀"
                elif 'SENTIMENT' in alert_type:
                    color = "blue"
                    icon = "📰"
                elif 'PATTERN' in alert_type:
                    color = "yellow"
                    icon = "📊"
                else:
                    color = "white"
                    icon = "⚠️"

                content.append(f"{icon} ", style=color)
                content.append(f"{symbol}: ", style=f"bold {color}")
                content.append(f"{message}\n", style=color)

        return Panel(
            content,
            title="🚨 Recent Alerts",
            title_align="left",
            border_style="red"
        )

    def create_sentiment_panel(self) -> Panel:
        """Create sentiment analysis panel"""
        if not self.data['sentiment']:
            content = Text("No sentiment data", style="dim")
        else:
            content = Text()
            content.append("Sentiment Overview\n", style="bold blue")
            content.append("-" * 20 + "\n")

            # Calculate average sentiment
            sentiments = [data.get('sentiment_score', 0.5)
                         for data in self.data['sentiment'].values()]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5

            sentiment_text = "Bullish" if avg_sentiment > 0.6 else "Bearish" if avg_sentiment < 0.4 else "Neutral"
            sentiment_color = "green" if avg_sentiment > 0.6 else "red" if avg_sentiment < 0.4 else "yellow"

            content.append("Market Sentiment: ", style="white")
            content.append(f"{sentiment_text}\n", style=f"bold {sentiment_color}")
            content.append(f"Score: {avg_sentiment:.2f}\n\n", style="white")

            # Top sentiment movers
            content.append("Top Sentiment:\n", style="bold")
            sorted_sentiment = sorted(
                self.data['sentiment'].items(),
                key=lambda x: x[1].get('sentiment_score', 0),
                reverse=True
            )[:5]

            for symbol, data in sorted_sentiment:
                score = data.get('sentiment_score', 0)
                articles = data.get('article_count', 0)
                content.append(f"{symbol}: {score:.2f} ({articles} articles)\n")

        return Panel(
            content,
            title="📰 Sentiment Analysis",
            title_align="left",
            border_style="blue"
        )

    def create_patterns_panel(self) -> Panel:
        """Create pattern detection panel"""
        if not self.data['patterns']:
            content = Text("No pattern data", style="dim")
        else:
            content = Text()
            content.append("Pattern Detection\n", style="bold yellow")
            content.append("-" * 20 + "\n")

            # Count patterns
            pattern_counts = {}
            for data in self.data['patterns'].values():
                pattern = data.get('pattern', 'Unknown')
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            # Display pattern summary
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                content.append(f"{pattern.replace('_', ' ').title()}: {count}\n")

            content.append("\nBreakout Candidates:\n", style="bold green")

            # Show breakout candidates
            breakout_candidates = [
                (symbol, data) for symbol, data in self.data['patterns'].items()
                if data.get('is_breakout', False)
            ]

            if breakout_candidates:
                for symbol, data in breakout_candidates[:5]:
                    confidence = data.get('confidence', 0)
                    content.append(f"🚀 {symbol}: {confidence:.1%}\n", style="green")
            else:
                content.append("None detected\n", style="dim")

        return Panel(
            content,
            title="📊 Pattern Analysis",
            title_align="left",
            border_style="yellow"
        )

    def create_footer(self) -> Panel:
        """Create footer with statistics"""
        stats_text = Text()

        # Calculate statistics
        total_symbols = len(self.data['stocks'])
        gainers = len([s for s in self.data['stocks'].values() if s.get('change_pct', 0) > 0])
        alerts_count = len(self.data['alerts'])

        stats_text.append(f"Symbols: {total_symbols} | ", style="white")
        stats_text.append(f"Gainers: {gainers} | ", style="green")
        stats_text.append(f"Alerts: {alerts_count} | ", style="red")
        stats_text.append(f"Updated: {datetime.now().strftime('%H:%M:%S')}", style="blue")

        return Panel(
            Align.center(stats_text),
            style="white"
        )

    def update_layout(self):
        """Update all layout components"""
        self.layout["header"].update(self.create_header())
        self.layout["scanner"].update(self.create_scanner_table())
        self.layout["alerts"].update(self.create_alerts_panel())
        self.layout["sentiment"].update(self.create_sentiment_panel())
        self.layout["patterns"].update(self.create_patterns_panel())
        self.layout["footer"].update(self.create_footer())

    def update_data(self, new_data: Dict):
        """Update display data"""
        for key, value in new_data.items():
            if key in self.data:
                if isinstance(value, dict):
                    self.data[key].update(value)
                elif isinstance(value, list):
                    self.data[key] = value
                else:
                    self.data[key] = value

    async def run_display(self, update_interval=1):
        """Run the live display"""
        self.setup_layout()

        with Live(self.layout, refresh_per_second=2, screen=True) as live:
            while True:
                self.update_layout()
                await asyncio.sleep(update_interval)

class DisplayManager:
    def __init__(self):
        self.display = AdvancedDisplay()
        self.running = False

    def add_stock_data(self, symbol: str, data: Dict):
        """Add or update stock data"""
        self.display.update_data({'stocks': {symbol: data}})

    def add_alert(self, alert: Dict):
        """Add new alert"""
        current_alerts = self.display.data['alerts']
        current_alerts.append({
            **alert,
            'timestamp': datetime.now()
        })

        # Keep only last 20 alerts
        if len(current_alerts) > 20:
            current_alerts[:] = current_alerts[-20:]

    def add_sentiment_data(self, symbol: str, sentiment_data: Dict):
        """Add sentiment data for symbol"""
        self.display.update_data({'sentiment': {symbol: sentiment_data}})

    def add_pattern_data(self, symbol: str, pattern_data: Dict):
        """Add pattern data for symbol"""
        self.display.update_data({'patterns': {symbol: pattern_data}})

    def add_prediction_data(self, symbol: str, prediction_data: Dict):
        """Add ML prediction data for symbol"""
        self.display.update_data({'predictions': {symbol: prediction_data}})

    async def start_display(self):
        """Start the display in background"""
        if not self.running:
            self.running = True
            await self.display.run_display()

    def stop_display(self):
        """Stop the display"""
        self.running = False

# Example usage
async def demo_advanced_display():
    """Demonstrate advanced display capabilities"""
    display_manager = DisplayManager()

    # Add sample data
    sample_stocks = {
        'AAPL': {'price': 150.25, 'change_pct': 2.5, 'volume': 25000000, 'rvol': 1.8},
        'TSLA': {'price': 220.50, 'change_pct': 8.2, 'volume': 45000000, 'rvol': 3.2},
        'NVDA': {'price': 425.75, 'change_pct': 5.1, 'volume': 30000000, 'rvol': 2.1},
    }

    for symbol, data in sample_stocks.items():
        display_manager.add_stock_data(symbol, data)

    # Add sample alerts
    display_manager.add_alert({
        'symbol': 'TSLA',
        'type': 'BREAKOUT_PREDICTION',
        'message': 'High breakout probability detected'
    })

    # Add sample sentiment
    display_manager.add_sentiment_data('AAPL', {
        'sentiment_score': 0.75,
        'article_count': 15,
        'confidence': 0.8
    })

    # Add sample pattern
    display_manager.add_pattern_data('TSLA', {
        'pattern': 'bull_flag',
        'confidence': 0.85,
        'is_breakout': True
    })

    # Start display
    await display_manager.start_display()

if __name__ == "__main__":
    asyncio.run(demo_advanced_display())
