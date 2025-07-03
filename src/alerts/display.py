"""
Rich terminal display for scanner results
"""

import asyncio
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..scanner.models import ScannerResult


class ScannerDisplay:
    """Display scanner results in a rich terminal interface"""

    def __init__(self, console: Console):
        self.console = console
        self.results: dict[str, ScannerResult] = {}
        self.last_update = datetime.now()
        self._running = False

    def update_result(self, result: ScannerResult):
        """Update result for a symbol"""
        self.results[result.symbol] = result
        self.last_update = datetime.now()

    def display_results(self, results: list[ScannerResult]):
        """Display a list of results in a table"""
        if not results:
            self.console.print("[yellow]No results to display[/yellow]")
            return

        table = Table(
            title=f"Scan Results - Top {len(results)} Movers",
            show_header=True,
            header_style="bold magenta",
        )

        # Add columns
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Change %", justify="right", width=10)
        table.add_column("Volume", justify="right", width=12)
        table.add_column("Vol Ratio", justify="right", width=10)
        table.add_column("RSI", justify="right", width=6)
        table.add_column("Signals", width=30)

        # Add rows
        for result in results:
            # Color code based on change
            if result.change_pct > 0:
                change_color = "green"
                change_text = f"+{result.change_pct:.2f}%"
            else:
                change_color = "red"
                change_text = f"{result.change_pct:.2f}%"

            # Format volume
            if result.volume >= 1_000_000:
                volume_text = f"{result.volume / 1_000_000:.1f}M"
            elif result.volume >= 1_000:
                volume_text = f"{result.volume / 1_000:.0f}K"
            else:
                volume_text = str(result.volume)

            # Get technical signals
            rsi = result.technical_signals.get("rsi")
            rsi_text = f"{rsi:.0f}" if rsi else "N/A"

            # Build signals text
            signals = []
            if result.volume_ratio > 2:
                signals.append("🔥 Volume Surge")
            if rsi and rsi > 70:
                signals.append("📈 Overbought")
            elif rsi and rsi < 30:
                signals.append("📉 Oversold")
            if result.technical_signals.get("macd_cross") == "bullish":
                signals.append("🟢 MACD Bull")
            elif result.technical_signals.get("macd_cross") == "bearish":
                signals.append("🔴 MACD Bear")

            table.add_row(
                result.symbol,
                f"${result.price:.2f}",
                Text(change_text, style=change_color),
                volume_text,
                f"{result.volume_ratio:.1f}x",
                rsi_text,
                " ".join(signals) if signals else "",
            )

        self.console.print(table)

    async def run(self, update_interval: float = 1.0):
        """Run live display updates"""
        self._running = True

        with Live(self._generate_layout(), console=self.console, refresh_per_second=1) as live:
            while self._running:
                live.update(self._generate_layout())
                await asyncio.sleep(update_interval)

    def stop(self):
        """Stop live display"""
        self._running = False

    def _generate_layout(self) -> Layout:
        """Generate the display layout"""
        layout = Layout()

        # Header
        header = Panel(
            Text(
                f"IB Stock Scanner - Live Results\n"
                f"Last Update: {self.last_update.strftime('%H:%M:%S')} | "
                f"Active Symbols: {len(self.results)}",
                justify="center",
            ),
            style="bold blue",
        )

        # Results table
        table = self._create_results_table()

        # Stats panel
        stats = self._create_stats_panel()

        # Compose layout
        layout.split_column(Layout(header, size=3), Layout(table, size=20), Layout(stats, size=6))

        return layout

    def _create_results_table(self) -> Table:
        """Create results table for live display"""
        table = Table(show_header=True, header_style="bold magenta", title="Top Movers")

        # Add columns
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Change %", justify="right", width=10)
        table.add_column("Volume", justify="right", width=12)
        table.add_column("Vol Ratio", justify="right", width=10)
        table.add_column("Criteria", width=20)

        # Sort results by absolute change
        sorted_results = sorted(
            self.results.values(), key=lambda x: abs(x.change_pct), reverse=True
        )[
            :20
        ]  # Top 20

        # Add rows
        for result in sorted_results:
            # Color code based on change
            if result.change_pct > 0:
                change_color = "green"
                change_text = f"+{result.change_pct:.2f}%"
            else:
                change_color = "red"
                change_text = f"{result.change_pct:.2f}%"

            # Format volume
            if result.volume >= 1_000_000:
                volume_text = f"{result.volume / 1_000_000:.1f}M"
            elif result.volume >= 1_000:
                volume_text = f"{result.volume / 1_000:.0f}K"
            else:
                volume_text = str(result.volume)

            # Criteria matched
            criteria_text = ", ".join(result.criteria_matched[:2])  # First 2
            if len(result.criteria_matched) > 2:
                criteria_text += f" +{len(result.criteria_matched) - 2}"

            table.add_row(
                result.symbol,
                f"${result.price:.2f}",
                Text(change_text, style=change_color),
                volume_text,
                f"{result.volume_ratio:.1f}x",
                criteria_text,
            )

        return table

    def _create_stats_panel(self) -> Panel:
        """Create statistics panel"""
        if not self.results:
            return Panel("No statistics available", title="Statistics")

        # Calculate statistics
        gainers = sum(1 for r in self.results.values() if r.change_pct > 0)
        losers = sum(1 for r in self.results.values() if r.change_pct < 0)
        volume_surges = sum(1 for r in self.results.values() if r.volume_ratio > 2)

        avg_change = sum(r.change_pct for r in self.results.values()) / len(self.results)
        max_gainer = max(self.results.values(), key=lambda x: x.change_pct)
        max_loser = min(self.results.values(), key=lambda x: x.change_pct)

        stats_text = Text()
        stats_text.append("Gainers: ", style="bold")
        stats_text.append(f"{gainers}", style="green")
        stats_text.append(" | ")
        stats_text.append("Losers: ", style="bold")
        stats_text.append(f"{losers}", style="red")
        stats_text.append(" | ")
        stats_text.append("Volume Surges: ", style="bold")
        stats_text.append(f"{volume_surges}\n", style="yellow")

        stats_text.append("Average Change: ", style="bold")
        if avg_change > 0:
            stats_text.append(f"+{avg_change:.2f}%\n", style="green")
        else:
            stats_text.append(f"{avg_change:.2f}%\n", style="red")

        stats_text.append("Top Gainer: ", style="bold")
        stats_text.append(f"{max_gainer.symbol} ", style="cyan")
        stats_text.append(f"+{max_gainer.change_pct:.2f}%\n", style="green")

        stats_text.append("Top Loser: ", style="bold")
        stats_text.append(f"{max_loser.symbol} ", style="cyan")
        stats_text.append(f"{max_loser.change_pct:.2f}%", style="red")

        return Panel(stats_text, title="Statistics", style="blue")
