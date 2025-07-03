"""
Main entry point for the IB Stock Scanner CLI
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .alerts.display import ScannerDisplay
from .config.settings import Settings
from .data.cache_manager import CacheManager
from .data.database import Database
from .scanner.criteria import CriteriaPresets, load_criteria_from_config
from .scanner.ib_connection import IBConnectionManager
from .scanner.scanner import StockScanner

# Setup rich console
console = Console()

# Global scanner instance for signal handling
scanner_instance: Optional[StockScanner] = None


def setup_logging(level: str = "INFO"):
    """Setup logging with rich handler"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False, markup=True)],
    )


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    console.print("\n[yellow]Received shutdown signal, cleaning up...[/yellow]")
    if scanner_instance:
        asyncio.create_task(scanner_instance.disconnect())
    sys.exit(0)


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.option(
    "--log-level", "-l", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"])
)
@click.pass_context
def cli(ctx, config, log_level):
    """IB Stock Scanner - Real-time stock scanning with AI enhancement"""
    # Setup logging
    setup_logging(log_level)

    # Load configuration
    if config:
        ctx.obj = Settings(config_path=config)
    else:
        ctx.obj = Settings()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    console.print("[bold blue]IB Stock Scanner[/bold blue] 🚀")
    console.print(f"Using config: {ctx.obj.config_path}")


@cli.command()
@click.option("--host", help="IB Gateway host")
@click.option("--port", type=int, help="IB Gateway port")
@click.option("--client-id", type=int, help="Client ID")
@click.pass_obj
def test_connection(settings, host, port, client_id):
    """Test connection to IB Gateway"""

    async def _test():
        # Override config if provided
        config = settings.config.ib_connection.dict()
        if host:
            config["host"] = host
        if port:
            config["port"] = port
        if client_id:
            config["client_id"] = client_id

        console.print(f"Testing connection to {config['host']}:{config['port']}...")

        ib_manager = IBConnectionManager(config)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Connecting...", total=None)

            if await ib_manager.connect():
                progress.update(task, description="[green]Connected successfully![/green]")

                # Test market data subscription
                console.print("\nTesting market data subscription...")
                ticker = await ib_manager.subscribe_market_data("AAPL")

                if ticker:
                    await asyncio.sleep(2)  # Wait for data

                    table = Table(title="AAPL Market Data")
                    table.add_column("Field", style="cyan")
                    table.add_column("Value", style="green")

                    table.add_row("Last Price", f"${ticker.last:.2f}" if ticker.last else "N/A")
                    table.add_row("Volume", f"{ticker.volume:,}" if ticker.volume else "N/A")
                    table.add_row("Bid", f"${ticker.bid:.2f}" if ticker.bid else "N/A")
                    table.add_row("Ask", f"${ticker.ask:.2f}" if ticker.ask else "N/A")

                    console.print(table)
                else:
                    console.print("[yellow]Could not subscribe to market data[/yellow]")

                await ib_manager.disconnect()
            else:
                progress.update(task, description="[red]Connection failed![/red]")
                console.print("\n[red]Could not connect to IB Gateway.[/red]")
                console.print("Please ensure:")
                console.print("  • IB Gateway or TWS is running")
                console.print("  • API connections are enabled")
                console.print("  • The host/port are correct")

    asyncio.run(_test())


@cli.command()
@click.option("--symbols", "-s", help="Comma-separated list of symbols")
@click.option(
    "--universe", "-u", default="TEST", type=click.Choice(["US_STOCKS", "TECH", "SP500", "TEST"])
)
@click.option(
    "--criteria",
    "-c",
    default="momentum",
    type=click.Choice(["momentum", "volume", "oversold", "technical", "custom"]),
)
@click.option("--continuous", is_flag=True, help="Run continuous scanning")
@click.option("--interval", "-i", default=60, help="Scan interval in seconds")
@click.option("--duration", "-d", type=int, help="Duration in seconds (for non-continuous)")
@click.option("--export", "-e", type=click.Path(), help="Export results to CSV")
@click.option("--demo", is_flag=True, help="Run in demo mode without IB connection")
@click.pass_obj
def scan(settings, symbols, universe, criteria, continuous, interval, duration, export, demo):
    """Run stock scanner with specified criteria"""
    global scanner_instance

    async def _scan():
        # Initialize components
        if not demo:
            ib_manager = IBConnectionManager()
            if not await ib_manager.connect():
                console.print("[red]Failed to connect to IB Gateway[/red]")
                return
        else:
            console.print("[yellow]Running in demo mode (no IB connection)[/yellow]")
            ib_manager = None

        database = Database()
        cache = CacheManager()
        scanner_instance = StockScanner(ib_manager, database, cache)

        # Setup display
        display = ScannerDisplay(console)
        scanner_instance.add_result_callback(display.update_result)

        # Get symbols
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
        else:
            symbol_list = await scanner_instance.get_universe_symbols(universe)

        console.print(f"Scanning {len(symbol_list)} symbols from {universe} universe")

        # Load criteria
        if criteria == "momentum":
            scan_criteria = [CriteriaPresets.momentum_gainers()]
        elif criteria == "volume":
            scan_criteria = [CriteriaPresets.volume_breakouts()]
        elif criteria == "oversold":
            scan_criteria = [CriteriaPresets.oversold_bounce()]
        elif criteria == "technical":
            scan_criteria = [CriteriaPresets.technical_breakout()]
        else:
            # Load from config
            scan_criteria = load_criteria_from_config(settings.config.model_dump())

        # Run scanner
        try:
            if continuous:
                console.print(f"Starting continuous scan with {interval}s interval")
                console.print("Press Ctrl+C to stop")

                # Start display update loop
                asyncio.create_task(display.run())

                # Start scanning
                await scanner_instance.scan_symbols(
                    symbol_list, scan_criteria, continuous=True, interval=interval
                )

                # Wait forever (until interrupted)
                await asyncio.Event().wait()

            else:
                # One-time scan
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Scanning...", total=None)

                    if duration:
                        # Scan for specified duration
                        end_time = asyncio.get_event_loop().time() + duration
                        results = []

                        while asyncio.get_event_loop().time() < end_time:
                            batch_results = await scanner_instance.scan_symbols(
                                symbol_list, scan_criteria, continuous=False
                            )
                            results.extend(batch_results)

                            remaining = int(end_time - asyncio.get_event_loop().time())
                            progress.update(task, description=f"Scanning... {remaining}s remaining")

                            if remaining > interval:
                                await asyncio.sleep(interval)
                    else:
                        # Single scan
                        results = await scanner_instance.scan_symbols(
                            symbol_list, scan_criteria, continuous=False
                        )

                    progress.update(task, description="[green]Scan complete![/green]")

                # Display results
                display.display_results(results[:20])  # Top 20

                # Export if requested
                if export and results:
                    import pandas as pd

                    df = pd.DataFrame([r.to_dict() for r in results])
                    df.to_csv(export, index=False)
                    console.print(f"Results exported to {export}")

        finally:
            if scanner_instance:
                await scanner_instance.disconnect()

    asyncio.run(_scan())


@cli.command()
@click.option("--symbols", "-s", required=True, help="Comma-separated list of symbols")
@click.option(
    "--criteria",
    "-c",
    default="momentum",
    type=click.Choice(["momentum", "volume", "oversold", "technical"]),
)
@click.option("--days", "-d", default=30, help="Number of days to backtest")
@click.pass_obj
def backtest(settings, symbols, criteria, days):
    """Backtest scanning criteria on historical data"""

    async def _backtest():
        # Initialize components
        ib_manager = IBConnectionManager()
        if not await ib_manager.connect():
            console.print("[red]Failed to connect to IB Gateway[/red]")
            return

        database = Database()
        cache = CacheManager()
        scanner = StockScanner(ib_manager, database, cache)

        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Load criteria
        if criteria == "momentum":
            scan_criteria = [CriteriaPresets.momentum_gainers()]
        elif criteria == "volume":
            scan_criteria = [CriteriaPresets.volume_breakouts()]
        elif criteria == "oversold":
            scan_criteria = [CriteriaPresets.oversold_bounce()]
        elif criteria == "technical":
            scan_criteria = [CriteriaPresets.technical_breakout()]

        console.print(
            f"Backtesting {criteria} criteria on {len(symbol_list)} symbols over {days} days"
        )

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Running backtest...", total=None)

            results = await scanner.backtest_criteria(symbol_list, scan_criteria, days)

            progress.update(task, description="[green]Backtest complete![/green]")

        # Display results
        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Signals", str(results["total_signals"]))
        table.add_row("Successful Signals", str(results["successful_signals"]))

        if results["total_signals"] > 0:
            success_rate = results["successful_signals"] / results["total_signals"] * 100
            table.add_row("Success Rate", f"{success_rate:.1f}%")
            table.add_row("Average Return", f"{results.get('average_return', 0):.2f}%")

        console.print(table)

        # Show per-symbol results
        if results["symbol_results"]:
            symbol_table = Table(title="Per-Symbol Results")
            symbol_table.add_column("Symbol", style="cyan")
            symbol_table.add_column("Signals", style="yellow")
            symbol_table.add_column("Avg Return", style="green")

            for symbol, data in results["symbol_results"].items():
                symbol_table.add_row(symbol, str(data["signals"]), f"{data['average_return']:.2f}%")

            console.print(symbol_table)

        await scanner.disconnect()

    asyncio.run(_backtest())


@cli.command()
@click.option(
    "--model", "-m", required=True, type=click.Choice(["breakout", "pattern", "sentiment"])
)
@click.option("--data-path", "-d", type=click.Path(exists=True), help="Path to training data")
@click.option("--output", "-o", type=click.Path(), help="Output path for trained model")
@click.pass_obj
def train_model(settings, model, data_path, output):
    """Train machine learning models"""
    console.print(f"[yellow]Training {model} model...[/yellow]")
    console.print("[red]Model training not yet implemented[/red]")
    console.print("This would:")
    console.print("  • Load historical data")
    console.print("  • Engineer features")
    console.print("  • Train the specified model")
    console.print("  • Save to models/ directory")


@cli.command()
@click.pass_obj
def cleanup(settings):
    """Clean up old data from database"""

    async def _cleanup():
        database = Database()

        console.print("Cleaning up old data...")
        database.cleanup_old_data(days=30)

        # Show database stats
        stats = database.get_database_stats()

        table = Table(title="Database Statistics")
        table.add_column("Table", style="cyan")
        table.add_column("Records", style="green")

        for table_name, count in stats.items():
            table.add_row(table_name.replace("_", " ").title(), f"{count:,}")

        console.print(table)

    asyncio.run(_cleanup())


@cli.command()
@click.pass_obj
def stats(settings):
    """Show system statistics"""

    async def _stats():
        database = Database()
        cache = CacheManager()

        # Database stats
        db_stats = database.get_database_stats()

        db_table = Table(title="Database Statistics")
        db_table.add_column("Table", style="cyan")
        db_table.add_column("Records", style="green")

        for table_name, count in db_stats.items():
            db_table.add_row(table_name.replace("_", " ").title(), f"{count:,}")

        console.print(db_table)

        # Cache stats
        cache_stats = cache.get_stats()

        cache_table = Table(title="Cache Statistics")
        cache_table.add_column("Metric", style="cyan")
        cache_table.add_column("Value", style="green")

        for metric, value in cache_stats.items():
            cache_table.add_row(metric.replace("_", " ").title(), str(value))

        console.print(cache_table)

    asyncio.run(_stats())


if __name__ == "__main__":
    cli()
