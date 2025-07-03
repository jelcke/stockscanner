"""
Basic Interactive Brokers Stock Scanner
Demonstrates core IB API connectivity and simple percentage gain filtering
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd
from ib_async import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockData:
    symbol: str
    last_price: float = 0.0
    volume: int = 0
    prev_close: float = 0.0
    change_pct: float = 0.0
    last_update: datetime = None

class BasicScanner:
    def __init__(self, host='127.0.0.1', port=7497, client_id=10):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.stocks: Dict[str, StockData] = {}

    async def connect(self):
        """Connect to IB Gateway"""
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            # Use delayed data for paper trading (no subscription required)
            self.ib.reqMarketDataType(3)  # 3 = Delayed data
            logger.info("Connected to IB Gateway")
            logger.info("Using delayed market data (15-20 minute delay)")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    async def add_symbol(self, symbol: str):
        """Add symbol to watchlist"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            contracts = await self.ib.qualifyContractsAsync(contract)

            if not contracts:
                logger.warning(f"Could not qualify contract for {symbol}")
                return

            contract = contracts[0]
            self.stocks[symbol] = StockData(symbol=symbol)

            # Get historical data for previous close
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='2 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True
            )

            if bars and len(bars) >= 2:
                self.stocks[symbol].prev_close = bars[-2].close

            # Subscribe to live data
            ticker = self.ib.reqMktData(contract, '', False, False)
            ticker.updateEvent += self._on_ticker_update

            logger.info(f"Added {symbol} to watchlist")

        except Exception as e:
            logger.error(f"Error adding {symbol}: {e}")

    def _on_ticker_update(self, ticker):
        """Handle ticker updates"""
        symbol = ticker.contract.symbol
        if symbol not in self.stocks:
            return

        stock = self.stocks[symbol]

        if ticker.last and not pd.isna(ticker.last):
            stock.last_price = ticker.last
            if stock.prev_close > 0:
                stock.change_pct = ((stock.last_price - stock.prev_close) / stock.prev_close) * 100

        if ticker.volume and not pd.isna(ticker.volume):
            stock.volume = ticker.volume

        stock.last_update = datetime.now()

    def get_top_movers(self, min_change_pct=0.0, limit=10) -> List[StockData]:
        """Get top moving stocks (gainers and losers)"""
        movers = [
            stock for stock in self.stocks.values()
            if abs(stock.change_pct) >= min_change_pct and stock.last_price > 0
        ]
        return sorted(movers, key=lambda x: x.change_pct, reverse=True)[:limit]

    def print_results(self):
        """Print scanning results"""
        movers = self.get_top_movers()

        print(f"\n{'='*60}")
        print(f"Stock Scanner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"{'Symbol':<8} {'Price':<10} {'Change %':<12} {'Volume':<12}")
        print(f"{'-'*60}")

        if not movers:
            print("Waiting for market data...")
        else:
            for stock in movers:
                # Color code based on positive/negative change
                if stock.change_pct > 0:
                    change_str = f"+{stock.change_pct:.2f}%"
                else:
                    change_str = f"{stock.change_pct:.2f}%"

                print(f"{stock.symbol:<8} ${stock.last_price:<9.2f} "
                      f"{change_str:<12} {stock.volume:<12,}")

    async def run(self, symbols: List[str], duration=60):
        """Run the scanner"""
        await self.connect()

        # Add symbols
        for symbol in symbols:
            await self.add_symbol(symbol)
            await asyncio.sleep(0.1)  # Rate limiting

        # Monitor for specified duration
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < duration:
            self.print_results()
            await asyncio.sleep(5)

        self.ib.disconnect()

# Example usage
async def main():
    scanner = BasicScanner()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    await scanner.run(symbols, duration=120)

if __name__ == "__main__":
    asyncio.run(main())
