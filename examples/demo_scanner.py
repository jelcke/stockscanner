"""
Demo Scanner - Simulated IB Stock Scanner
This demonstrates the scanner functionality without requiring an IB connection
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class StockData:
    symbol: str
    last_price: float = 0.0
    volume: int = 0
    prev_close: float = 0.0
    change_pct: float = 0.0
    last_update: datetime = None

class DemoScanner:
    def __init__(self):
        self.stocks: Dict[str, StockData] = {}
        self.running = False
        
    def initialize_stocks(self, symbols: List[str]):
        """Initialize stocks with random data"""
        for symbol in symbols:
            base_price = random.uniform(50, 500)
            self.stocks[symbol] = StockData(
                symbol=symbol,
                last_price=base_price,
                prev_close=base_price * random.uniform(0.95, 1.05),
                volume=random.randint(1000000, 50000000),
                last_update=datetime.now()
            )
            # Calculate initial change percentage
            stock = self.stocks[symbol]
            stock.change_pct = ((stock.last_price - stock.prev_close) / stock.prev_close) * 100
            
    async def simulate_market_data(self):
        """Simulate market data updates"""
        while self.running:
            # Update random stocks
            for symbol in random.sample(list(self.stocks.keys()), min(3, len(self.stocks))):
                stock = self.stocks[symbol]
                
                # Simulate price movement
                change = random.uniform(-0.02, 0.02)  # ±2% change
                stock.last_price = stock.last_price * (1 + change)
                
                # Update change percentage
                stock.change_pct = ((stock.last_price - stock.prev_close) / stock.prev_close) * 100
                
                # Simulate volume changes
                stock.volume += random.randint(10000, 100000)
                
                stock.last_update = datetime.now()
                
            await asyncio.sleep(1)  # Update every second
            
    def get_top_gainers(self, min_change_pct=5.0, limit=10) -> List[StockData]:
        """Get top gaining stocks"""
        gainers = [
            stock for stock in self.stocks.values()
            if stock.change_pct >= min_change_pct and stock.last_price > 0
        ]
        return sorted(gainers, key=lambda x: x.change_pct, reverse=True)[:limit]
        
    def get_top_losers(self, max_change_pct=-5.0, limit=10) -> List[StockData]:
        """Get top losing stocks"""
        losers = [
            stock for stock in self.stocks.values()
            if stock.change_pct <= max_change_pct and stock.last_price > 0
        ]
        return sorted(losers, key=lambda x: x.change_pct)[:limit]
        
    def get_high_volume(self, min_volume=10000000, limit=10) -> List[StockData]:
        """Get high volume stocks"""
        high_volume = [
            stock for stock in self.stocks.values()
            if stock.volume >= min_volume
        ]
        return sorted(high_volume, key=lambda x: x.volume, reverse=True)[:limit]
        
    def print_results(self):
        """Print scanning results"""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        # Print header
        print(f"\n{'='*80}")
        print(f"Demo Stock Scanner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Top Gainers
        gainers = self.get_top_gainers(min_change_pct=2.0)
        if gainers:
            print(f"\n📈 TOP GAINERS (>2%)")
            print(f"{'-'*80}")
            print(f"{'Symbol':<8} {'Price':<10} {'Change %':<12} {'Volume':<15} {'Last Update':<20}")
            print(f"{'-'*80}")
            
            for stock in gainers[:5]:
                print(f"{stock.symbol:<8} ${stock.last_price:<9.2f} "
                      f"\033[92m{stock.change_pct:<11.2f}%\033[0m {stock.volume:<15,} "
                      f"{stock.last_update.strftime('%H:%M:%S'):<20}")
        
        # Top Losers
        losers = self.get_top_losers(max_change_pct=-2.0)
        if losers:
            print(f"\n📉 TOP LOSERS (<-2%)")
            print(f"{'-'*80}")
            print(f"{'Symbol':<8} {'Price':<10} {'Change %':<12} {'Volume':<15} {'Last Update':<20}")
            print(f"{'-'*80}")
            
            for stock in losers[:5]:
                print(f"{stock.symbol:<8} ${stock.last_price:<9.2f} "
                      f"\033[91m{stock.change_pct:<11.2f}%\033[0m {stock.volume:<15,} "
                      f"{stock.last_update.strftime('%H:%M:%S'):<20}")
        
        # High Volume
        high_volume = self.get_high_volume()
        if high_volume:
            print(f"\n📊 HIGH VOLUME")
            print(f"{'-'*80}")
            print(f"{'Symbol':<8} {'Price':<10} {'Change %':<12} {'Volume':<15} {'Last Update':<20}")
            print(f"{'-'*80}")
            
            for stock in high_volume[:5]:
                change_color = "\033[92m" if stock.change_pct > 0 else "\033[91m"
                print(f"{stock.symbol:<8} ${stock.last_price:<9.2f} "
                      f"{change_color}{stock.change_pct:<11.2f}%\033[0m {stock.volume:<15,} "
                      f"{stock.last_update.strftime('%H:%M:%S'):<20}")
                      
    async def run(self, symbols: List[str], duration=60):
        """Run the demo scanner"""
        print("Initializing demo scanner...")
        self.initialize_stocks(symbols)
        
        self.running = True
        
        # Start market data simulation
        market_task = asyncio.create_task(self.simulate_market_data())
        
        # Monitor for specified duration
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < duration:
            self.print_results()
            await asyncio.sleep(3)  # Update display every 3 seconds
            
        self.running = False
        await market_task

# Example usage
async def main():
    scanner = DemoScanner()
    
    # Large list of symbols for more interesting demo
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
        'NFLX', 'BABA', 'JPM', 'V', 'WMT', 'DIS', 'MA', 'HD', 'PYPL',
        'INTC', 'CSCO', 'ADBE', 'CRM', 'ORCL', 'AVGO', 'TXN', 'QCOM',
        'BA', 'MMM', 'CAT', 'GE', 'F', 'GM', 'UBER', 'LYFT', 'SNAP',
        'TWTR', 'SQ', 'ROKU', 'ZM', 'DOCU', 'CRWD', 'DDOG', 'SNOW'
    ]
    
    print("Starting Demo Stock Scanner...")
    print("This simulates market data without requiring IB connection")
    print("Press Ctrl+C to stop\n")
    
    try:
        await scanner.run(symbols, duration=120)  # Run for 2 minutes
    except KeyboardInterrupt:
        print("\nScanner stopped by user")

if __name__ == "__main__":
    asyncio.run(main())