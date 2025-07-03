#!/usr/bin/env python
"""Quick test to see raw scan data"""

import asyncio
from src.scanner.ib_connection import IBConnectionManager
from src.scanner.data_processor import DataProcessor
from src.data.database import Database
from src.data.cache_manager import CacheManager

async def test_scan():
    # Initialize components
    ib_manager = IBConnectionManager()
    database = Database()
    cache = CacheManager()
    processor = DataProcessor(database, cache)
    
    # Connect to IB
    if not await ib_manager.connect():
        print("Failed to connect to IB Gateway")
        return
        
    # Test symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        ticker = await ib_manager.subscribe_market_data(symbol)
        
        if ticker:
            # Wait for data
            await asyncio.sleep(2)
            
            print(f"Last Price: ${ticker.last}")
            print(f"Volume: {ticker.volume:,}")
            print(f"Bid: ${ticker.bid}")
            print(f"Ask: ${ticker.ask}")
            print(f"Open: ${ticker.open}")
            print(f"High: ${ticker.high}")
            print(f"Low: ${ticker.low}")
            
            # Process ticker
            result = await processor.process_ticker(ticker)
            if result:
                print(f"Change %: {result.change_pct:.2f}%")
                print(f"Volume Ratio: {result.volume_ratio:.2f}")
                print(f"Criteria Matched: {result.criteria_matched}")
        else:
            print(f"Failed to get market data for {symbol}")
    
    await ib_manager.disconnect()

if __name__ == "__main__":
    asyncio.run(test_scan())