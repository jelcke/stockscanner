"""
Test IB Connection and Market Data
Simple script to verify IB API connection and data flow
"""

import asyncio
from ib_async import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    """Test basic IB connection and market data"""
    ib = IB()
    
    try:
        # Connect to IB
        await ib.connectAsync('127.0.0.1', 7497, clientId=11)
        logger.info("Connected to IB Gateway")
        
        # Request delayed market data
        ib.reqMarketDataType(3)
        logger.info("Requested delayed market data")
        
        # Create a simple stock contract
        contract = Stock('AAPL', 'SMART', 'USD')
        await ib.qualifyContractsAsync(contract)
        logger.info(f"Contract qualified: {contract}")
        
        # Request market data
        ticker = ib.reqMktData(contract, '', False, False)
        logger.info("Market data requested")
        
        # Wait for data
        logger.info("Waiting for market data...")
        for i in range(10):
            await asyncio.sleep(2)
            logger.info(f"Ticker update {i+1}:")
            logger.info(f"  Last: {ticker.last}")
            logger.info(f"  Bid: {ticker.bid}")
            logger.info(f"  Ask: {ticker.ask}")
            logger.info(f"  Volume: {ticker.volume}")
            logger.info(f"  Close: {ticker.close}")
            
            if ticker.last:
                logger.info("✓ Market data received successfully!")
                break
        else:
            logger.warning("No market data received after 20 seconds")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        
    finally:
        ib.disconnect()
        logger.info("Disconnected")

if __name__ == "__main__":
    asyncio.run(test_connection())