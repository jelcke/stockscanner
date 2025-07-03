#!/usr/bin/env python3
"""
ML Breakout Predictor Example
============================
This example uses machine learning to predict potential breakout stocks
based on historical patterns and technical indicators.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from ib_stock_scanner import ScannerConfig, StockScanner
from ib_stock_scanner.criteria import PriceCriteria, VolumeCriteria
from ib_stock_scanner.ml import BreakoutPredictor, FeatureEngineering

warnings.filterwarnings('ignore')

def main():
    # Configure scanner
    config = ScannerConfig(
        host="127.0.0.1",
        port=7497,
        client_id=4
    )

    scanner = StockScanner(config)

    # Initialize ML components
    feature_eng = FeatureEngineering()
    predictor = BreakoutPredictor(model_path="models/breakout_model.pkl")

    # Check if model exists, if not train it
    if not predictor.model_exists():
        print("Training breakout prediction model...")
        train_model(scanner, feature_eng, predictor)

    # Basic screening criteria
    criteria = [
        PriceCriteria(min_price=10.0, max_price=500.0),
        VolumeCriteria(min_avg_volume=500_000)
    ]

    # Get candidates
    print("\nScanning for breakout candidates...")
    candidates = scanner.scan(
        criteria=criteria,
        universe="US_STOCKS",
        max_results=200
    )

    print(f"Found {len(candidates)} candidates")

    # Predict breakouts
    predictions = []

    for stock in candidates:
        print(f"Analyzing {stock.symbol}...", end="\r")

        # Get historical data
        hist_data = scanner.get_historical_data(
            symbol=stock.symbol,
            period="6M",
            bar_size="1D"
        )

        if hist_data is None or len(hist_data) < 100:
            continue

        # Engineer features
        features = feature_eng.create_features(hist_data)

        if features is None:
            continue

        # Make prediction
        breakout_prob, confidence = predictor.predict(features)

        if breakout_prob > 0.65:  # 65% probability threshold
            # Calculate additional metrics
            current_price = hist_data['close'].iloc[-1]
            avg_volume = hist_data['volume'].mean()
            volatility = hist_data['close'].pct_change().std()

            # Support/Resistance levels
            resistance = calculate_resistance(hist_data)
            support = calculate_support(hist_data)
            distance_to_resistance = (resistance - current_price) / current_price

            predictions.append({
                'symbol': stock.symbol,
                'company': stock.company_name,
                'sector': stock.sector,
                'price': current_price,
                'breakout_probability': breakout_prob,
                'confidence': confidence,
                'resistance': resistance,
                'support': support,
                'distance_to_resistance': distance_to_resistance,
                'avg_volume': avg_volume,
                'volatility': volatility,
                'volume_surge': hist_data['volume'].iloc[-1] / avg_volume
            })

    # Sort by probability
    df = pd.DataFrame(predictions)
    df = df.sort_values('breakout_probability', ascending=False)

    # Display results
    print("\n" + "="*120)
    print("BREAKOUT PREDICTIONS - NEXT 5 DAYS")
    print("="*120)

    print(f"\n{'Symbol':<8} {'Company':<25} {'Sector':<20} {'Price':<8} "
          f"{'Prob':<8} {'Conf':<8} {'Resistance':<12} {'Dist':<8}")
    print("-" * 120)

    for _, row in df.head(20).iterrows():
        print(f"{row['symbol']:<8} "
              f"{row['company'][:23]:<25} "
              f"{row['sector'][:18]:<20} "
              f"${row['price']:<7.2f} "
              f"{row['breakout_probability']:<7.2%} "
              f"{row['confidence']:<7.2f} "
              f"${row['resistance']:<11.2f} "
              f"{row['distance_to_resistance']:<7.2%}")

    # Detailed analysis for top picks
    print("\n" + "="*120)
    print("DETAILED BREAKOUT ANALYSIS - TOP 5")
    print("="*120)

    for _, row in df.head(5).iterrows():
        print(f"\n{row['symbol']} - {row['company']}")
        print("-" * 60)
        print(f"Breakout Probability: {row['breakout_probability']:.2%}")
        print(f"Model Confidence: {row['confidence']:.2f}/1.00")
        print(f"Current Price: ${row['price']:.2f}")
        print(f"Resistance Level: ${row['resistance']:.2f} ({row['distance_to_resistance']:.2%} away)")
        print(f"Support Level: ${row['support']:.2f}")
        print(f"Volume Surge: {row['volume_surge']:.2f}x average")
        print(f"Volatility: {row['volatility']:.2%} daily")

        # Risk/Reward calculation
        potential_gain = row['distance_to_resistance']
        potential_loss = (row['price'] - row['support']) / row['price']
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else float('inf')

        print("\nRisk/Reward Analysis:")
        print(f"  Potential Gain: {potential_gain:.2%}")
        print(f"  Potential Loss: {potential_loss:.2%}")
        print(f"  Risk/Reward Ratio: {risk_reward:.2f}")

        # Entry strategy
        entry_price = row['price'] * 1.005  # 0.5% above current
        stop_loss = row['support'] * 0.98   # 2% below support
        target1 = row['price'] * 1.05       # 5% gain
        target2 = row['resistance'] * 0.98  # Just below resistance

        print("\nSuggested Entry Strategy:")
        print(f"  Entry: ${entry_price:.2f}")
        print(f"  Stop Loss: ${stop_loss:.2f} ({((stop_loss - entry_price) / entry_price):.2%})")
        print(f"  Target 1: ${target1:.2f} ({((target1 - entry_price) / entry_price):.2%})")
        print(f"  Target 2: ${target2:.2f} ({((target2 - entry_price) / entry_price):.2%})")

    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"breakout_predictions_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\nPredictions saved to: {filename}")

    # Generate trading plan
    generate_trading_plan(df.head(5))

def train_model(scanner, feature_eng, predictor):
    """Train the breakout prediction model"""

    # Get training symbols (e.g., S&P 500 components)
    training_symbols = scanner.get_sp500_symbols()[:100]  # Use 100 for demo

    training_data = []
    labels = []

    for symbol in training_symbols:
        print(f"Processing {symbol} for training...", end="\r")

        hist_data = scanner.get_historical_data(
            symbol=symbol,
            period="2Y",
            bar_size="1D"
        )

        if hist_data is None or len(hist_data) < 200:
            continue

        # Create training samples
        for i in range(100, len(hist_data) - 10):
            # Features from past 100 days
            feature_data = hist_data.iloc[i-100:i]
            features = feature_eng.create_features(feature_data)

            if features is None:
                continue

            # Label: Did it break out in next 5 days?
            future_data = hist_data.iloc[i:i+5]
            breakout = check_breakout(feature_data, future_data)

            training_data.append(features)
            labels.append(1 if breakout else 0)

    # Train model
    X = np.array(training_data)
    y = np.array(labels)

    predictor.train(X, y)
    print(f"\nModel trained on {len(X)} samples")

def calculate_resistance(data, lookback=20):
    """Calculate resistance level"""
    recent_highs = data['high'].rolling(window=lookback).max()
    resistance = recent_highs.iloc[-lookback:].max()
    return resistance

def calculate_support(data, lookback=20):
    """Calculate support level"""
    recent_lows = data['low'].rolling(window=lookback).min()
    support = recent_lows.iloc[-lookback:].min()
    return support

def check_breakout(past_data, future_data):
    """Check if a breakout occurred"""
    resistance = calculate_resistance(past_data)

    # Breakout: Close above resistance with volume
    for _, day in future_data.iterrows():
        if (day['close'] > resistance * 1.02 and
            day['volume'] > past_data['volume'].mean() * 1.5):
            return True
    return False

def generate_trading_plan(top_picks):
    """Generate a trading plan for top picks"""

    print("\n" + "="*80)
    print("AUTOMATED TRADING PLAN")
    print("="*80)

    total_capital = 100000  # $100k demo account
    position_size = total_capital / len(top_picks)

    print(f"\nCapital Allocation: ${total_capital:,.2f}")
    print(f"Positions: {len(top_picks)}")
    print(f"Position Size: ${position_size:,.2f} each")

    print("\nOrder Queue:")
    print("-" * 80)

    for i, (_, stock) in enumerate(top_picks.iterrows(), 1):
        shares = int(position_size / (stock['price'] * 1.005))
        entry = stock['price'] * 1.005
        stop = stock['support'] * 0.98
        target = stock['resistance'] * 0.98

        print(f"\n{i}. {stock['symbol']}:")
        print(f"   Buy {shares} shares at ${entry:.2f}")
        print(f"   Stop Loss: ${stop:.2f}")
        print(f"   Take Profit: ${target:.2f}")
        print(f"   Risk: ${(entry - stop) * shares:,.2f}")
        print(f"   Potential Profit: ${(target - entry) * shares:,.2f}")

if __name__ == "__main__":
    main()
