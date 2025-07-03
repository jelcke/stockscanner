"""
Technical Indicators Utilities
==============================
Common technical indicators for stock analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: Price series
        period: Number of periods
        
    Returns:
        SMA series
    """
    return data.rolling(window=period, min_periods=1).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        data: Price series
        period: Number of periods
        
    Returns:
        EMA series
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        data: Price series
        period: Number of periods (default: 14)
        
    Returns:
        RSI series
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(data: pd.Series, 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price series
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
        
    Returns:
        DataFrame with MACD, Signal, and Histogram
    """
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def calculate_bollinger_bands(data: pd.Series, 
                            period: int = 20, 
                            std_dev: float = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price series
        period: SMA period (default: 20)
        std_dev: Number of standard deviations (default: 2)
        
    Returns:
        DataFrame with upper, middle, and lower bands
    """
    middle_band = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return pd.DataFrame({
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band,
        'bandwidth': (upper_band - lower_band) / middle_band,
        'percent_b': (data - lower_band) / (upper_band - lower_band)
    })


def calculate_atr(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods (default: 14)
        
    Returns:
        ATR series
    """
    # Calculate True Range
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr


def calculate_adx(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods (default: 14)
        
    Returns:
        DataFrame with ADX, +DI, and -DI
    """
    # Calculate directional movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    # Calculate True Range
    atr = calculate_atr(high, low, close, period)
    
    # Calculate directional indicators
    pos_di = 100 * calculate_ema(pos_dm, period) / atr
    neg_di = 100 * calculate_ema(neg_dm, period) / atr
    
    # Calculate ADX
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    adx = calculate_ema(dx, period)
    
    return pd.DataFrame({
        'adx': adx,
        'plus_di': pos_di,
        'minus_di': neg_di
    })


def calculate_stochastic(high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series,
                        k_period: int = 14, 
                        d_period: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)
        
    Returns:
        DataFrame with %K and %D
    """
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
    
    return pd.DataFrame({
        'k': k_percent,
        'd': d_percent
    })


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume.
    
    Args:
        close: Close price series
        volume: Volume series
        
    Returns:
        OBV series
    """
    obv = (volume * (~close.diff().le(0) * 2 - 1)).cumsum()
    return obv


def calculate_vwap(high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series, 
                  volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Weighted Average Price.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        
    Returns:
        VWAP series
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap


def calculate_pivot_points(high: pd.Series, 
                         low: pd.Series, 
                         close: pd.Series) -> pd.DataFrame:
    """
    Calculate Pivot Points.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        
    Returns:
        DataFrame with pivot levels
    """
    pivot = (high + low + close) / 3
    
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return pd.DataFrame({
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        's1': s1,
        's2': s2,
        's3': s3
    })


def calculate_fibonacci_retracement(high: float, low: float) -> dict:
    """
    Calculate Fibonacci retracement levels.
    
    Args:
        high: High price
        low: Low price
        
    Returns:
        Dictionary with Fibonacci levels
    """
    diff = high - low
    
    return {
        '0.0%': high,
        '23.6%': high - diff * 0.236,
        '38.2%': high - diff * 0.382,
        '50.0%': high - diff * 0.5,
        '61.8%': high - diff * 0.618,
        '78.6%': high - diff * 0.786,
        '100.0%': low
    }


def calculate_support_resistance(data: pd.DataFrame, 
                               window: int = 20, 
                               num_levels: int = 3) -> dict:
    """
    Calculate support and resistance levels.
    
    Args:
        data: DataFrame with OHLC data
        window: Rolling window for finding levels
        num_levels: Number of levels to return
        
    Returns:
        Dictionary with support and resistance levels
    """
    # Find local maxima and minima
    highs = data['high'].rolling(window=window, center=True).max()
    lows = data['low'].rolling(window=window, center=True).min()
    
    # Identify peaks and troughs
    peaks = data['high'][(data['high'] == highs) & (data['high'].shift() != highs)]
    troughs = data['low'][(data['low'] == lows) & (data['low'].shift() != lows)]
    
    # Get unique levels
    resistance_levels = sorted(peaks.unique(), reverse=True)[:num_levels]
    support_levels = sorted(troughs.unique())[:num_levels]
    
    return {
        'resistance': resistance_levels,
        'support': support_levels
    }


def calculate_money_flow(high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series, 
                        volume: pd.Series, 
                        period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: Number of periods (default: 14)
        
    Returns:
        MFI series
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    # Separate positive and negative money flow
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    
    # Calculate money flow ratio
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    
    mfr = positive_sum / negative_sum
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi


def calculate_cci(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods (default: 20)
        
    Returns:
        CCI series
    """
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )
    
    cci = (typical_price - sma) / (0.015 * mad)
    
    return cci


def calculate_williams_r(high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series, 
                        period: int = 14) -> pd.Series:
    """
    Calculate Williams %R.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods (default: 14)
        
    Returns:
        Williams %R series
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    return williams_r


def calculate_rate_of_change(data: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Rate of Change (ROC).
    
    Args:
        data: Price series
        period: Number of periods (default: 10)
        
    Returns:
        ROC series
    """
    roc = ((data - data.shift(period)) / data.shift(period)) * 100
    return roc


def calculate_volatility(data: pd.Series, 
                        period: int = 20, 
                        annualize: bool = True) -> pd.Series:
    """
    Calculate historical volatility.
    
    Args:
        data: Price series
        period: Number of periods (default: 20)
        annualize: Whether to annualize volatility (default: True)
        
    Returns:
        Volatility series
    """
    returns = data.pct_change()
    volatility = returns.rolling(window=period).std()
    
    if annualize:
        volatility = volatility * np.sqrt(252)  # Assuming 252 trading days
    
    return volatility