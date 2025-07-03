"""
Chart Pattern Detection Utilities
=================================
Functions for detecting common chart patterns.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
from scipy.signal import argrelextrema
from scipy.stats import linregress


@dataclass
class PatternResult:
    """Result of pattern detection"""
    pattern_type: str
    confidence: float
    start_index: int
    end_index: int
    start_date: datetime
    end_date: datetime
    key_points: List[Tuple[int, float]]
    metadata: Dict


def find_peaks_troughs(data: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks and troughs in price data.
    
    Args:
        data: Price series
        order: Number of points on each side to compare
        
    Returns:
        Tuple of (peaks_indices, troughs_indices)
    """
    peaks = argrelextrema(data.values, np.greater, order=order)[0]
    troughs = argrelextrema(data.values, np.less, order=order)[0]
    
    return peaks, troughs


def calculate_trendline(points: List[Tuple[int, float]], 
                       data_length: int) -> Tuple[float, float, float]:
    """
    Calculate trendline from points.
    
    Args:
        points: List of (index, value) tuples
        data_length: Length of the full dataset
        
    Returns:
        Tuple of (slope, intercept, r_squared)
    """
    if len(points) < 2:
        return 0, 0, 0
    
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    
    slope, intercept, r_value, _, _ = linregress(x, y)
    r_squared = r_value ** 2
    
    return slope, intercept, r_squared


def detect_triangle(data: pd.DataFrame, 
                   min_points: int = 4,
                   min_touches: int = 2) -> Optional[PatternResult]:
    """
    Detect triangle patterns (ascending, descending, symmetrical).
    
    Args:
        data: DataFrame with OHLC data
        min_points: Minimum number of points to form pattern
        min_touches: Minimum touches on each trendline
        
    Returns:
        PatternResult or None
    """
    if len(data) < 30:
        return None
    
    highs = data['high']
    lows = data['low']
    
    # Find peaks and troughs
    peaks, troughs = find_peaks_troughs(highs, order=5)
    _, trough_indices = find_peaks_troughs(lows, order=5)
    
    if len(peaks) < min_touches or len(trough_indices) < min_touches:
        return None
    
    # Get recent peaks and troughs
    recent_peaks = [(idx, highs.iloc[idx]) for idx in peaks[-min_touches:]]
    recent_troughs = [(idx, lows.iloc[idx]) for idx in trough_indices[-min_touches:]]
    
    # Calculate trendlines
    upper_slope, upper_intercept, upper_r2 = calculate_trendline(recent_peaks, len(data))
    lower_slope, lower_intercept, lower_r2 = calculate_trendline(recent_troughs, len(data))
    
    # Determine triangle type
    if upper_r2 < 0.8 or lower_r2 < 0.8:
        return None
    
    pattern_type = None
    confidence = 0
    
    # Ascending triangle: flat top, rising bottom
    if abs(upper_slope) < 0.001 and lower_slope > 0.001:
        pattern_type = "ascending_triangle"
        confidence = min(upper_r2, lower_r2) * 0.9
    
    # Descending triangle: falling top, flat bottom
    elif upper_slope < -0.001 and abs(lower_slope) < 0.001:
        pattern_type = "descending_triangle"
        confidence = min(upper_r2, lower_r2) * 0.9
    
    # Symmetrical triangle: converging lines
    elif upper_slope < -0.001 and lower_slope > 0.001:
        pattern_type = "symmetrical_triangle"
        confidence = min(upper_r2, lower_r2) * 0.85
    
    if pattern_type:
        start_idx = min(recent_peaks[0][0], recent_troughs[0][0])
        end_idx = len(data) - 1
        
        return PatternResult(
            pattern_type=pattern_type,
            confidence=confidence,
            start_index=start_idx,
            end_index=end_idx,
            start_date=data.index[start_idx],
            end_date=data.index[end_idx],
            key_points=recent_peaks + recent_troughs,
            metadata={
                'upper_slope': upper_slope,
                'lower_slope': lower_slope,
                'upper_r2': upper_r2,
                'lower_r2': lower_r2
            }
        )
    
    return None


def detect_flag(data: pd.DataFrame, 
               trend_period: int = 20,
               flag_period: int = 10) -> Optional[PatternResult]:
    """
    Detect flag patterns (bull flag, bear flag).
    
    Args:
        data: DataFrame with OHLC data
        trend_period: Period for the pole
        flag_period: Period for the flag
        
    Returns:
        PatternResult or None
    """
    if len(data) < trend_period + flag_period:
        return None
    
    # Split data into pole and flag sections
    pole_data = data.iloc[-trend_period-flag_period:-flag_period]
    flag_data = data.iloc[-flag_period:]
    
    # Calculate pole trend
    pole_return = (pole_data['close'].iloc[-1] - pole_data['close'].iloc[0]) / pole_data['close'].iloc[0]
    
    # Calculate flag characteristics
    flag_high = flag_data['high'].max()
    flag_low = flag_data['low'].min()
    flag_range = (flag_high - flag_low) / flag_low
    
    # Check for bull flag
    if pole_return > 0.1 and flag_range < 0.05:  # Strong up move, then consolidation
        # Calculate flag angle
        flag_slope, _, flag_r2 = calculate_trendline(
            [(i, flag_data['close'].iloc[i]) for i in range(len(flag_data))],
            len(flag_data)
        )
        
        if -0.01 < flag_slope < 0.01 and flag_r2 > 0.7:  # Slight downward or flat
            return PatternResult(
                pattern_type="bull_flag",
                confidence=0.8 * flag_r2,
                start_index=len(data) - trend_period - flag_period,
                end_index=len(data) - 1,
                start_date=data.index[-trend_period-flag_period],
                end_date=data.index[-1],
                key_points=[(len(data) - trend_period - flag_period, pole_data['close'].iloc[0]),
                           (len(data) - flag_period, pole_data['close'].iloc[-1])],
                metadata={
                    'pole_return': pole_return,
                    'flag_range': flag_range,
                    'flag_slope': flag_slope
                }
            )
    
    # Check for bear flag
    elif pole_return < -0.1 and flag_range < 0.05:  # Strong down move, then consolidation
        flag_slope, _, flag_r2 = calculate_trendline(
            [(i, flag_data['close'].iloc[i]) for i in range(len(flag_data))],
            len(flag_data)
        )
        
        if -0.01 < flag_slope < 0.01 and flag_r2 > 0.7:  # Slight upward or flat
            return PatternResult(
                pattern_type="bear_flag",
                confidence=0.8 * flag_r2,
                start_index=len(data) - trend_period - flag_period,
                end_index=len(data) - 1,
                start_date=data.index[-trend_period-flag_period],
                end_date=data.index[-1],
                key_points=[(len(data) - trend_period - flag_period, pole_data['close'].iloc[0]),
                           (len(data) - flag_period, pole_data['close'].iloc[-1])],
                metadata={
                    'pole_return': pole_return,
                    'flag_range': flag_range,
                    'flag_slope': flag_slope
                }
            )
    
    return None


def detect_head_and_shoulders(data: pd.DataFrame, 
                             min_pattern_bars: int = 30) -> Optional[PatternResult]:
    """
    Detect head and shoulders patterns.
    
    Args:
        data: DataFrame with OHLC data
        min_pattern_bars: Minimum bars for pattern
        
    Returns:
        PatternResult or None
    """
    if len(data) < min_pattern_bars:
        return None
    
    highs = data['high']
    lows = data['low']
    
    # Find peaks and troughs
    peaks, troughs = find_peaks_troughs(highs, order=5)
    
    if len(peaks) < 3 or len(troughs) < 2:
        return None
    
    # Look for head and shoulders pattern in recent peaks
    for i in range(len(peaks) - 2):
        left_shoulder_idx = peaks[i]
        head_idx = peaks[i + 1]
        right_shoulder_idx = peaks[i + 2]
        
        left_shoulder = highs.iloc[left_shoulder_idx]
        head = highs.iloc[head_idx]
        right_shoulder = highs.iloc[right_shoulder_idx]
        
        # Check if head is higher than shoulders
        if head > left_shoulder and head > right_shoulder:
            # Check if shoulders are roughly equal (within 3%)
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            
            if shoulder_diff < 0.03:
                # Find neckline (trough between shoulders)
                neckline_troughs = [t for t in troughs if left_shoulder_idx < t < right_shoulder_idx]
                
                if neckline_troughs:
                    neckline_level = lows.iloc[neckline_troughs].mean()
                    
                    # Calculate pattern height and confidence
                    pattern_height = (head - neckline_level) / neckline_level
                    confidence = 0.9 - shoulder_diff * 10  # Higher confidence for more symmetric shoulders
                    
                    return PatternResult(
                        pattern_type="head_and_shoulders",
                        confidence=confidence,
                        start_index=left_shoulder_idx,
                        end_index=right_shoulder_idx,
                        start_date=data.index[left_shoulder_idx],
                        end_date=data.index[right_shoulder_idx],
                        key_points=[
                            (left_shoulder_idx, left_shoulder),
                            (head_idx, head),
                            (right_shoulder_idx, right_shoulder)
                        ],
                        metadata={
                            'neckline': neckline_level,
                            'pattern_height': pattern_height,
                            'shoulder_symmetry': 1 - shoulder_diff
                        }
                    )
    
    # Check for inverse head and shoulders
    trough_vals = lows.iloc[troughs]
    for i in range(len(troughs) - 2):
        left_shoulder_idx = troughs[i]
        head_idx = troughs[i + 1]
        right_shoulder_idx = troughs[i + 2]
        
        left_shoulder = lows.iloc[left_shoulder_idx]
        head = lows.iloc[head_idx]
        right_shoulder = lows.iloc[right_shoulder_idx]
        
        # Check if head is lower than shoulders
        if head < left_shoulder and head < right_shoulder:
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            
            if shoulder_diff < 0.03:
                # Find neckline
                neckline_peaks = [p for p in peaks if left_shoulder_idx < p < right_shoulder_idx]
                
                if neckline_peaks:
                    neckline_level = highs.iloc[neckline_peaks].mean()
                    pattern_height = (neckline_level - head) / head
                    confidence = 0.9 - shoulder_diff * 10
                    
                    return PatternResult(
                        pattern_type="inverse_head_and_shoulders",
                        confidence=confidence,
                        start_index=left_shoulder_idx,
                        end_index=right_shoulder_idx,
                        start_date=data.index[left_shoulder_idx],
                        end_date=data.index[right_shoulder_idx],
                        key_points=[
                            (left_shoulder_idx, left_shoulder),
                            (head_idx, head),
                            (right_shoulder_idx, right_shoulder)
                        ],
                        metadata={
                            'neckline': neckline_level,
                            'pattern_height': pattern_height,
                            'shoulder_symmetry': 1 - shoulder_diff
                        }
                    )
    
    return None


def detect_double_top_bottom(data: pd.DataFrame, 
                           tolerance: float = 0.02) -> Optional[PatternResult]:
    """
    Detect double top and double bottom patterns.
    
    Args:
        data: DataFrame with OHLC data
        tolerance: Price tolerance for matching peaks/troughs
        
    Returns:
        PatternResult or None
    """
    if len(data) < 20:
        return None
    
    highs = data['high']
    lows = data['low']
    
    # Find peaks and troughs
    peaks, troughs = find_peaks_troughs(highs, order=5)
    _, trough_indices = find_peaks_troughs(lows, order=5)
    
    # Check for double top
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak2_idx = peaks[i + 1]
            
            peak1_val = highs.iloc[peak1_idx]
            peak2_val = highs.iloc[peak2_idx]
            
            # Check if peaks are similar in height
            peak_diff = abs(peak1_val - peak2_val) / peak1_val
            
            if peak_diff < tolerance:
                # Find valley between peaks
                valley_indices = [t for t in trough_indices if peak1_idx < t < peak2_idx]
                
                if valley_indices:
                    valley_val = lows.iloc[valley_indices].min()
                    pattern_depth = (peak1_val - valley_val) / peak1_val
                    
                    if pattern_depth > 0.03:  # Significant valley
                        confidence = 0.9 - peak_diff * 20
                        
                        return PatternResult(
                            pattern_type="double_top",
                            confidence=confidence,
                            start_index=peak1_idx,
                            end_index=peak2_idx,
                            start_date=data.index[peak1_idx],
                            end_date=data.index[peak2_idx],
                            key_points=[
                                (peak1_idx, peak1_val),
                                (peak2_idx, peak2_val)
                            ],
                            metadata={
                                'valley_level': valley_val,
                                'pattern_depth': pattern_depth,
                                'peak_similarity': 1 - peak_diff
                            }
                        )
    
    # Check for double bottom
    if len(trough_indices) >= 2:
        for i in range(len(trough_indices) - 1):
            trough1_idx = trough_indices[i]
            trough2_idx = trough_indices[i + 1]
            
            trough1_val = lows.iloc[trough1_idx]
            trough2_val = lows.iloc[trough2_idx]
            
            # Check if troughs are similar in depth
            trough_diff = abs(trough1_val - trough2_val) / trough1_val
            
            if trough_diff < tolerance:
                # Find peak between troughs
                peak_indices = [p for p in peaks if trough1_idx < p < trough2_idx]
                
                if peak_indices:
                    peak_val = highs.iloc[peak_indices].max()
                    pattern_height = (peak_val - trough1_val) / trough1_val
                    
                    if pattern_height > 0.03:  # Significant peak
                        confidence = 0.9 - trough_diff * 20
                        
                        return PatternResult(
                            pattern_type="double_bottom",
                            confidence=confidence,
                            start_index=trough1_idx,
                            end_index=trough2_idx,
                            start_date=data.index[trough1_idx],
                            end_date=data.index[trough2_idx],
                            key_points=[
                                (trough1_idx, trough1_val),
                                (trough2_idx, trough2_val)
                            ],
                            metadata={
                                'peak_level': peak_val,
                                'pattern_height': pattern_height,
                                'trough_similarity': 1 - trough_diff
                            }
                        )
    
    return None


def detect_cup_and_handle(data: pd.DataFrame, 
                         min_cup_depth: float = 0.15,
                         max_handle_retrace: float = 0.5) -> Optional[PatternResult]:
    """
    Detect cup and handle pattern.
    
    Args:
        data: DataFrame with OHLC data
        min_cup_depth: Minimum depth of cup (default: 15%)
        max_handle_retrace: Maximum handle retracement (default: 50%)
        
    Returns:
        PatternResult or None
    """
    if len(data) < 30:
        return None
    
    closes = data['close']
    highs = data['high']
    lows = data['low']
    
    # Find the highest point (left lip of cup)
    left_lip_idx = highs.idxmax()
    left_lip_val = highs.max()
    left_lip_pos = data.index.get_loc(left_lip_idx)
    
    if left_lip_pos < 10 or left_lip_pos > len(data) - 10:
        return None
    
    # Find the lowest point after left lip (bottom of cup)
    cup_bottom_idx = lows.iloc[left_lip_pos:].idxmin()
    cup_bottom_val = lows.iloc[left_lip_pos:].min()
    cup_bottom_pos = data.index.get_loc(cup_bottom_idx)
    
    # Calculate cup depth
    cup_depth = (left_lip_val - cup_bottom_val) / left_lip_val
    
    if cup_depth < min_cup_depth:
        return None
    
    # Find right lip (recovery to near left lip level)
    right_section = highs.iloc[cup_bottom_pos:]
    potential_right_lips = right_section[right_section > left_lip_val * 0.95]
    
    if len(potential_right_lips) == 0:
        return None
    
    right_lip_idx = potential_right_lips.index[0]
    right_lip_pos = data.index.get_loc(right_lip_idx)
    
    # Check for handle formation after right lip
    if right_lip_pos >= len(data) - 5:
        return None
    
    handle_section = data.iloc[right_lip_pos:]
    handle_low = handle_section['low'].min()
    
    # Calculate handle retracement
    handle_retrace = (highs.iloc[right_lip_pos] - handle_low) / (highs.iloc[right_lip_pos] - cup_bottom_val)
    
    if handle_retrace <= max_handle_retrace:
        # Check if cup is relatively symmetrical
        cup_left_bars = cup_bottom_pos - left_lip_pos
        cup_right_bars = right_lip_pos - cup_bottom_pos
        symmetry = min(cup_left_bars, cup_right_bars) / max(cup_left_bars, cup_right_bars)
        
        confidence = 0.7 + (symmetry * 0.2) + ((1 - handle_retrace) * 0.1)
        
        return PatternResult(
            pattern_type="cup_and_handle",
            confidence=min(confidence, 0.95),
            start_index=left_lip_pos,
            end_index=len(data) - 1,
            start_date=data.index[left_lip_pos],
            end_date=data.index[-1],
            key_points=[
                (left_lip_pos, left_lip_val),
                (cup_bottom_pos, cup_bottom_val),
                (right_lip_pos, highs.iloc[right_lip_pos])
            ],
            metadata={
                'cup_depth': cup_depth,
                'handle_retrace': handle_retrace,
                'symmetry': symmetry,
                'cup_duration': right_lip_pos - left_lip_pos
            }
        )
    
    return None


def detect_wedge(data: pd.DataFrame, min_points: int = 5) -> Optional[PatternResult]:
    """
    Detect wedge patterns (rising wedge, falling wedge).
    
    Args:
        data: DataFrame with OHLC data
        min_points: Minimum points to form pattern
        
    Returns:
        PatternResult or None
    """
    if len(data) < 20:
        return None
    
    highs = data['high']
    lows = data['low']
    
    # Find peaks and troughs
    peaks, troughs = find_peaks_troughs(highs, order=3)
    _, trough_indices = find_peaks_troughs(lows, order=3)
    
    if len(peaks) < 3 or len(trough_indices) < 3:
        return None
    
    # Get recent peaks and troughs
    recent_peaks = [(idx, highs.iloc[idx]) for idx in peaks[-3:]]
    recent_troughs = [(idx, lows.iloc[idx]) for idx in trough_indices[-3:]]
    
    # Calculate trendlines
    upper_slope, upper_intercept, upper_r2 = calculate_trendline(recent_peaks, len(data))
    lower_slope, lower_intercept, lower_r2 = calculate_trendline(recent_troughs, len(data))
    
    if upper_r2 < 0.8 or lower_r2 < 0.8:
        return None
    
    # Check for converging lines
    if abs(upper_slope - lower_slope) < 0.001:
        return None
    
    # Rising wedge: both lines rising, upper less steep
    if upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope:
        pattern_type = "rising_wedge"
        confidence = min(upper_r2, lower_r2) * 0.85
    
    # Falling wedge: both lines falling, lower less steep
    elif upper_slope < 0 and lower_slope < 0 and upper_slope < lower_slope:
        pattern_type = "falling_wedge"
        confidence = min(upper_r2, lower_r2) * 0.85
    
    else:
        return None
    
    start_idx = min(recent_peaks[0][0], recent_troughs[0][0])
    end_idx = len(data) - 1
    
    return PatternResult(
        pattern_type=pattern_type,
        confidence=confidence,
        start_index=start_idx,
        end_index=end_idx,
        start_date=data.index[start_idx],
        end_date=data.index[end_idx],
        key_points=recent_peaks + recent_troughs,
        metadata={
            'upper_slope': upper_slope,
            'lower_slope': lower_slope,
            'convergence_rate': abs(upper_slope - lower_slope)
        }
    )


def detect_channel(data: pd.DataFrame, 
                  min_touches: int = 3,
                  tolerance: float = 0.02) -> Optional[PatternResult]:
    """
    Detect channel patterns (ascending, descending, horizontal).
    
    Args:
        data: DataFrame with OHLC data
        min_touches: Minimum touches on each line
        tolerance: Tolerance for parallel lines
        
    Returns:
        PatternResult or None
    """
    if len(data) < 20:
        return None
    
    highs = data['high']
    lows = data['low']
    
    # Find peaks and troughs
    peaks, troughs = find_peaks_troughs(highs, order=5)
    _, trough_indices = find_peaks_troughs(lows, order=5)
    
    if len(peaks) < min_touches or len(trough_indices) < min_touches:
        return None
    
    # Get points for trendlines
    peak_points = [(idx, highs.iloc[idx]) for idx in peaks]
    trough_points = [(idx, lows.iloc[idx]) for idx in trough_indices]
    
    # Calculate trendlines
    upper_slope, upper_intercept, upper_r2 = calculate_trendline(peak_points, len(data))
    lower_slope, lower_intercept, lower_r2 = calculate_trendline(trough_points, len(data))
    
    if upper_r2 < 0.85 or lower_r2 < 0.85:
        return None
    
    # Check if lines are parallel (similar slopes)
    slope_diff = abs(upper_slope - lower_slope) / (abs(upper_slope) + 0.001)
    
    if slope_diff > tolerance:
        return None
    
    # Determine channel type
    avg_slope = (upper_slope + lower_slope) / 2
    
    if abs(avg_slope) < 0.001:
        pattern_type = "horizontal_channel"
    elif avg_slope > 0.001:
        pattern_type = "ascending_channel"
    else:
        pattern_type = "descending_channel"
    
    # Calculate channel width
    channel_width = abs(upper_intercept - lower_intercept) / ((upper_intercept + lower_intercept) / 2)
    
    confidence = min(upper_r2, lower_r2) * (1 - slope_diff)
    
    start_idx = min(peaks[0], trough_indices[0])
    end_idx = len(data) - 1
    
    return PatternResult(
        pattern_type=pattern_type,
        confidence=confidence,
        start_index=start_idx,
        end_index=end_idx,
        start_date=data.index[start_idx],
        end_date=data.index[end_idx],
        key_points=peak_points + trough_points,
        metadata={
            'upper_slope': upper_slope,
            'lower_slope': lower_slope,
            'channel_width': channel_width,
            'parallelism': 1 - slope_diff
        }
    )