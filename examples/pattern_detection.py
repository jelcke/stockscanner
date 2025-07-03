"""
AI Pattern Recognition for Stock Charts
Uses computer vision techniques to identify chart patterns
"""

import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class ChartPatternDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = MinMaxScaler()
        self.pattern_names = [
            'cup_and_handle', 'ascending_triangle', 'bull_flag',
            'double_bottom', 'head_shoulders', 'breakout'
        ]
        
        if model_path:
            self.load_model(model_path)
        else:
            self.model = self._build_model()
            
    def _build_model(self):
        """Build CNN model for pattern recognition"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.pattern_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def price_to_image(self, prices: List[float], volumes: List[float] = None) -> np.ndarray:
        """Convert price data to image format for pattern recognition"""
        if len(prices) < 20:
            raise ValueError("Need at least 20 price points")
            
        # Normalize prices
        prices_array = np.array(prices).reshape(-1, 1)
        normalized_prices = self.scaler.fit_transform(prices_array).flatten()
        
        # Create 64x64 image
        img = np.zeros((64, 64))
        
        # Map prices to y-coordinates
        x_coords = np.linspace(0, 63, len(normalized_prices)).astype(int)
        y_coords = (normalized_prices * 63).astype(int)
        
        # Draw price line
        for i in range(len(x_coords) - 1):
            cv2.line(img, 
                    (x_coords[i], 63 - y_coords[i]), 
                    (x_coords[i + 1], 63 - y_coords[i + 1]), 
                    255, 2)
                    
        # Add volume bars if provided
        if volumes:
            volume_array = np.array(volumes)
            normalized_volumes = (volume_array / volume_array.max()) * 20
            
            for i, (x, vol) in enumerate(zip(x_coords, normalized_volumes)):
                cv2.line(img, (x, 63), (x, int(63 - vol)), 128, 1)
                
        return img.reshape(64, 64, 1) / 255.0
        
    def detect_pattern(self, prices: List[float], volumes: List[float] = None) -> Dict[str, float]:
        """Detect chart pattern and return confidence scores"""
        try:
            image = self.price_to_image(prices, volumes)
            prediction = self.model.predict(image.reshape(1, 64, 64, 1), verbose=0)
            
            results = {}
            for i, pattern in enumerate(self.pattern_names):
                results[pattern] = float(prediction[0][i])
                
            return results
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {pattern: 0.0 for pattern in self.pattern_names}
            
    def get_dominant_pattern(self, prices: List[float], volumes: List[float] = None) -> Tuple[str, float]:
        """Get the most likely pattern and its confidence"""
        results = self.detect_pattern(prices, volumes)
        best_pattern = max(results, key=results.get)
        confidence = results[best_pattern]
        
        return best_pattern, confidence
        
    def is_breakout_pattern(self, prices: List[float], threshold=0.7) -> bool:
        """Check if current pattern indicates potential breakout"""
        breakout_patterns = ['cup_and_handle', 'ascending_triangle', 'bull_flag']
        results = self.detect_pattern(prices)
        
        for pattern in breakout_patterns:
            if results.get(pattern, 0) > threshold:
                return True
                
        return False

class PatternScanner:
    def __init__(self, detector: ChartPatternDetector):
        self.detector = detector
        self.historical_data = {}
        
    def add_price_data(self, symbol: str, price: float, volume: int, timestamp: datetime):
        """Add price point to historical data"""
        if symbol not in self.historical_data:
            self.historical_data[symbol] = {'prices': [], 'volumes': [], 'timestamps': []}
            
        data = self.historical_data[symbol]
        data['prices'].append(price)
        data['volumes'].append(volume)
        data['timestamps'].append(timestamp)
        
        # Keep only last 50 points for pattern analysis
        if len(data['prices']) > 50:
            data['prices'] = data['prices'][-50:]
            data['volumes'] = data['volumes'][-50:]
            data['timestamps'] = data['timestamps'][-50:]
            
    def scan_for_patterns(self, min_confidence=0.6) -> Dict[str, Dict]:
        """Scan all symbols for chart patterns"""
        results = {}
        
        for symbol, data in self.historical_data.items():
            if len(data['prices']) < 20:
                continue
                
            try:
                pattern, confidence = self.detector.get_dominant_pattern(
                    data['prices'], data['volumes']
                )
                
                if confidence > min_confidence:
                    results[symbol] = {
                        'pattern': pattern,
                        'confidence': confidence,
                        'current_price': data['prices'][-1],
                        'is_breakout': self.detector.is_breakout_pattern(data['prices'])
                    }
                    
            except Exception as e:
                logger.error(f"Pattern scan failed for {symbol}: {e}")
                
        return results
        
    def generate_pattern_alert(self, symbol: str, pattern_data: Dict) -> str:
        """Generate human-readable pattern alert"""
        pattern = pattern_data['pattern']
        confidence = pattern_data['confidence']
        price = pattern_data['current_price']
        
        alert = f"PATTERN ALERT: {symbol}\n"
        alert += f"Pattern: {pattern.replace('_', ' ').title()}\n"
        alert += f"Confidence: {confidence:.1%}\n"
        alert += f"Current Price: ${price:.2f}\n"
        
        if pattern_data.get('is_breakout'):
            alert += "🚀 POTENTIAL BREAKOUT DETECTED!"
            
        return alert

# Example usage
def demo_pattern_detection():
    # Create sample price data (ascending triangle pattern)
    prices = [100, 101, 102, 101, 103, 102, 104, 103, 105, 104, 
              106, 105, 107, 106, 108, 107, 109, 108, 110, 111]
    volumes = [1000, 1200, 800, 1500, 900, 1100, 1300, 950, 1400, 1000,
               1600, 1100, 1800, 1200, 2000, 1300, 2200, 1500, 2500, 3000]
               
    detector = ChartPatternDetector()
    pattern, confidence = detector.get_dominant_pattern(prices, volumes)
    
    print(f"Detected Pattern: {pattern}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Breakout Potential: {detector.is_breakout_pattern(prices)}")

if __name__ == "__main__":
    demo_pattern_detection()