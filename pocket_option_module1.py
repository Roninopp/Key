"""
MODULE 1: IMAGE PROCESSING & PRICE ACTION ANALYSIS
Pocket Option Chart Analyzer - Focus on PRICE ACTION
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class CandleType(Enum):
    BULLISH_ENGULFING = "Bullish Engulfing"
    BEARISH_ENGULFING = "Bearish Engulfing"
    HAMMER = "Hammer"
    SHOOTING_STAR = "Shooting Star"
    DOJI = "Doji"
    MORNING_STAR = "Morning Star"
    EVENING_STAR = "Evening Star"
    BULLISH_PIN_BAR = "Bullish Pin Bar"
    BEARISH_PIN_BAR = "Bearish Pin Bar"
    INSIDE_BAR = "Inside Bar"
    OUTSIDE_BAR = "Outside Bar"

class TrendDirection(Enum):
    STRONG_UPTREND = "Strong Uptrend"
    UPTREND = "Uptrend"
    SIDEWAYS = "Sideways"
    DOWNTREND = "Downtrend"
    STRONG_DOWNTREND = "Strong Downtrend"

@dataclass
class Candle:
    """Single candlestick data"""
    open: float
    high: float
    low: float
    close: float
    x: int  # Position on chart
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

@dataclass
class PriceActionAnalysis:
    """Complete price action analysis result"""
    detected_patterns: List[Tuple[CandleType, float]]  # Pattern + Strength
    trend: TrendDirection
    trend_strength: float  # 0-100
    support_level: Optional[float]
    resistance_level: Optional[float]
    key_level_distance: float  # Distance to nearest S/R
    momentum: str  # "Increasing", "Decreasing", "Neutral"
    market_structure: str  # "HH/HL", "LL/LH", "Ranging"
    price_rejection_zones: List[float]
    candles: List[Candle]

class PocketOptionChartAnalyzer:
    """
    Main analyzer for Pocket Option charts
    FOCUS: PRICE ACTION FIRST, Indicators SECONDARY
    """
    
    def __init__(self):
        self.min_candles = 20  # Minimum candles needed for analysis
        self.support_resistance_touches = 2  # Touches to confirm S/R
        
    def analyze_chart(self, image_path: str) -> PriceActionAnalysis:
        """
        Main analysis function - extracts and analyzes price action
        """
        # Step 1: Load and preprocess image
        img = self._load_image(image_path)
        
        # Step 2: Extract candlesticks from image
        candles = self._extract_candles(img)
        
        if len(candles) < self.min_candles:
            raise ValueError(f"Need at least {self.min_candles} candles for analysis")
        
        # Step 3: PRICE ACTION ANALYSIS (CORE)
        patterns = self._detect_patterns(candles)
        trend = self._analyze_trend(candles)
        trend_strength = self._calculate_trend_strength(candles)
        support, resistance = self._find_support_resistance(candles)
        momentum = self._analyze_momentum(candles)
        structure = self._analyze_market_structure(candles)
        rejection_zones = self._find_rejection_zones(candles)
        key_distance = self._distance_to_key_level(candles, support, resistance)
        
        return PriceActionAnalysis(
            detected_patterns=patterns,
            trend=trend,
            trend_strength=trend_strength,
            support_level=support,
            resistance_level=resistance,
            key_level_distance=key_distance,
            momentum=momentum,
            market_structure=structure,
            price_rejection_zones=rejection_zones,
            candles=candles
        )
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess chart image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        return img
    
    def _extract_candles(self, img: np.ndarray) -> List[Candle]:
        """
        Extract candlestick data from chart image
        Uses computer vision to detect candle bodies and wicks
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to detect candles
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (candle bodies)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candles = []
        
        # Sort contours by x position (left to right)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out noise (too small objects)
            if w < 3 or h < 5:
                continue
            
            # Extract candle data
            # This is simplified - real implementation needs more sophisticated detection
            candle_high = self._extract_price_at_y(img, y)
            candle_low = self._extract_price_at_y(img, y + h)
            
            # Detect if bullish or bearish by color
            roi = img[y:y+h, x:x+w]
            avg_color = np.mean(roi, axis=(0,1))
            is_green = avg_color[1] > avg_color[2]  # Green > Red in BGR
            
            if is_green:
                candle_close = candle_high
                candle_open = candle_low
            else:
                candle_close = candle_low
                candle_open = candle_high
            
            candle = Candle(
                open=candle_open,
                high=candle_high,
                low=candle_low,
                close=candle_close,
                x=x
            )
            candles.append(candle)
        
        return candles[-50:]  # Return last 50 candles
    
    def _extract_price_at_y(self, img: np.ndarray, y: int) -> float:
        """
        Extract price value at specific Y coordinate using OCR
        This is simplified - real implementation needs chart boundary detection
        """
        # Mock implementation - replace with actual OCR
        # In real version: crop price axis area, apply OCR
        img_height = img.shape[0]
        price_range = 100  # Assume 100 pips range
        normalized_y = y / img_height
        return 1.0850 + (1 - normalized_y) * 0.01  # Mock price
    
    def _detect_patterns(self, candles: List[Candle]) -> List[Tuple[CandleType, float]]:
        """
        CRITICAL: Detect candlestick patterns with strength score
        This is the CORE of price action analysis
        """
        patterns = []
        
        if len(candles) < 3:
            return patterns
        
        # Check last 3 candles for patterns
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        # ENGULFING PATTERNS (Very Strong)
        bull_engulf = self._is_bullish_engulfing(c2, c3)
        if bull_engulf > 0:
            patterns.append((CandleType.BULLISH_ENGULFING, bull_engulf))
        
        bear_engulf = self._is_bearish_engulfing(c2, c3)
        if bear_engulf > 0:
            patterns.append((CandleType.BEARISH_ENGULFING, bear_engulf))
        
        # PIN BAR (Rejection patterns)
        bull_pin = self._is_bullish_pin_bar(c3)
        if bull_pin > 0:
            patterns.append((CandleType.BULLISH_PIN_BAR, bull_pin))
        
        bear_pin = self._is_bearish_pin_bar(c3)
        if bear_pin > 0:
            patterns.append((CandleType.BEARISH_PIN_BAR, bear_pin))
        
        # HAMMER & SHOOTING STAR
        hammer = self._is_hammer(c3)
        if hammer > 0:
            patterns.append((CandleType.HAMMER, hammer))
        
        shooting = self._is_shooting_star(c3)
        if shooting > 0:
            patterns.append((CandleType.SHOOTING_STAR, shooting))
        
        # DOJI (Indecision)
        doji = self._is_doji(c3)
        if doji > 0:
            patterns.append((CandleType.DOJI, doji))
        
        # STAR PATTERNS (3-candle)
        morning = self._is_morning_star(c1, c2, c3)
        if morning > 0:
            patterns.append((CandleType.MORNING_STAR, morning))
        
        evening = self._is_evening_star(c1, c2, c3)
        if evening > 0:
            patterns.append((CandleType.EVENING_STAR, evening))
        
        # INSIDE/OUTSIDE BAR
        inside = self._is_inside_bar(c2, c3)
        if inside > 0:
            patterns.append((CandleType.INSIDE_BAR, inside))
        
        outside = self._is_outside_bar(c2, c3)
        if outside > 0:
            patterns.append((CandleType.OUTSIDE_BAR, outside))
        
        return patterns
    
    def _is_bullish_engulfing(self, c1: Candle, c2: Candle) -> float:
        """Bullish engulfing pattern - strength 0-100"""
        if not c1.is_bearish or not c2.is_bullish:
            return 0
        
        # c2 must engulf c1
        if c2.open <= c1.close and c2.close >= c1.open:
            # Calculate strength based on size
            engulf_ratio = c2.body_size / c1.body_size if c1.body_size > 0 else 1
            strength = min(engulf_ratio * 50, 100)
            return strength
        return 0
    
    def _is_bearish_engulfing(self, c1: Candle, c2: Candle) -> float:
        """Bearish engulfing pattern - strength 0-100"""
        if not c1.is_bullish or not c2.is_bearish:
            return 0
        
        if c2.open >= c1.close and c2.close <= c1.open:
            engulf_ratio = c2.body_size / c1.body_size if c1.body_size > 0 else 1
            strength = min(engulf_ratio * 50, 100)
            return strength
        return 0
    
    def _is_bullish_pin_bar(self, c: Candle) -> float:
        """Bullish pin bar - long lower wick, small body"""
        if c.total_range == 0:
            return 0
        
        # Lower wick should be 2x+ body size
        if c.lower_wick > c.body_size * 2 and c.upper_wick < c.body_size:
            wick_ratio = c.lower_wick / c.total_range
            strength = wick_ratio * 100
            return min(strength, 100)
        return 0
    
    def _is_bearish_pin_bar(self, c: Candle) -> float:
        """Bearish pin bar - long upper wick, small body"""
        if c.total_range == 0:
            return 0
        
        if c.upper_wick > c.body_size * 2 and c.lower_wick < c.body_size:
            wick_ratio = c.upper_wick / c.total_range
            strength = wick_ratio * 100
            return min(strength, 100)
        return 0
    
    def _is_hammer(self, c: Candle) -> float:
        """Hammer pattern at support"""
        if c.total_range == 0:
            return 0
        
        # Long lower wick (2x body), short upper wick
        if c.lower_wick > c.body_size * 2 and c.upper_wick < c.body_size * 0.5:
            return 80
        return 0
    
    def _is_shooting_star(self, c: Candle) -> float:
        """Shooting star at resistance"""
        if c.total_range == 0:
            return 0
        
        if c.upper_wick > c.body_size * 2 and c.lower_wick < c.body_size * 0.5:
            return 80
        return 0
    
    def _is_doji(self, c: Candle) -> float:
        """Doji - indecision candle"""
        if c.total_range == 0:
            return 0
        
        # Body is very small compared to range
        if c.body_size < c.total_range * 0.1:
            return 60
        return 0
    
    def _is_morning_star(self, c1: Candle, c2: Candle, c3: Candle) -> float:
        """Morning star - bullish reversal"""
        if not (c1.is_bearish and c2.body_size < c1.body_size * 0.5 and c3.is_bullish):
            return 0
        
        if c3.close > (c1.open + c1.close) / 2:
            return 85
        return 0
    
    def _is_evening_star(self, c1: Candle, c2: Candle, c3: Candle) -> float:
        """Evening star - bearish reversal"""
        if not (c1.is_bullish and c2.body_size < c1.body_size * 0.5 and c3.is_bearish):
            return 0
        
        if c3.close < (c1.open + c1.close) / 2:
            return 85
        return 0
    
    def _is_inside_bar(self, c1: Candle, c2: Candle) -> float:
        """Inside bar - consolidation"""
        if c2.high < c1.high and c2.low > c1.low:
            return 70
        return 0
    
    def _is_outside_bar(self, c1: Candle, c2: Candle) -> float:
        """Outside bar - volatility expansion"""
        if c2.high > c1.high and c2.low < c1.low:
            return 75
        return 0
    
    def _analyze_trend(self, candles: List[Candle]) -> TrendDirection:
        """Analyze overall trend using Higher Highs/Lower Lows"""
        if len(candles) < 10:
            return TrendDirection.SIDEWAYS
        
        recent = candles[-10:]
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        
        # Simple trend: compare first half vs second half
        first_half_high = max(highs[:5])
        second_half_high = max(highs[5:])
        first_half_low = min(lows[:5])
        second_half_low = min(lows[5:])
        
        # Higher highs and higher lows = uptrend
        if second_half_high > first_half_high and second_half_low > first_half_low:
            price_change = (second_half_high - first_half_high) / first_half_high
            return TrendDirection.STRONG_UPTREND if price_change > 0.01 else TrendDirection.UPTREND
        
        # Lower highs and lower lows = downtrend
        elif second_half_high < first_half_high and second_half_low < first_half_low:
            price_change = (first_half_high - second_half_high) / first_half_high
            return TrendDirection.STRONG_DOWNTREND if price_change > 0.01 else TrendDirection.DOWNTREND
        
        return TrendDirection.SIDEWAYS
    
    def _calculate_trend_strength(self, candles: List[Candle]) -> float:
        """Calculate trend strength 0-100"""
        if len(candles) < 5:
            return 0
        
        # Count consecutive candles in same direction
        recent = candles[-5:]
        bullish_count = sum(1 for c in recent if c.is_bullish)
        
        if bullish_count >= 4:
            return 85  # Strong uptrend
        elif bullish_count <= 1:
            return 85  # Strong downtrend
        elif bullish_count == 3 or bullish_count == 2:
            return 50  # Moderate trend
        return 30  # Weak/Sideways
    
    def _find_support_resistance(self, candles: List[Candle]) -> Tuple[Optional[float], Optional[float]]:
        """Find key support and resistance levels"""
        if len(candles) < 20:
            return None, None
        
        # Get all highs and lows
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        # Find price levels that were touched multiple times
        price_range = max(highs) - min(lows)
        tolerance = price_range * 0.001  # 0.1% tolerance
        
        # Find resistance (high that was tested multiple times)
        resistance_candidates = []
        for h in highs:
            touches = sum(1 for x in highs if abs(x - h) <= tolerance)
            if touches >= self.support_resistance_touches:
                resistance_candidates.append(h)
        
        resistance = max(resistance_candidates) if resistance_candidates else None
        
        # Find support
        support_candidates = []
        for l in lows:
            touches = sum(1 for x in lows if abs(x - l) <= tolerance)
            if touches >= self.support_resistance_touches:
                support_candidates.append(l)
        
        support = min(support_candidates) if support_candidates else None
        
        return support, resistance
    
    def _distance_to_key_level(self, candles: List[Candle], 
                               support: Optional[float], 
                               resistance: Optional[float]) -> float:
        """Calculate distance to nearest S/R as percentage"""
        if not candles:
            return 0
        
        current_price = candles[-1].close
        
        distances = []
        if support:
            distances.append(abs(current_price - support) / current_price * 100)
        if resistance:
            distances.append(abs(current_price - resistance) / current_price * 100)
        
        return min(distances) if distances else 0
    
    def _analyze_momentum(self, candles: List[Candle]) -> str:
        """Analyze price momentum"""
        if len(candles) < 5:
            return "Neutral"
        
        recent = candles[-5:]
        body_sizes = [c.body_size for c in recent]
        
        # Increasing body sizes = increasing momentum
        if body_sizes[-1] > body_sizes[-2] > body_sizes[-3]:
            return "Increasing"
        elif body_sizes[-1] < body_sizes[-2] < body_sizes[-3]:
            return "Decreasing"
        
        return "Neutral"
    
    def _analyze_market_structure(self, candles: List[Candle]) -> str:
        """Analyze market structure: HH/HL, LL/LH, or Ranging"""
        if len(candles) < 10:
            return "Ranging"
        
        recent = candles[-10:]
        
        # Find swing highs and lows
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        
        # Check for Higher Highs and Higher Lows
        if max(highs[-5:]) > max(highs[:5]) and min(lows[-5:]) > min(lows[:5]):
            return "HH/HL (Uptrend)"
        
        # Check for Lower Lows and Lower Highs
        elif max(highs[-5:]) < max(highs[:5]) and min(lows[-5:]) < min(lows[:5]):
            return "LL/LH (Downtrend)"
        
        return "Ranging"
    
    def _find_rejection_zones(self, candles: List[Candle]) -> List[float]:
        """Find price levels where rejection occurred (long wicks)"""
        rejection_zones = []
        
        for candle in candles[-10:]:
            # Upper rejection (long upper wick)
            if candle.upper_wick > candle.body_size * 2:
                rejection_zones.append(candle.high)
            
            # Lower rejection (long lower wick)
            if candle.lower_wick > candle.body_size * 2:
                rejection_zones.append(candle.low)
        
        return rejection_zones


# USAGE EXAMPLE
if __name__ == "__main__":
    analyzer = PocketOptionChartAnalyzer()
    
    # Analyze chart
    try:
        result = analyzer.analyze_chart("chart.png")
        
        print("=" * 50)
        print("PRICE ACTION ANALYSIS")
        print("=" * 50)
        
        print(f"\n📊 TREND: {result.trend.value}")
        print(f"💪 Trend Strength: {result.trend_strength:.1f}%")
        print(f"⚡ Momentum: {result.momentum}")
        print(f"🏗️ Market Structure: {result.market_structure}")
        
        print(f"\n🎯 KEY LEVELS:")
        if result.support_level:
            print(f"   Support: {result.support_level:.5f}")
        if result.resistance_level:
            print(f"   Resistance: {result.resistance_level:.5f}")
        print(f"   Distance to Key Level: {result.key_level_distance:.2f}%")
        
        print(f"\n📍 DETECTED PATTERNS:")
        for pattern, strength in result.detected_patterns:
            print(f"   ✓ {pattern.value} (Strength: {strength:.1f}%)")
        
        if result.price_rejection_zones:
            print(f"\n🚫 REJECTION ZONES: {len(result.price_rejection_zones)} detected")
        
    except Exception as e:
        print(f"Error: {e}")
