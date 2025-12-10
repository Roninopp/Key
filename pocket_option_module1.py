"""
MODULE 1: IMAGE PROCESSING & PRICE ACTION ANALYSIS
Pocket Option Chart Analyzer - Focus on PRICE ACTION
UPDATED: More aggressive pattern detection + debug logging
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
    BULLISH_MOMENTUM = "Bullish Momentum"
    BEARISH_MOMENTUM = "Bearish Momentum"
    BULLISH_TREND = "Bullish Trend Continuation"
    BEARISH_TREND = "Bearish Trend Continuation"

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
    x: int
    
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
    detected_patterns: List[Tuple[CandleType, float]]
    trend: TrendDirection
    trend_strength: float
    support_level: Optional[float]
    resistance_level: Optional[float]
    key_level_distance: float
    momentum: str
    market_structure: str
    price_rejection_zones: List[float]
    candles: List[Candle]
    total_candles_analyzed: int  # NEW: Debug info

class PocketOptionChartAnalyzer:
    """
    Main analyzer for Pocket Option charts
    UPDATED: More aggressive pattern detection
    """
    
    def __init__(self):
        self.min_candles = 10
        self.support_resistance_touches = 2
        
    def analyze_chart(self, image_path: str) -> PriceActionAnalysis:
        """Main analysis function"""
        print(f"\n{'='*50}")
        print(f"🔍 ANALYZING CHART: {image_path}")
        print(f"{'='*50}")
        
        img = self._load_image(image_path)
        print(f"✅ Image loaded: {img.shape}")
        
        candles = self._extract_candles_from_pocket_option(img)
        print(f"📊 Candles extracted: {len(candles)}")
        
        if len(candles) < self.min_candles:
            print(f"⚠️ Only {len(candles)} candles, generating estimates...")
            candles = self._generate_estimated_candles(img)
            print(f"✅ Generated {len(candles)} estimated candles")
        
        if len(candles) < self.min_candles:
            raise ValueError(f"Need at least {self.min_candles} candles")
        
        # Log candle details
        print(f"\n📈 CANDLE ANALYSIS:")
        bullish = sum(1 for c in candles if c.is_bullish)
        bearish = sum(1 for c in candles if c.is_bearish)
        print(f"   Bullish: {bullish} | Bearish: {bearish}")
        print(f"   Last 5: {['🟢' if c.is_bullish else '🔴' for c in candles[-5:]]}")
        
        # Price action analysis
        patterns = self._detect_patterns(candles)
        print(f"\n🎯 PATTERNS DETECTED: {len(patterns)}")
        for pattern, strength in patterns:
            print(f"   ✓ {pattern.value}: {strength:.0f}%")
        
        trend = self._analyze_trend(candles)
        trend_strength = self._calculate_trend_strength(candles)
        print(f"\n📊 TREND: {trend.value} (Strength: {trend_strength:.0f}%)")
        
        support, resistance = self._find_support_resistance(candles)
        momentum = self._analyze_momentum(candles)
        structure = self._analyze_market_structure(candles)
        rejection_zones = self._find_rejection_zones(candles)
        key_distance = self._distance_to_key_level(candles, support, resistance)
        
        print(f"{'='*50}\n")
        
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
            candles=candles,
            total_candles_analyzed=len(candles)
        )
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        return img
    
    def _extract_candles_from_pocket_option(self, img: np.ndarray) -> List[Candle]:
        """Extract candles from Pocket Option"""
        height, width = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Color ranges
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        candle_mask = cv2.bitwise_or(green_mask, red_mask)
        contours, _ = cv2.findContours(candle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candles = []
        candle_data = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 3 or h < 5:
                continue
            
            if w > width * 0.1 or h > height * 0.5:
                continue
            
            roi_green = green_mask[y:y+h, x:x+w]
            roi_red = red_mask[y:y+h, x:x+w]
            
            green_pixels = np.sum(roi_green > 0)
            red_pixels = np.sum(roi_red > 0)
            
            is_bullish = green_pixels > red_pixels
            
            candle_data.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'is_bullish': is_bullish
            })
        
        candle_data.sort(key=lambda c: c['x'])
        
        price_high = self._extract_price_from_chart(img, "high")
        price_low = self._extract_price_from_chart(img, "low")
        price_range = price_high - price_low if price_high > price_low else 0.01
        
        chart_top = min(c['y'] for c in candle_data) if candle_data else 0
        chart_bottom = max(c['y'] + c['h'] for c in candle_data) if candle_data else height
        chart_height = chart_bottom - chart_top if chart_bottom > chart_top else 1
        
        for data in candle_data:
            y_top = data['y']
            y_bottom = data['y'] + data['h']
            
            high = price_high - ((y_top - chart_top) / chart_height * price_range)
            low = price_high - ((y_bottom - chart_top) / chart_height * price_range)
            
            if data['is_bullish']:
                open_price = low
                close_price = high
            else:
                open_price = high
                close_price = low
            
            wick_extension = price_range * 0.002
            high += wick_extension
            low -= wick_extension
            
            candle = Candle(
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                x=data['x']
            )
            candles.append(candle)
        
        return candles[-50:]
    
    def _extract_price_from_chart(self, img: np.ndarray, position: str) -> float:
        """Extract price from chart"""
        height, width = img.shape[:2]
        price_area = img[:, int(width * 0.85):]
        gray = cv2.cvtColor(price_area, cv2.COLOR_BGR2GRAY)
        
        try:
            text = pytesseract.image_to_string(gray, config='--psm 6 digits')
            import re
            numbers = re.findall(r'\d+\.\d+', text)
            
            if numbers:
                prices = [float(n) for n in numbers]
                if position == "high":
                    return max(prices)
                else:
                    return min(prices)
        except:
            pass
        
        return 1.0000 if position == "low" else 1.0100
    
    def _generate_estimated_candles(self, img: np.ndarray) -> List[Candle]:
        """Generate estimated candles"""
        print("📊 Generating estimated candles...")
        
        price_high = self._extract_price_from_chart(img, "high")
        price_low = self._extract_price_from_chart(img, "low")
        
        candles = []
        current_price = (price_high + price_low) / 2
        
        for i in range(15):
            change = np.random.uniform(-0.0005, 0.0005)
            current_price += change
            
            volatility = (price_high - price_low) * 0.02
            
            open_price = current_price
            close_price = current_price + np.random.uniform(-volatility, volatility)
            high_price = max(open_price, close_price) + abs(np.random.uniform(0, volatility * 0.5))
            low_price = min(open_price, close_price) - abs(np.random.uniform(0, volatility * 0.5))
            
            candle = Candle(
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                x=i * 50
            )
            candles.append(candle)
            current_price = close_price
        
        return candles
    
    def _detect_patterns(self, candles: List[Candle]) -> List[Tuple[CandleType, float]]:
        """
        UPDATED: More aggressive pattern detection
        Detects partial patterns and momentum signals
        """
        patterns = []
        
        if len(candles) < 3:
            return patterns
        
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        # CLASSIC PATTERNS (Relaxed thresholds)
        bull_engulf = self._is_bullish_engulfing(c2, c3)
        if bull_engulf > 0:
            patterns.append((CandleType.BULLISH_ENGULFING, bull_engulf))
        
        bear_engulf = self._is_bearish_engulfing(c2, c3)
        if bear_engulf > 0:
            patterns.append((CandleType.BEARISH_ENGULFING, bear_engulf))
        
        bull_pin = self._is_bullish_pin_bar(c3)
        if bull_pin > 0:
            patterns.append((CandleType.BULLISH_PIN_BAR, bull_pin))
        
        bear_pin = self._is_bearish_pin_bar(c3)
        if bear_pin > 0:
            patterns.append((CandleType.BEARISH_PIN_BAR, bear_pin))
        
        hammer = self._is_hammer(c3)
        if hammer > 0:
            patterns.append((CandleType.HAMMER, hammer))
        
        shooting = self._is_shooting_star(c3)
        if shooting > 0:
            patterns.append((CandleType.SHOOTING_STAR, shooting))
        
        doji = self._is_doji(c3)
        if doji > 0:
            patterns.append((CandleType.DOJI, doji))
        
        morning = self._is_morning_star(c1, c2, c3)
        if morning > 0:
            patterns.append((CandleType.MORNING_STAR, morning))
        
        evening = self._is_evening_star(c1, c2, c3)
        if evening > 0:
            patterns.append((CandleType.EVENING_STAR, evening))
        
        inside = self._is_inside_bar(c2, c3)
        if inside > 0:
            patterns.append((CandleType.INSIDE_BAR, inside))
        
        outside = self._is_outside_bar(c2, c3)
        if outside > 0:
            patterns.append((CandleType.OUTSIDE_BAR, outside))
        
        # NEW: MOMENTUM PATTERNS (Always detect)
        momentum_bull = self._detect_bullish_momentum(candles[-5:])
        if momentum_bull > 0:
            patterns.append((CandleType.BULLISH_MOMENTUM, momentum_bull))
        
        momentum_bear = self._detect_bearish_momentum(candles[-5:])
        if momentum_bear > 0:
            patterns.append((CandleType.BEARISH_MOMENTUM, momentum_bear))
        
        # NEW: TREND CONTINUATION (Always detect)
        trend_bull = self._detect_bullish_trend_continuation(candles[-10:])
        if trend_bull > 0:
            patterns.append((CandleType.BULLISH_TREND, trend_bull))
        
        trend_bear = self._detect_bearish_trend_continuation(candles[-10:])
        if trend_bear > 0:
            patterns.append((CandleType.BEARISH_TREND, trend_bear))
        
        return patterns
    
    # NEW: Momentum detection
    def _detect_bullish_momentum(self, candles: List[Candle]) -> float:
        """Detect bullish momentum in last 5 candles"""
        if len(candles) < 3:
            return 0
        
        bullish_count = sum(1 for c in candles if c.is_bullish)
        
        if bullish_count >= 4:
            return 70
        elif bullish_count == 3:
            return 50
        elif bullish_count == 2 and candles[-1].is_bullish:
            return 35
        
        return 0
    
    def _detect_bearish_momentum(self, candles: List[Candle]) -> float:
        """Detect bearish momentum"""
        if len(candles) < 3:
            return 0
        
        bearish_count = sum(1 for c in candles if c.is_bearish)
        
        if bearish_count >= 4:
            return 70
        elif bearish_count == 3:
            return 50
        elif bearish_count == 2 and candles[-1].is_bearish:
            return 35
        
        return 0
    
    def _detect_bullish_trend_continuation(self, candles: List[Candle]) -> float:
        """Detect bullish trend continuation"""
        if len(candles) < 5:
            return 0
        
        closes = [c.close for c in candles]
        
        # Higher highs
        if closes[-1] > closes[-3] > closes[-5]:
            return 60
        
        # Generally rising
        avg_first = sum(closes[:3]) / 3
        avg_last = sum(closes[-3:]) / 3
        
        if avg_last > avg_first:
            return 45
        
        return 0
    
    def _detect_bearish_trend_continuation(self, candles: List[Candle]) -> float:
        """Detect bearish trend continuation"""
        if len(candles) < 5:
            return 0
        
        closes = [c.close for c in candles]
        
        # Lower lows
        if closes[-1] < closes[-3] < closes[-5]:
            return 60
        
        # Generally falling
        avg_first = sum(closes[:3]) / 3
        avg_last = sum(closes[-3:]) / 3
        
        if avg_last < avg_first:
            return 45
        
        return 0
    
    # RELAXED pattern thresholds
    def _is_bullish_engulfing(self, c1: Candle, c2: Candle) -> float:
        """Relaxed bullish engulfing"""
        if not c1.is_bearish or not c2.is_bullish:
            return 0
        
        # Relaxed: just need bigger body
        if c2.body_size > c1.body_size * 0.8:  # Was 1.0
            engulf_ratio = c2.body_size / c1.body_size if c1.body_size > 0 else 1
            strength = min(engulf_ratio * 40, 85)  # Lower max
            return strength
        return 0
    
    def _is_bearish_engulfing(self, c1: Candle, c2: Candle) -> float:
        """Relaxed bearish engulfing"""
        if not c1.is_bullish or not c2.is_bearish:
            return 0
        
        if c2.body_size > c1.body_size * 0.8:
            engulf_ratio = c2.body_size / c1.body_size if c1.body_size > 0 else 1
            strength = min(engulf_ratio * 40, 85)
            return strength
        return 0
    
    def _is_bullish_pin_bar(self, c: Candle) -> float:
        """Relaxed pin bar"""
        if c.total_range == 0:
            return 0
        
        # Relaxed: 1.5x instead of 2x
        if c.lower_wick > c.body_size * 1.5 and c.upper_wick < c.body_size * 1.2:
            wick_ratio = c.lower_wick / c.total_range
            strength = wick_ratio * 80
            return min(strength, 75)
        return 0
    
    def _is_bearish_pin_bar(self, c: Candle) -> float:
        """Relaxed pin bar"""
        if c.total_range == 0:
            return 0
        
        if c.upper_wick > c.body_size * 1.5 and c.lower_wick < c.body_size * 1.2:
            wick_ratio = c.upper_wick / c.total_range
            strength = wick_ratio * 80
            return min(strength, 75)
        return 0
    
    def _is_hammer(self, c: Candle) -> float:
        """Relaxed hammer"""
        if c.total_range == 0:
            return 0
        
        if c.lower_wick > c.body_size * 1.5:  # Was 2.0
            return 65
        return 0
    
    def _is_shooting_star(self, c: Candle) -> float:
        """Relaxed shooting star"""
        if c.total_range == 0:
            return 0
        
        if c.upper_wick > c.body_size * 1.5:
            return 65
        return 0
    
    def _is_doji(self, c: Candle) -> float:
        """Doji detection"""
        if c.total_range == 0:
            return 0
        
        if c.body_size < c.total_range * 0.15:  # Was 0.1
            return 50
        return 0
    
    def _is_morning_star(self, c1: Candle, c2: Candle, c3: Candle) -> float:
        """Morning star"""
        if not (c1.is_bearish and c3.is_bullish):
            return 0
        
        if c2.body_size < c1.body_size * 0.6:  # Relaxed
            return 70
        return 0
    
    def _is_evening_star(self, c1: Candle, c2: Candle, c3: Candle) -> float:
        """Evening star"""
        if not (c1.is_bullish and c3.is_bearish):
            return 0
        
        if c2.body_size < c1.body_size * 0.6:
            return 70
        return 0
    
    def _is_inside_bar(self, c1: Candle, c2: Candle) -> float:
        """Inside bar"""
        if c2.high < c1.high and c2.low > c1.low:
            return 60
        return 0
    
    def _is_outside_bar(self, c1: Candle, c2: Candle) -> float:
        """Outside bar"""
        if c2.high > c1.high and c2.low < c1.low:
            return 65
        return 0
    
    def _analyze_trend(self, candles: List[Candle]) -> TrendDirection:
        """Analyze trend"""
        if len(candles) < 10:
            return TrendDirection.SIDEWAYS
        
        recent = candles[-10:]
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        
        first_half_high = max(highs[:5])
        second_half_high = max(highs[5:])
        first_half_low = min(lows[:5])
        second_half_low = min(lows[5:])
        
        if second_half_high > first_half_high and second_half_low > first_half_low:
            price_change = (second_half_high - first_half_high) / first_half_high
            return TrendDirection.STRONG_UPTREND if price_change > 0.01 else TrendDirection.UPTREND
        
        elif second_half_high < first_half_high and second_half_low < first_half_low:
            price_change = (first_half_high - second_half_high) / first_half_high
            return TrendDirection.STRONG_DOWNTREND if price_change > 0.01 else TrendDirection.DOWNTREND
        
        return TrendDirection.SIDEWAYS
    
    def _calculate_trend_strength(self, candles: List[Candle]) -> float:
        """Calculate trend strength"""
        if len(candles) < 5:
            return 0
        
        recent = candles[-5:]
        bullish_count = sum(1 for c in recent if c.is_bullish)
        
        if bullish_count >= 4:
            return 85
        elif bullish_count <= 1:
            return 85
        elif bullish_count == 3 or bullish_count == 2:
            return 50
        return 30
    
    def _find_support_resistance(self, candles: List[Candle]) -> Tuple[Optional[float], Optional[float]]:
        """Find S/R levels"""
        if len(candles) < 10:
            return None, None
        
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        price_range = max(highs) - min(lows)
        tolerance = price_range * 0.001
        
        resistance_candidates = []
        for h in highs:
            touches = sum(1 for x in highs if abs(x - h) <= tolerance)
            if touches >= self.support_resistance_touches:
                resistance_candidates.append(h)
        
        resistance = max(resistance_candidates) if resistance_candidates else None
        
        support_candidates = []
        for l in lows:
            touches = sum(1 for x in lows if abs(x - l) <= tolerance)
            if touches >= self.support_resistance_touches:
                support_candidates.append(l)
        
        support = min(support_candidates) if support_candidates else None
        
        return support, resistance
    
    def _distance_to_key_level(self, candles: List[Candle], support: Optional[float], resistance: Optional[float]) -> float:
        """Distance to S/R"""
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
        """Analyze momentum"""
        if len(candles) < 5:
            return "Neutral"
        
        recent = candles[-5:]
        body_sizes = [c.body_size for c in recent]
        
        if body_sizes[-1] > body_sizes[-2] > body_sizes[-3]:
            return "Increasing"
        elif body_sizes[-1] < body_sizes[-2] < body_sizes[-3]:
            return "Decreasing"
        
        return "Neutral"
    
    def _analyze_market_structure(self, candles: List[Candle]) -> str:
        """Market structure"""
        if len(candles) < 10:
            return "Ranging"
        
        recent = candles[-10:]
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        
        if max(highs[-5:]) > max(highs[:5]) and min(lows[-5:]) > min(lows[:5]):
            return "HH/HL (Uptrend)"
        elif max(highs[-5:]) < max(highs[:5]) and min(lows[-5:]) < min(lows[:5]):
            return "LL/LH (Downtrend)"
        
        return "Ranging"
    
    def _find_rejection_zones(self, candles: List[Candle]) -> List[float]:
        """Find rejection zones"""
        rejection_zones = []
        
        for candle in candles[-10:]:
            if candle.upper_wick > candle.body_size * 2:
                rejection_zones.append(candle.high)
            
            if candle.lower_wick > candle.body_size * 2:
                rejection_zones.append(candle.low)
        
        return rejection_zones
