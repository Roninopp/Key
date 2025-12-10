"""
MODULE 3: SIGNAL GENERATOR & DECISION ENGINE
Combines price action + timing to generate UP/DOWN signals with confidence
UPDATED: Lower threshold (40%+) for more trading opportunities
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
from datetime import datetime

# Import from previous modules
# from module1 import PriceActionAnalysis, CandleType, TrendDirection
# from module2 import TimingInfo, SignalValidity

class SignalDirection(Enum):
    UP = "⬆️ UP"
    DOWN = "⬇️ DOWN"
    NO_TRADE = "⏸️ NO TRADE"

class ConfidenceLevel(Enum):
    VERY_HIGH = "VERY HIGH (80-100%)"
    HIGH = "HIGH (70-79%)"
    MODERATE = "MODERATE (60-69%)"
    MEDIUM = "MEDIUM (50-59%)"
    LOW = "LOW (40-49%)"
    VERY_LOW = "TOO LOW (<40%)"

@dataclass
class TradingSignal:
    """Complete trading signal with all information"""
    direction: SignalDirection
    confidence: float  # 0-100
    confidence_level: ConfidenceLevel
    entry_time: datetime
    expiry_time: datetime
    timeframe: str
    reasoning: List[str]  # Why this signal
    warnings: List[str]   # Risk warnings
    pattern_strength: float
    trend_alignment: bool
    timing_valid: bool
    should_trade: bool
    formatted_message: str

class SignalGenerator:
    """
    Main signal generation engine
    UPDATED: Lower minimum confidence to 40% for more signals
    """
    
    def __init__(self):
        # UPDATED: Lower thresholds
        self.min_tradeable_confidence = 40  # Was 60 - NOW 40%
        self.high_confidence_threshold = 70  # Was 75
        self.very_high_confidence_threshold = 80  # Was 85
        
        # Pattern weights (UPDATED: More balanced)
        self.pattern_weights = {
            'BULLISH_ENGULFING': 28,
            'BEARISH_ENGULFING': 28,
            'BULLISH_PIN_BAR': 24,
            'BEARISH_PIN_BAR': 24,
            'HAMMER': 20,
            'SHOOTING_STAR': 20,
            'MORNING_STAR': 26,
            'EVENING_STAR': 26,
            'INSIDE_BAR': 12,
            'OUTSIDE_BAR': 15,
            'DOJI': 8,
            'BULLISH_MOMENTUM': 18,  # NEW
            'BEARISH_MOMENTUM': 18,  # NEW
            'BULLISH_TREND': 16,     # NEW
            'BEARISH_TREND': 16,     # NEW
        }
        
        # Bonuses
        self.trend_alignment_bonus = 18  # Was 20
        self.trend_strength_multiplier = 0.25  # Was 0.3
        self.key_level_proximity_bonus = 12  # Was 15
        
    def generate_signal(self, price_analysis, timing_info) -> TradingSignal:
        """
        Generate complete trading signal
        UPDATED: More lenient signal generation
        """
        
        print(f"\n{'='*50}")
        print(f"🎯 GENERATING SIGNAL")
        print(f"{'='*50}")
        
        # Step 1: Check timing
        if not self._is_timing_valid(timing_info):
            print("❌ Timing invalid")
            return self._create_no_trade_signal(
                "Signal expired or invalid timing",
                timing_info
            )
        
        print(f"✅ Timing valid: {timing_info.signal_validity}")
        
        # Step 2: Analyze patterns
        direction, base_confidence, pattern_reasons = self._analyze_patterns(
            price_analysis.detected_patterns
        )
        
        print(f"📊 Base analysis: {direction.value} @ {base_confidence:.0f}%")
        print(f"   Patterns used: {len(pattern_reasons)}")
        
        # UPDATED: If no clear pattern, use trend as fallback
        if direction == SignalDirection.NO_TRADE:
            print("⚠️ No patterns detected, using trend fallback...")
            direction, base_confidence, pattern_reasons = self._fallback_trend_signal(
                price_analysis
            )
        
        if direction == SignalDirection.NO_TRADE:
            print("❌ Still no clear signal after fallback")
            return self._create_no_trade_signal(
                "No clear trading opportunity detected",
                timing_info
            )
        
        # Step 3: Apply bonuses
        trend_bonus, trend_aligned = self._calculate_trend_bonus(
            direction,
            price_analysis.trend,
            price_analysis.trend_strength
        )
        print(f"📈 Trend bonus: +{trend_bonus:.0f}% (Aligned: {trend_aligned})")
        
        sr_bonus = self._calculate_sr_bonus(
            direction,
            price_analysis.key_level_distance,
            price_analysis.support_level,
            price_analysis.resistance_level,
            price_analysis.candles[-1].close if price_analysis.candles else 0
        )
        print(f"🎯 S/R bonus: +{sr_bonus:.0f}%")
        
        momentum_bonus = self._calculate_momentum_bonus(
            direction,
            price_analysis.momentum
        )
        print(f"⚡ Momentum bonus: +{momentum_bonus:.0f}%")
        
        structure_bonus = self._calculate_structure_bonus(
            direction,
            price_analysis.market_structure
        )
        print(f"🏗️ Structure bonus: +{structure_bonus:.0f}%")
        
        # Step 4: Calculate final confidence
        final_confidence = self._calculate_final_confidence(
            base_confidence,
            trend_bonus,
            sr_bonus,
            momentum_bonus,
            structure_bonus
        )
        
        print(f"\n💪 FINAL CONFIDENCE: {final_confidence:.0f}%")
        print(f"   Threshold: {self.min_tradeable_confidence}%")
        
        # Step 5: Determine if tradeable
        should_trade = final_confidence >= self.min_tradeable_confidence
        
        print(f"✅ Should trade: {should_trade}")
        
        confidence_level = self._get_confidence_level(final_confidence)
        
        # Step 6: Build reasoning & warnings
        reasoning = self._build_reasoning(
            pattern_reasons,
            trend_aligned,
            sr_bonus > 0,
            momentum_bonus > 0,
            structure_bonus > 0,
            price_analysis
        )
        
        warnings = self._build_warnings(
            final_confidence,
            trend_aligned,
            timing_info,
            price_analysis
        )
        
        pattern_strength = sum(strength for _, strength in price_analysis.detected_patterns) / len(price_analysis.detected_patterns) if price_analysis.detected_patterns else 0
        
        print(f"{'='*50}\n")
        
        # Create signal
        signal = TradingSignal(
            direction=direction,
            confidence=final_confidence,
            confidence_level=confidence_level,
            entry_time=timing_info.next_candle_open,
            expiry_time=timing_info.next_candle_open,
            timeframe=timing_info.timeframe,
            reasoning=reasoning,
            warnings=warnings,
            pattern_strength=pattern_strength,
            trend_alignment=trend_aligned,
            timing_valid=True,
            should_trade=should_trade,
            formatted_message=""
        )
        
        signal.formatted_message = self._format_signal_message(signal, timing_info, price_analysis)
        
        return signal
    
    def _is_timing_valid(self, timing_info) -> bool:
        """Check timing validity"""
        from module2 import SignalValidity
        return timing_info.signal_validity in [
            SignalValidity.VALID,
            SignalValidity.EXPIRING_SOON
        ]
    
    def _analyze_patterns(self, detected_patterns: List[Tuple]) -> Tuple[SignalDirection, float, List[str]]:
        """
        Analyze patterns - UPDATED for new pattern types
        """
        if not detected_patterns:
            return SignalDirection.NO_TRADE, 0, []
        
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        for pattern_type, strength in detected_patterns:
            pattern_name = pattern_type.name if hasattr(pattern_type, 'name') else str(pattern_type)
            weight = self.pattern_weights.get(pattern_name, 10)
            
            weighted_strength = (strength / 100) * weight
            
            # Bullish patterns
            if any(x in pattern_name for x in ['BULLISH', 'HAMMER', 'MORNING']):
                bullish_score += weighted_strength
                reasons.append(f"✓ {pattern_name.replace('_', ' ').title()} ({strength:.0f}%)")
            
            # Bearish patterns
            elif any(x in pattern_name for x in ['BEARISH', 'SHOOTING', 'EVENING']):
                bearish_score += weighted_strength
                reasons.append(f"✓ {pattern_name.replace('_', ' ').title()} ({strength:.0f}%)")
            
            # Neutral
            else:
                reasons.append(f"⚠️ {pattern_name.replace('_', ' ').title()} (Neutral)")
        
        # UPDATED: Lower threshold from 15 to 8
        if bullish_score > bearish_score and bullish_score > 8:
            direction = SignalDirection.UP
            confidence = min(bullish_score, 55)  # Cap at 55 for base
        elif bearish_score > bullish_score and bearish_score > 8:
            direction = SignalDirection.DOWN
            confidence = min(bearish_score, 55)
        else:
            direction = SignalDirection.NO_TRADE
            confidence = 0
        
        return direction, confidence, reasons
    
    def _fallback_trend_signal(self, price_analysis) -> Tuple[SignalDirection, float, List[str]]:
        """
        NEW: Fallback to trend-based signal when no patterns detected
        """
        from module1 import TrendDirection
        
        trend = price_analysis.trend
        trend_strength = price_analysis.trend_strength
        
        reasons = []
        
        # Strong trends
        if trend in [TrendDirection.STRONG_UPTREND, TrendDirection.UPTREND]:
            if trend_strength > 60:
                reasons.append(f"✓ Strong uptrend detected ({trend_strength:.0f}%)")
                return SignalDirection.UP, 35, reasons  # Base 35%
            elif trend_strength > 40:
                reasons.append(f"✓ Uptrend present ({trend_strength:.0f}%)")
                return SignalDirection.UP, 25, reasons
        
        elif trend in [TrendDirection.STRONG_DOWNTREND, TrendDirection.DOWNTREND]:
            if trend_strength > 60:
                reasons.append(f"✓ Strong downtrend detected ({trend_strength:.0f}%)")
                return SignalDirection.DOWN, 35, reasons
            elif trend_strength > 40:
                reasons.append(f"✓ Downtrend present ({trend_strength:.0f}%)")
                return SignalDirection.DOWN, 25, reasons
        
        # Last resort: check last 3 candles direction
        if len(price_analysis.candles) >= 3:
            recent = price_analysis.candles[-3:]
            bullish_count = sum(1 for c in recent if c.is_bullish)
            
            if bullish_count >= 2:
                reasons.append(f"✓ Recent bullish momentum ({bullish_count}/3 green candles)")
                return SignalDirection.UP, 30, reasons
            elif bullish_count <= 1:
                reasons.append(f"✓ Recent bearish momentum ({3-bullish_count}/3 red candles)")
                return SignalDirection.DOWN, 30, reasons
        
        return SignalDirection.NO_TRADE, 0, []
    
    def _calculate_trend_bonus(self, direction: SignalDirection, trend, trend_strength: float) -> Tuple[float, bool]:
        """Calculate trend bonus"""
        from module1 import TrendDirection
        
        bonus = 0
        aligned = False
        
        if direction == SignalDirection.UP:
            if trend in [TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND]:
                aligned = True
                bonus = self.trend_alignment_bonus + (trend_strength * self.trend_strength_multiplier)
        
        elif direction == SignalDirection.DOWN:
            if trend in [TrendDirection.DOWNTREND, TrendDirection.STRONG_DOWNTREND]:
                aligned = True
                bonus = self.trend_alignment_bonus + (trend_strength * self.trend_strength_multiplier)
        
        return bonus, aligned
    
    def _calculate_sr_bonus(self, direction: SignalDirection, distance: float, 
                           support: Optional[float], resistance: Optional[float], 
                           current_price: float) -> float:
        """Calculate S/R bonus"""
        
        if distance > 0.5:
            return 0
        
        bonus = 0
        
        if direction == SignalDirection.UP and support:
            if abs(current_price - support) / current_price * 100 < 0.3:  # Relaxed from 0.2
                bonus = self.key_level_proximity_bonus
        
        elif direction == SignalDirection.DOWN and resistance:
            if abs(current_price - resistance) / current_price * 100 < 0.3:
                bonus = self.key_level_proximity_bonus
        
        return bonus
    
    def _calculate_momentum_bonus(self, direction: SignalDirection, momentum: str) -> float:
        """Calculate momentum bonus"""
        bonus = 0
        
        if direction == SignalDirection.UP and momentum == "Increasing":
            bonus = 10
        elif direction == SignalDirection.DOWN and momentum == "Decreasing":
            bonus = 10
        
        return bonus
    
    def _calculate_structure_bonus(self, direction: SignalDirection, structure: str) -> float:
        """Calculate structure bonus"""
        bonus = 0
        
        if direction == SignalDirection.UP and "Uptrend" in structure:
            bonus = 8
        elif direction == SignalDirection.DOWN and "Downtrend" in structure:
            bonus = 8
        
        return bonus
    
    def _calculate_final_confidence(self, base: float, trend: float, sr: float, 
                                   momentum: float, structure: float) -> float:
        """Calculate final confidence"""
        total = base + trend + sr + momentum + structure
        return min(total, 100)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level - UPDATED thresholds"""
        if confidence >= 80:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 70:
            return ConfidenceLevel.HIGH
        elif confidence >= 60:
            return ConfidenceLevel.MODERATE
        elif confidence >= 50:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _build_reasoning(self, pattern_reasons: List[str], trend_aligned: bool,
                        near_sr: bool, momentum_good: bool, structure_good: bool,
                        price_analysis) -> List[str]:
        """Build reasoning"""
        reasoning = pattern_reasons.copy()
        
        if trend_aligned:
            reasoning.append(f"✓ Trend aligned: {price_analysis.trend.value}")
        else:
            reasoning.append(f"⚠️ Against trend: {price_analysis.trend.value}")
        
        if near_sr:
            reasoning.append("✓ Near key support/resistance")
        
        if momentum_good:
            reasoning.append(f"✓ Momentum: {price_analysis.momentum}")
        
        if structure_good:
            reasoning.append(f"✓ Structure: {price_analysis.market_structure}")
        
        return reasoning
    
    def _build_warnings(self, confidence: float, trend_aligned: bool,
                       timing_info, price_analysis) -> List[str]:
        """Build warnings - UPDATED for lower confidence"""
        warnings = []
        
        if confidence < 50:
            warnings.append("⚠️ Low confidence - use small position size")
        elif confidence < 60:
            warnings.append("⚠️ Moderate confidence - trade with caution")
        
        if not trend_aligned:
            warnings.append("⚠️ Trading against trend - higher risk")
        
        if timing_info.time_lag > 10:
            warnings.append(f"⚠️ Signal lag: {timing_info.time_lag:.0f}s")
        
        from module2 import SignalValidity
        if timing_info.signal_validity == SignalValidity.EXPIRING_SOON:
            warnings.append("⚡ Signal expiring - enter quickly!")
        
        if price_analysis.trend.name == "SIDEWAYS":
            warnings.append("⚠️ Sideways market - less predictable")
        
        if not warnings:
            warnings.append("✅ Good trading setup")
        
        return warnings
    
    def _create_no_trade_signal(self, reason: str, timing_info) -> TradingSignal:
        """Create NO TRADE signal"""
        signal = TradingSignal(
            direction=SignalDirection.NO_TRADE,
            confidence=0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            entry_time=timing_info.next_candle_open if timing_info else datetime.now(),
            expiry_time=timing_info.next_candle_open if timing_info else datetime.now(),
            timeframe=timing_info.timeframe if timing_info else "1m",
            reasoning=[f"❌ {reason}"],
            warnings=["⏸️ Wait for better setup"],
            pattern_strength=0,
            trend_alignment=False,
            timing_valid=False,
            should_trade=False,
            formatted_message=""
        )
        
        signal.formatted_message = f"⏸️ **NO TRADE**\n\n❌ {reason}\n\n⏳ Wait for next setup"
        return signal
    
    def _format_signal_message(self, signal: TradingSignal, timing_info, price_analysis) -> str:
        """Format signal message - UPDATED for new confidence levels"""
        
        if not signal.should_trade:
            return signal.formatted_message if signal.formatted_message else "⏸️ NO TRADE"
        
        msg = "🎯 **TRADING SIGNAL**\n"
        msg += "=" * 40 + "\n\n"
        
        msg += f"**Direction:** {signal.direction.value}\n"
        msg += f"**Confidence:** {signal.confidence:.0f}% ({signal.confidence_level.value})\n"
        msg += f"**Timeframe:** {signal.timeframe}\n"
        msg += f"**Candles Analyzed:** {price_analysis.total_candles_analyzed}\n\n"
        
        msg += "⏰ **TIMING:**\n"
        msg += f"   Entry: {signal.entry_time.strftime('%H:%M:%S')}\n"
        msg += f"   Countdown: {timing_info.seconds_until_entry:.0f}s\n"
        msg += f"   {timing_info.entry_message}\n\n"
        
        msg += "📊 **ANALYSIS:**\n"
        for reason in signal.reasoning[:5]:
            msg += f"   {reason}\n"
        
        if signal.warnings:
            msg += "\n⚠️ **WARNINGS:**\n"
            for warning in signal.warnings:
                msg += f"   {warning}\n"
        
        msg += "\n" + "=" * 40 + "\n"
        
        # UPDATED recommendations for lower confidence
        if signal.confidence >= 80:
            msg += "✅ **VERY STRONG SIGNAL**\n"
            msg += "💰 High confidence trade\n"
        elif signal.confidence >= 70:
            msg += "✅ **STRONG SIGNAL**\n"
            msg += "👍 Recommended trade\n"
        elif signal.confidence >= 60:
            msg += "✅ **GOOD SIGNAL**\n"
            msg += "👌 Solid setup\n"
        elif signal.confidence >= 50:
            msg += "⚠️ **MODERATE SIGNAL**\n"
            msg += "🤔 Trade with caution\n"
        elif signal.confidence >= 40:
            msg += "⚠️ **LOW CONFIDENCE**\n"
            msg += "🎲 High risk - small position\n"
        else:
            msg += "❌ **TOO WEAK**\n"
            msg += "⏸️ Skip this trade\n"
        
        msg += "\n💡 *Trade at your own risk. Not financial advice.*"
        
        return msg


# INTEGRATION CLASS
class PocketOptionTradingBot:
    """Complete bot integration"""
    
    def __init__(self):
        from module1 import PocketOptionChartAnalyzer
        from module2 import TimingSystemIntegrator
        
        self.price_analyzer = PocketOptionChartAnalyzer()
        self.timing_system = TimingSystemIntegrator()
        self.signal_generator = SignalGenerator()
    
    def analyze_and_generate_signal(self, image_path: str) -> TradingSignal:
        """Complete analysis pipeline"""
        
        # Analyze price action
        price_analysis = self.price_analyzer.analyze_chart(image_path)
        
        # Analyze timing
        candle_completion = 90.0
        if price_analysis.candles:
            last_candle = price_analysis.candles[-1]
            if last_candle.total_range > 0:
                candle_completion = (last_candle.body_size / last_candle.total_range) * 100
        
        timing_info = self.timing_system.analyze_chart_timing(image_path, candle_completion)
        
        # Generate signal
        signal = self.signal_generator.generate_signal(price_analysis, timing_info)
        
        return signal
