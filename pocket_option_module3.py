"""
MODULE 3: SIGNAL GENERATOR & DECISION ENGINE
Combines price action + timing to generate UP/DOWN signals with confidence
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
    VERY_HIGH = "VERY HIGH (85-100%)"
    HIGH = "HIGH (75-84%)"
    MODERATE = "MODERATE (65-74%)"
    LOW = "LOW (55-64%)"
    VERY_LOW = "TOO LOW (<55%)"

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
    Combines price action analysis + timing to generate signals
    """
    
    def __init__(self):
        # Confidence thresholds
        self.min_tradeable_confidence = 60  # Don't trade below this
        self.high_confidence_threshold = 75
        self.very_high_confidence_threshold = 85
        
        # Pattern weights (how much each pattern contributes to confidence)
        self.pattern_weights = {
            'BULLISH_ENGULFING': 30,
            'BEARISH_ENGULFING': 30,
            'BULLISH_PIN_BAR': 25,
            'BEARISH_PIN_BAR': 25,
            'HAMMER': 20,
            'SHOOTING_STAR': 20,
            'MORNING_STAR': 28,
            'EVENING_STAR': 28,
            'INSIDE_BAR': 10,
            'OUTSIDE_BAR': 15,
            'DOJI': 5,  # Weak signal alone
        }
        
        # Trend alignment bonus
        self.trend_alignment_bonus = 20
        self.trend_strength_multiplier = 0.3
        
        # Support/Resistance bonus
        self.key_level_proximity_bonus = 15  # If near S/R
        
    def generate_signal(self, price_analysis, timing_info) -> TradingSignal:
        """
        Generate complete trading signal
        
        Args:
            price_analysis: PriceActionAnalysis from Module 1
            timing_info: TimingInfo from Module 2
        
        Returns:
            TradingSignal with direction, confidence, and all details
        """
        
        # Step 1: Check if timing is valid
        if not self._is_timing_valid(timing_info):
            return self._create_no_trade_signal(
                "Signal expired or invalid timing",
                timing_info
            )
        
        # Step 2: Analyze patterns and calculate base confidence
        direction, base_confidence, pattern_reasons = self._analyze_patterns(
            price_analysis.detected_patterns
        )
        
        if direction == SignalDirection.NO_TRADE:
            return self._create_no_trade_signal(
                "No clear pattern detected",
                timing_info
            )
        
        # Step 3: Apply trend alignment adjustment
        trend_bonus, trend_aligned = self._calculate_trend_bonus(
            direction,
            price_analysis.trend,
            price_analysis.trend_strength
        )
        
        # Step 4: Apply support/resistance bonus
        sr_bonus = self._calculate_sr_bonus(
            direction,
            price_analysis.key_level_distance,
            price_analysis.support_level,
            price_analysis.resistance_level,
            price_analysis.candles[-1].close if price_analysis.candles else 0
        )
        
        # Step 5: Apply momentum bonus
        momentum_bonus = self._calculate_momentum_bonus(
            direction,
            price_analysis.momentum
        )
        
        # Step 6: Apply market structure bonus
        structure_bonus = self._calculate_structure_bonus(
            direction,
            price_analysis.market_structure
        )
        
        # Step 7: Calculate final confidence
        final_confidence = self._calculate_final_confidence(
            base_confidence,
            trend_bonus,
            sr_bonus,
            momentum_bonus,
            structure_bonus
        )
        
        # Step 8: Determine if should trade
        should_trade = final_confidence >= self.min_tradeable_confidence
        
        # Step 9: Get confidence level
        confidence_level = self._get_confidence_level(final_confidence)
        
        # Step 10: Build reasoning
        reasoning = self._build_reasoning(
            pattern_reasons,
            trend_aligned,
            sr_bonus > 0,
            momentum_bonus > 0,
            structure_bonus > 0,
            price_analysis
        )
        
        # Step 11: Build warnings
        warnings = self._build_warnings(
            final_confidence,
            trend_aligned,
            timing_info,
            price_analysis
        )
        
        # Step 12: Calculate pattern strength
        pattern_strength = sum(strength for _, strength in price_analysis.detected_patterns) / len(price_analysis.detected_patterns) if price_analysis.detected_patterns else 0
        
        # Step 13: Create signal
        signal = TradingSignal(
            direction=direction,
            confidence=final_confidence,
            confidence_level=confidence_level,
            entry_time=timing_info.next_candle_open,
            expiry_time=timing_info.next_candle_open,  # Will be calculated based on timeframe
            timeframe=timing_info.timeframe,
            reasoning=reasoning,
            warnings=warnings,
            pattern_strength=pattern_strength,
            trend_alignment=trend_aligned,
            timing_valid=True,
            should_trade=should_trade,
            formatted_message=""
        )
        
        # Step 14: Format message
        signal.formatted_message = self._format_signal_message(signal, timing_info, price_analysis)
        
        return signal
    
    def _is_timing_valid(self, timing_info) -> bool:
        """Check if timing is valid for trading"""
        from module2 import SignalValidity
        return timing_info.signal_validity in [
            SignalValidity.VALID,
            SignalValidity.EXPIRING_SOON
        ]
    
    def _analyze_patterns(self, detected_patterns: List[Tuple]) -> Tuple[SignalDirection, float, List[str]]:
        """
        Analyze detected patterns and determine direction + base confidence
        
        Returns: (direction, confidence, reasons)
        """
        if not detected_patterns:
            return SignalDirection.NO_TRADE, 0, []
        
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        for pattern_type, strength in detected_patterns:
            pattern_name = pattern_type.name if hasattr(pattern_type, 'name') else str(pattern_type)
            weight = self.pattern_weights.get(pattern_name, 10)
            
            # Calculate weighted score
            weighted_strength = (strength / 100) * weight
            
            # Determine if bullish or bearish
            if 'BULLISH' in pattern_name or pattern_name in ['HAMMER', 'MORNING_STAR']:
                bullish_score += weighted_strength
                reasons.append(f"✓ {pattern_name.replace('_', ' ').title()} ({strength:.0f}% strength)")
            elif 'BEARISH' in pattern_name or pattern_name in ['SHOOTING_STAR', 'EVENING_STAR']:
                bearish_score += weighted_strength
                reasons.append(f"✓ {pattern_name.replace('_', ' ').title()} ({strength:.0f}% strength)")
            else:
                # Neutral patterns (Doji, Inside Bar)
                reasons.append(f"⚠️ {pattern_name.replace('_', ' ').title()} (Neutral)")
        
        # Determine direction
        if bullish_score > bearish_score and bullish_score > 15:
            direction = SignalDirection.UP
            confidence = min(bullish_score, 60)  # Base confidence capped at 60
        elif bearish_score > bullish_score and bearish_score > 15:
            direction = SignalDirection.DOWN
            confidence = min(bearish_score, 60)
        else:
            direction = SignalDirection.NO_TRADE
            confidence = 0
        
        return direction, confidence, reasons
    
    def _calculate_trend_bonus(self, direction: SignalDirection, trend, trend_strength: float) -> Tuple[float, bool]:
        """Calculate bonus for trend alignment"""
        from module1 import TrendDirection
        
        bonus = 0
        aligned = False
        
        # UP signal
        if direction == SignalDirection.UP:
            if trend in [TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND]:
                aligned = True
                bonus = self.trend_alignment_bonus + (trend_strength * self.trend_strength_multiplier)
        
        # DOWN signal
        elif direction == SignalDirection.DOWN:
            if trend in [TrendDirection.DOWNTREND, TrendDirection.STRONG_DOWNTREND]:
                aligned = True
                bonus = self.trend_alignment_bonus + (trend_strength * self.trend_strength_multiplier)
        
        return bonus, aligned
    
    def _calculate_sr_bonus(self, direction: SignalDirection, distance: float, 
                           support: Optional[float], resistance: Optional[float], 
                           current_price: float) -> float:
        """Calculate bonus for being near support/resistance"""
        
        if distance > 0.5:  # More than 0.5% away from key level
            return 0
        
        bonus = 0
        
        # UP signal near support
        if direction == SignalDirection.UP and support:
            if abs(current_price - support) / current_price * 100 < 0.2:
                bonus = self.key_level_proximity_bonus
        
        # DOWN signal near resistance
        elif direction == SignalDirection.DOWN and resistance:
            if abs(current_price - resistance) / current_price * 100 < 0.2:
                bonus = self.key_level_proximity_bonus
        
        return bonus
    
    def _calculate_momentum_bonus(self, direction: SignalDirection, momentum: str) -> float:
        """Calculate bonus for momentum alignment"""
        bonus = 0
        
        if direction == SignalDirection.UP and momentum == "Increasing":
            bonus = 10
        elif direction == SignalDirection.DOWN and momentum == "Decreasing":
            bonus = 10
        
        return bonus
    
    def _calculate_structure_bonus(self, direction: SignalDirection, structure: str) -> float:
        """Calculate bonus for market structure alignment"""
        bonus = 0
        
        if direction == SignalDirection.UP and "Uptrend" in structure:
            bonus = 8
        elif direction == SignalDirection.DOWN and "Downtrend" in structure:
            bonus = 8
        
        return bonus
    
    def _calculate_final_confidence(self, base: float, trend: float, sr: float, 
                                   momentum: float, structure: float) -> float:
        """Calculate final confidence score"""
        total = base + trend + sr + momentum + structure
        return min(total, 100)  # Cap at 100%
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level enum"""
        if confidence >= self.very_high_confidence_threshold:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= self.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= 65:
            return ConfidenceLevel.MODERATE
        elif confidence >= 55:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _build_reasoning(self, pattern_reasons: List[str], trend_aligned: bool,
                        near_sr: bool, momentum_good: bool, structure_good: bool,
                        price_analysis) -> List[str]:
        """Build complete reasoning list"""
        reasoning = pattern_reasons.copy()
        
        if trend_aligned:
            reasoning.append(f"✓ Trend aligned: {price_analysis.trend.value}")
        else:
            reasoning.append(f"⚠️ Against trend: {price_analysis.trend.value}")
        
        if near_sr:
            reasoning.append("✓ Near key support/resistance level")
        
        if momentum_good:
            reasoning.append(f"✓ Momentum: {price_analysis.momentum}")
        
        if structure_good:
            reasoning.append(f"✓ Market structure: {price_analysis.market_structure}")
        
        return reasoning
    
    def _build_warnings(self, confidence: float, trend_aligned: bool,
                       timing_info, price_analysis) -> List[str]:
        """Build warning list"""
        warnings = []
        
        if confidence < 70:
            warnings.append("⚠️ Moderate confidence - trade with caution")
        
        if not trend_aligned:
            warnings.append("⚠️ Trading against the trend - higher risk")
        
        if timing_info.time_lag > 10:
            warnings.append(f"⚠️ Signal lag: {timing_info.time_lag:.0f}s")
        
        from module2 import SignalValidity
        if timing_info.signal_validity == SignalValidity.EXPIRING_SOON:
            warnings.append("⚡ Signal expiring soon - enter quickly!")
        
        if price_analysis.trend.name == "SIDEWAYS":
            warnings.append("⚠️ Sideways market - unpredictable")
        
        if not warnings:
            warnings.append("✅ Good trading conditions")
        
        return warnings
    
    def _create_no_trade_signal(self, reason: str, timing_info) -> TradingSignal:
        """Create a NO TRADE signal"""
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
        """Format signal into Telegram message"""
        
        if not signal.should_trade:
            return signal.formatted_message if signal.formatted_message else "⏸️ NO TRADE"
        
        msg = "🎯 **TRADING SIGNAL**\n"
        msg += "=" * 40 + "\n\n"
        
        # Direction and Confidence
        msg += f"**Direction:** {signal.direction.value}\n"
        msg += f"**Confidence:** {signal.confidence:.0f}% ({signal.confidence_level.value})\n"
        msg += f"**Timeframe:** {signal.timeframe}\n\n"
        
        # Timing
        msg += "⏰ **TIMING:**\n"
        msg += f"   Entry: {signal.entry_time.strftime('%H:%M:%S')}\n"
        msg += f"   Countdown: {timing_info.seconds_until_entry:.0f}s\n"
        msg += f"   {timing_info.entry_message}\n\n"
        
        # Reasoning
        msg += "📊 **ANALYSIS:**\n"
        for reason in signal.reasoning[:5]:  # Top 5 reasons
            msg += f"   {reason}\n"
        
        # Warnings
        if signal.warnings:
            msg += "\n⚠️ **WARNINGS:**\n"
            for warning in signal.warnings:
                msg += f"   {warning}\n"
        
        # Trade recommendation
        msg += "\n" + "=" * 40 + "\n"
        
        if signal.confidence >= 80:
            msg += "✅ **STRONG SIGNAL - RECOMMENDED TRADE**\n"
        elif signal.confidence >= 70:
            msg += "✅ **GOOD SIGNAL - TRADE WITH CONFIDENCE**\n"
        elif signal.confidence >= 60:
            msg += "⚠️ **MODERATE SIGNAL - TRADE WITH CAUTION**\n"
        else:
            msg += "⏸️ **WEAK SIGNAL - CONSIDER SKIPPING**\n"
        
        msg += "\n💡 *Trade at your own risk. This is not financial advice.*"
        
        return msg


# COMPLETE INTEGRATION
class PocketOptionTradingBot:
    """
    Main bot class that integrates all modules
    """
    
    def __init__(self):
        # Import all modules
        from module1 import PocketOptionChartAnalyzer
        from module2 import TimingSystemIntegrator
        
        self.price_analyzer = PocketOptionChartAnalyzer()
        self.timing_system = TimingSystemIntegrator()
        self.signal_generator = SignalGenerator()
    
    def analyze_and_generate_signal(self, image_path: str) -> TradingSignal:
        """
        Complete analysis pipeline: Chart -> Signal
        """
        
        # Step 1: Analyze price action
        price_analysis = self.price_analyzer.analyze_chart(image_path)
        
        # Step 2: Analyze timing
        candle_completion = 90.0  # Can calculate from last candle
        if price_analysis.candles:
            last_candle = price_analysis.candles[-1]
            if last_candle.total_range > 0:
                candle_completion = (last_candle.body_size / last_candle.total_range) * 100
        
        timing_info = self.timing_system.analyze_chart_timing(image_path, candle_completion)
        
        # Step 3: Generate signal
        signal = self.signal_generator.generate_signal(price_analysis, timing_info)
        
        return signal


# USAGE EXAMPLE
if __name__ == "__main__":
    
    # Initialize complete bot
    bot = PocketOptionTradingBot()
    
    # Analyze chart
    signal = bot.analyze_and_generate_signal("chart.png")
    
    # Print result
    print(signal.formatted_message)
    
    print("\n" + "="*50)
    print("DECISION:")
    print("="*50)
    
    if signal.should_trade:
        print(f"✅ TRADE: {signal.direction.value}")
        print(f"💪 Confidence: {signal.confidence:.0f}%")
        print(f"⏰ Entry: {signal.entry_time.strftime('%H:%M:%S')}")
    else:
        print("❌ DON'T TRADE")
        print(f"Reason: {signal.reasoning[0] if signal.reasoning else 'No clear signal'}")
