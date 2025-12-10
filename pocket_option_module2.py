"""
MODULE 2: TIMING SYSTEM & ENTRY CALCULATOR
Extracts chart timestamp, calculates entry timing, prevents expired signals
"""

import cv2
import numpy as np
import pytesseract
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

class SignalValidity(Enum):
    VALID = "VALID - Trade Now"
    EXPIRING_SOON = "Expiring - Hurry!"
    EXPIRED = "EXPIRED - Don't Trade"
    WAIT_NEXT_CANDLE = "Wait for Next Candle"

@dataclass
class TimingInfo:
    """Complete timing analysis for trade entry"""
    chart_timestamp: datetime
    current_time: datetime
    time_lag: float  # seconds
    timeframe: str  # "1m", "5m", "15m"
    candle_close_time: datetime
    next_candle_open: datetime
    seconds_until_entry: float
    signal_validity: SignalValidity
    candle_completion: float  # 0-100%
    entry_message: str

class ChartTimestampExtractor:
    """
    Extracts timestamp and timeframe from Pocket Option chart screenshots
    """
    
    def __init__(self):
        # Common Pocket Option timestamp regions (top-right or top-left)
        self.timestamp_regions = [
            (0.7, 0.0, 1.0, 0.1),   # Top-right
            (0.0, 0.0, 0.3, 0.1),   # Top-left
            (0.4, 0.0, 0.6, 0.1),   # Top-center
        ]
        
        # Timeframe detection regions (usually top-left)
        self.timeframe_regions = [
            (0.0, 0.0, 0.2, 0.15),
        ]
        
    def extract_timestamp(self, image_path: str) -> Optional[datetime]:
        """
        Extract timestamp from Pocket Option chart
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Try each region
        for region in self.timestamp_regions:
            timestamp = self._extract_from_region(img, region)
            if timestamp:
                return timestamp
        
        return None
    
    def extract_timeframe(self, image_path: str) -> str:
        """
        Extract timeframe (1m, 5m, 15m, etc.) from chart
        """
        img = cv2.imread(image_path)
        if img is None:
            return "1m"  # Default
        
        for region in self.timeframe_regions:
            timeframe = self._extract_timeframe_from_region(img, region)
            if timeframe:
                return timeframe
        
        return "1m"  # Default to 1-minute
    
    def _extract_from_region(self, img: np.ndarray, region: Tuple[float, float, float, float]) -> Optional[datetime]:
        """Extract timestamp from specific image region"""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = region
        
        # Crop region
        crop = img[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
        
        # Preprocess for OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # OCR
        text = pytesseract.image_to_string(gray, config='--psm 7')
        
        # Parse timestamp
        timestamp = self._parse_timestamp(text)
        return timestamp
    
    def _extract_timeframe_from_region(self, img: np.ndarray, region: Tuple[float, float, float, float]) -> Optional[str]:
        """Extract timeframe from region"""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = region
        
        crop = img[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        text = pytesseract.image_to_string(gray, config='--psm 7').upper()
        
        # Look for timeframe patterns
        patterns = [
            r'(\d+)\s*M',   # "5M", "1M"
            r'(\d+)\s*MIN', # "5MIN"
            r'M(\d+)',      # "M5", "M1"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                minutes = match.group(1)
                return f"{minutes}m"
        
        return None
    
    def _parse_timestamp(self, text: str) -> Optional[datetime]:
        """
        Parse various timestamp formats from Pocket Option
        Examples: "14:23:45", "2:23:45 PM", "14:23"
        """
        text = text.strip()
        
        # Try different formats
        formats = [
            r'(\d{1,2}):(\d{2}):(\d{2})',      # 14:23:45
            r'(\d{1,2}):(\d{2})',               # 14:23
            r'(\d{1,2}):(\d{2})\s*(AM|PM)',    # 2:23 PM
        ]
        
        for fmt in formats:
            match = re.search(fmt, text)
            if match:
                try:
                    if len(match.groups()) == 3 and match.group(3) in ['AM', 'PM']:
                        # 12-hour format
                        hour = int(match.group(1))
                        minute = int(match.group(2))
                        if match.group(3) == 'PM' and hour != 12:
                            hour += 12
                        if match.group(3) == 'AM' and hour == 12:
                            hour = 0
                        second = 0
                    else:
                        # 24-hour format
                        hour = int(match.group(1))
                        minute = int(match.group(2))
                        second = int(match.group(3)) if len(match.groups()) >= 3 else 0
                    
                    # Create datetime with today's date
                    now = datetime.now()
                    timestamp = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
                    return timestamp
                except:
                    continue
        
        return None


class EntryTimingCalculator:
    """
    Calculates optimal entry timing based on chart analysis
    """
    
    def __init__(self):
        self.max_valid_lag = 15  # Maximum seconds lag before signal expires
        self.warning_lag = 10     # Warn user if lag approaching
        
    def calculate_timing(self, chart_timestamp: datetime, timeframe: str, 
                        candle_completion: float = 90.0) -> TimingInfo:
        """
        Calculate complete timing information for trade entry
        
        Args:
            chart_timestamp: Time extracted from chart
            timeframe: "1m", "5m", "15m", etc.
            candle_completion: How much of current candle is complete (0-100%)
        """
        current_time = datetime.now()
        time_lag = (current_time - chart_timestamp).total_seconds()
        
        # Parse timeframe
        timeframe_minutes = self._parse_timeframe(timeframe)
        
        # Calculate when current candle closes
        candle_close_time = self._calculate_candle_close(chart_timestamp, timeframe_minutes)
        
        # Next candle opens 1 second after close
        next_candle_open = candle_close_time + timedelta(seconds=1)
        
        # Calculate seconds until entry
        seconds_until_entry = (next_candle_open - current_time).total_seconds()
        
        # Determine signal validity
        validity = self._determine_validity(time_lag, seconds_until_entry, candle_completion)
        
        # Generate entry message
        entry_message = self._generate_entry_message(validity, seconds_until_entry, time_lag)
        
        return TimingInfo(
            chart_timestamp=chart_timestamp,
            current_time=current_time,
            time_lag=time_lag,
            timeframe=timeframe,
            candle_close_time=candle_close_time,
            next_candle_open=next_candle_open,
            seconds_until_entry=seconds_until_entry,
            signal_validity=validity,
            candle_completion=candle_completion,
            entry_message=entry_message
        )
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        match = re.search(r'(\d+)', timeframe)
        if match:
            return int(match.group(1))
        return 1  # Default 1 minute
    
    def _calculate_candle_close(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """Calculate when current candle closes"""
        # Round down to nearest timeframe
        minutes_since_hour = timestamp.minute
        candles_since_hour = minutes_since_hour // timeframe_minutes
        
        # Calculate next candle close
        next_close_minute = (candles_since_hour + 1) * timeframe_minutes
        
        candle_close = timestamp.replace(second=0, microsecond=0)
        candle_close = candle_close.replace(minute=0) + timedelta(minutes=next_close_minute)
        
        return candle_close
    
    def _determine_validity(self, time_lag: float, seconds_until_entry: float, 
                           candle_completion: float) -> SignalValidity:
        """Determine if signal is still valid for trading"""
        
        # Signal expired if lag too high
        if time_lag > self.max_valid_lag:
            return SignalValidity.EXPIRED
        
        # Pattern not complete yet
        if candle_completion < 85:
            return SignalValidity.WAIT_NEXT_CANDLE
        
        # Warning if lag approaching limit
        if time_lag > self.warning_lag:
            return SignalValidity.EXPIRING_SOON
        
        # Check if entry time is reasonable (5-60 seconds away)
        if seconds_until_entry < 5:
            return SignalValidity.EXPIRED  # Too late
        
        if seconds_until_entry > 60:
            return SignalValidity.WAIT_NEXT_CANDLE
        
        return SignalValidity.VALID
    
    def _generate_entry_message(self, validity: SignalValidity, 
                                seconds_until_entry: float, 
                                time_lag: float) -> str:
        """Generate human-readable entry message"""
        
        if validity == SignalValidity.EXPIRED:
            return f"❌ SIGNAL EXPIRED ({time_lag:.0f}s lag) - Wait for next setup"
        
        if validity == SignalValidity.WAIT_NEXT_CANDLE:
            return f"⏳ Pattern forming - Entry in {seconds_until_entry:.0f}s"
        
        if validity == SignalValidity.EXPIRING_SOON:
            return f"⚡ HURRY! Entry in {seconds_until_entry:.0f}s ({time_lag:.0f}s lag)"
        
        if validity == SignalValidity.VALID:
            if seconds_until_entry <= 10:
                return f"🚨 ENTER NOW! ({seconds_until_entry:.0f}s remaining)"
            elif seconds_until_entry <= 30:
                return f"⏰ Get ready! Entry in {seconds_until_entry:.0f}s"
            else:
                return f"✅ Valid signal - Entry in {seconds_until_entry:.0f}s"
        
        return "⏸️ Analyzing..."


class TimingSystemIntegrator:
    """
    Main class that combines timestamp extraction and timing calculation
    """
    
    def __init__(self):
        self.timestamp_extractor = ChartTimestampExtractor()
        self.timing_calculator = EntryTimingCalculator()
    
    def analyze_chart_timing(self, image_path: str, candle_completion: float = 90.0) -> Optional[TimingInfo]:
        """
        Complete timing analysis pipeline
        
        Args:
            image_path: Path to chart screenshot
            candle_completion: Estimated candle completion % (can be calculated from price action)
        
        Returns:
            TimingInfo object with complete timing data
        """
        
        # Extract timestamp from chart
        chart_timestamp = self.timestamp_extractor.extract_timestamp(image_path)
        
        if chart_timestamp is None:
            # Fallback: assume screenshot was taken NOW
            chart_timestamp = datetime.now()
        
        # Extract timeframe
        timeframe = self.timestamp_extractor.extract_timeframe(image_path)
        
        # Calculate timing
        timing_info = self.timing_calculator.calculate_timing(
            chart_timestamp, 
            timeframe, 
            candle_completion
        )
        
        return timing_info
    
    def format_timing_message(self, timing_info: TimingInfo) -> str:
        """
        Format timing info into Telegram-ready message
        """
        
        msg = "⏱️ **TIMING ANALYSIS**\n\n"
        
        # Signal validity status
        if timing_info.signal_validity == SignalValidity.VALID:
            msg += "✅ **Status:** VALID SIGNAL\n"
        elif timing_info.signal_validity == SignalValidity.EXPIRING_SOON:
            msg += "⚡ **Status:** EXPIRING SOON\n"
        elif timing_info.signal_validity == SignalValidity.EXPIRED:
            msg += "❌ **Status:** EXPIRED\n"
        else:
            msg += "⏳ **Status:** WAIT FOR COMPLETION\n"
        
        msg += f"\n📊 **Timeframe:** {timing_info.timeframe}\n"
        msg += f"⏰ **Chart Time:** {timing_info.chart_timestamp.strftime('%H:%M:%S')}\n"
        msg += f"🕐 **Current Time:** {timing_info.current_time.strftime('%H:%M:%S')}\n"
        msg += f"⚠️ **Lag:** {timing_info.time_lag:.1f}s\n"
        
        msg += f"\n🎯 **ENTRY INFO:**\n"
        msg += f"   Candle closes: {timing_info.candle_close_time.strftime('%H:%M:%S')}\n"
        msg += f"   Entry time: {timing_info.next_candle_open.strftime('%H:%M:%S')}\n"
        msg += f"   Countdown: {timing_info.seconds_until_entry:.0f}s\n"
        
        msg += f"\n📈 **Candle Completion:** {timing_info.candle_completion:.0f}%\n"
        
        msg += f"\n{timing_info.entry_message}\n"
        
        return msg


# USAGE EXAMPLE
if __name__ == "__main__":
    
    # Initialize timing system
    timing_system = TimingSystemIntegrator()
    
    # Analyze chart
    timing_info = timing_system.analyze_chart_timing(
        image_path="chart.png",
        candle_completion=92.0  # Can be calculated from Module 1
    )
    
    if timing_info:
        # Print formatted message
        message = timing_system.format_timing_message(timing_info)
        print(message)
        
        print("\n" + "="*50)
        print("TRADING DECISION:")
        print("="*50)
        
        if timing_info.signal_validity == SignalValidity.VALID:
            print(f"✅ TRADE THIS SIGNAL")
            print(f"⏰ Enter at: {timing_info.next_candle_open.strftime('%H:%M:%S')}")
            print(f"⏱️ In {timing_info.seconds_until_entry:.0f} seconds")
            
        elif timing_info.signal_validity == SignalValidity.EXPIRING_SOON:
            print(f"⚡ QUICK! Enter NOW or skip")
            print(f"⚠️ Only {timing_info.seconds_until_entry:.0f}s remaining")
            
        elif timing_info.signal_validity == SignalValidity.EXPIRED:
            print(f"❌ DON'T TRADE - Signal expired")
            print(f"⏰ Wait for next setup")
            
        else:
            print(f"⏳ WAIT - Pattern not complete")
            print(f"🎯 Entry in {timing_info.seconds_until_entry:.0f}s")
    
    else:
        print("❌ Could not extract timing information")


"""
INTEGRATION WITH MODULE 1:

from module1 import PocketOptionChartAnalyzer
from module2 import TimingSystemIntegrator

# Analyze price action
price_analyzer = PocketOptionChartAnalyzer()
price_analysis = price_analyzer.analyze_chart("chart.png")

# Analyze timing
timing_system = TimingSystemIntegrator()
timing_info = timing_system.analyze_chart_timing(
    "chart.png",
    candle_completion=price_analysis.candles[-1].body_size / price_analysis.candles[-1].total_range * 100
)

# Combined decision
if timing_info.signal_validity == SignalValidity.VALID and len(price_analysis.detected_patterns) > 0:
    print("✅ TRADE NOW!")
else:
    print("❌ Don't trade")
"""
