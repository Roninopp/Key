"""
MODULE 5: SQLITE DATABASE & LEARNING SYSTEM
Smart memory system that learns from your trades and improves predictions
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum

class TradeResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    SKIP = "SKIP"
    PENDING = "PENDING"

@dataclass
class TradeRecord:
    """Single trade record"""
    id: Optional[int]
    timestamp: datetime
    asset: str
    timeframe: str
    direction: str
    confidence: float
    patterns: str  # JSON
    trend: str
    result: TradeResult
    entry_time: datetime
    chart_hash: str

@dataclass
class PatternStats:
    """Pattern performance statistics"""
    pattern_name: str
    total_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_confidence: float
    best_timeframe: str

@dataclass
class LearningInsights:
    """AI learning insights"""
    total_trades: int
    overall_win_rate: float
    best_patterns: List[PatternStats]
    worst_patterns: List[PatternStats]
    best_timeframes: Dict[str, float]
    best_trading_hours: List[int]
    confidence_calibration: Dict[str, float]
    recommendations: List[str]


class TradingDatabase:
    """
    SQLite database for storing and analyzing trades
    """
    
    def __init__(self, db_path: str = "pocket_option_trades.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                patterns TEXT NOT NULL,
                trend TEXT NOT NULL,
                result TEXT NOT NULL,
                entry_time DATETIME NOT NULL,
                chart_hash TEXT UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Pattern performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_performance (
                pattern_name TEXT PRIMARY KEY,
                total_signals INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Timeframe statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timeframe_stats (
                timeframe TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0
            )
        """)
        
        # Hourly performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_performance (
                hour INTEGER PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0
            )
        """)
        
        # Confidence calibration
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS confidence_calibration (
                confidence_range TEXT PRIMARY KEY,
                total_signals INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                actual_win_rate REAL DEFAULT 0.0
            )
        """)
        
        # Bot settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bot_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        print("✅ Database initialized successfully")
    
    def save_trade(self, trade: TradeRecord) -> int:
        """Save trade to database"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO trades (timestamp, asset, timeframe, direction, confidence, 
                                  patterns, trend, result, entry_time, chart_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.timestamp,
                trade.asset,
                trade.timeframe,
                trade.direction,
                trade.confidence,
                trade.patterns,
                trade.trend,
                trade.result.value,
                trade.entry_time,
                trade.chart_hash
            ))
            
            self.conn.commit()
            return cursor.lastrowid
            
        except sqlite3.IntegrityError:
            # Duplicate chart hash
            return -1
    
    def update_trade_result(self, trade_id: int, result: TradeResult):
        """Update trade result (WIN/LOSS)"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE trades SET result = ? WHERE id = ?
        """, (result.value, trade_id))
        
        self.conn.commit()
        
        # Update statistics
        self._update_statistics()
    
    def get_recent_trades(self, limit: int = 50) -> List[TradeRecord]:
        """Get recent trades"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        
        trades = []
        for row in cursor.fetchall():
            trades.append(TradeRecord(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                asset=row[2],
                timeframe=row[3],
                direction=row[4],
                confidence=row[5],
                patterns=row[6],
                trend=row[7],
                result=TradeResult(row[8]),
                entry_time=datetime.fromisoformat(row[9]),
                chart_hash=row[10]
            ))
        
        return trades
    
    def get_overall_statistics(self) -> Dict:
        """Get overall trading statistics"""
        cursor = self.conn.cursor()
        
        # Total trades
        cursor.execute("SELECT COUNT(*) FROM trades WHERE result != 'PENDING'")
        total = cursor.fetchone()[0]
        
        # Wins
        cursor.execute("SELECT COUNT(*) FROM trades WHERE result = 'WIN'")
        wins = cursor.fetchone()[0]
        
        # Losses
        cursor.execute("SELECT COUNT(*) FROM trades WHERE result = 'LOSS'")
        losses = cursor.fetchone()[0]
        
        # Win rate
        win_rate = (wins / total * 100) if total > 0 else 0
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM trades WHERE result != 'PENDING'")
        avg_confidence = cursor.fetchone()[0] or 0
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_confidence': avg_confidence
        }
    
    def _update_statistics(self):
        """Update all statistics tables"""
        self._update_pattern_performance()
        self._update_timeframe_stats()
        self._update_hourly_performance()
        self._update_confidence_calibration()
    
    def _update_pattern_performance(self):
        """Update pattern performance statistics"""
        cursor = self.conn.cursor()
        
        # Get all completed trades
        cursor.execute("""
            SELECT patterns, result, confidence FROM trades 
            WHERE result IN ('WIN', 'LOSS')
        """)
        
        pattern_data = {}
        
        for row in cursor.fetchall():
            patterns = json.loads(row[0])
            result = row[1]
            confidence = row[2]
            
            for pattern in patterns:
                if pattern not in pattern_data:
                    pattern_data[pattern] = {
                        'total': 0,
                        'wins': 0,
                        'losses': 0,
                        'confidences': []
                    }
                
                pattern_data[pattern]['total'] += 1
                pattern_data[pattern]['confidences'].append(confidence)
                
                if result == 'WIN':
                    pattern_data[pattern]['wins'] += 1
                else:
                    pattern_data[pattern]['losses'] += 1
        
        # Update database
        for pattern, data in pattern_data.items():
            win_rate = (data['wins'] / data['total'] * 100) if data['total'] > 0 else 0
            avg_conf = sum(data['confidences']) / len(data['confidences']) if data['confidences'] else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO pattern_performance 
                (pattern_name, total_signals, wins, losses, win_rate, avg_confidence, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pattern, data['total'], data['wins'], data['losses'], 
                  win_rate, avg_conf, datetime.now()))
        
        self.conn.commit()
    
    def _update_timeframe_stats(self):
        """Update timeframe statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT timeframe, COUNT(*) as total,
                   SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                   AVG(confidence) as avg_conf
            FROM trades
            WHERE result IN ('WIN', 'LOSS')
            GROUP BY timeframe
        """)
        
        for row in cursor.fetchall():
            timeframe, total, wins, losses, avg_conf = row
            win_rate = (wins / total * 100) if total > 0 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO timeframe_stats
                (timeframe, total_trades, wins, losses, win_rate, avg_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timeframe, total, wins, losses, win_rate, avg_conf))
        
        self.conn.commit()
    
    def _update_hourly_performance(self):
        """Update hourly performance statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT strftime('%H', timestamp) as hour,
                   COUNT(*) as total,
                   SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses
            FROM trades
            WHERE result IN ('WIN', 'LOSS')
            GROUP BY hour
        """)
        
        for row in cursor.fetchall():
            hour, total, wins, losses = row
            win_rate = (wins / total * 100) if total > 0 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO hourly_performance
                (hour, total_trades, wins, losses, win_rate)
                VALUES (?, ?, ?, ?, ?)
            """, (int(hour), total, wins, losses, win_rate))
        
        self.conn.commit()
    
    def _update_confidence_calibration(self):
        """Check if confidence scores match actual results"""
        cursor = self.conn.cursor()
        
        ranges = [
            ('60-70', 60, 70),
            ('70-80', 70, 80),
            ('80-90', 80, 90),
            ('90-100', 90, 100)
        ]
        
        for range_name, min_conf, max_conf in ranges:
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins
                FROM trades
                WHERE result IN ('WIN', 'LOSS')
                AND confidence >= ? AND confidence < ?
            """, (min_conf, max_conf))
            
            row = cursor.fetchone()
            total, wins = row
            actual_rate = (wins / total * 100) if total > 0 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO confidence_calibration
                (confidence_range, total_signals, wins, actual_win_rate)
                VALUES (?, ?, ?, ?)
            """, (range_name, total, wins, actual_rate))
        
        self.conn.commit()
    
    def get_pattern_performance(self) -> List[PatternStats]:
        """Get all pattern performance statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT pattern_name, total_signals, wins, losses, win_rate, avg_confidence
            FROM pattern_performance
            ORDER BY win_rate DESC
        """)
        
        stats = []
        for row in cursor.fetchall():
            stats.append(PatternStats(
                pattern_name=row[0],
                total_signals=row[1],
                wins=row[2],
                losses=row[3],
                win_rate=row[4],
                avg_confidence=row[5],
                best_timeframe=""  # Can be calculated if needed
            ))
        
        return stats
    
    def get_learning_insights(self) -> LearningInsights:
        """Generate AI learning insights"""
        overall_stats = self.get_overall_statistics()
        pattern_stats = self.get_pattern_performance()
        
        # Best patterns (top 3)
        best_patterns = sorted(pattern_stats, key=lambda x: x.win_rate, reverse=True)[:3]
        
        # Worst patterns (bottom 3)
        worst_patterns = sorted(pattern_stats, key=lambda x: x.win_rate)[:3]
        
        # Best timeframes
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT timeframe, win_rate FROM timeframe_stats
            ORDER BY win_rate DESC
        """)
        best_timeframes = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Best trading hours
        cursor.execute("""
            SELECT hour FROM hourly_performance
            WHERE win_rate > 60
            ORDER BY win_rate DESC
            LIMIT 5
        """)
        best_hours = [row[0] for row in cursor.fetchall()]
        
        # Confidence calibration
        cursor.execute("""
            SELECT confidence_range, actual_win_rate FROM confidence_calibration
        """)
        calibration = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_stats, best_patterns, worst_patterns, best_hours, calibration
        )
        
        return LearningInsights(
            total_trades=overall_stats['total_trades'],
            overall_win_rate=overall_stats['win_rate'],
            best_patterns=best_patterns,
            worst_patterns=worst_patterns,
            best_timeframes=best_timeframes,
            best_trading_hours=best_hours,
            confidence_calibration=calibration,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, stats, best, worst, hours, calibration) -> List[str]:
        """Generate smart recommendations based on data"""
        recommendations = []
        
        # Win rate recommendations
        if stats['win_rate'] < 55:
            recommendations.append("⚠️ Win rate below 55% - Reduce trading frequency")
        elif stats['win_rate'] > 70:
            recommendations.append("✅ Excellent win rate! Keep following your strategy")
        
        # Pattern recommendations
        if best:
            recommendations.append(f"🎯 Focus on: {best[0].pattern_name} ({best[0].win_rate:.0f}% win rate)")
        
        if worst:
            recommendations.append(f"⚠️ Avoid: {worst[0].pattern_name} ({worst[0].win_rate:.0f}% win rate)")
        
        # Time recommendations
        if hours:
            recommendations.append(f"⏰ Best trading hours: {', '.join(str(h) for h in hours[:3])}:00")
        
        # Confidence calibration
        for range_name, actual_rate in calibration.items():
            expected_min = int(range_name.split('-')[0])
            if actual_rate < expected_min - 10:
                recommendations.append(f"📉 {range_name}% confidence only achieving {actual_rate:.0f}%")
        
        # Minimum trades
        if stats['total_trades'] < 20:
            recommendations.append("📊 Need more data - Continue trading to improve accuracy")
        
        return recommendations
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class SmartConfidenceAdjuster:
    """
    Adjusts confidence scores based on historical performance
    """
    
    def __init__(self, database: TradingDatabase):
        self.db = database
    
    def adjust_confidence(self, base_confidence: float, patterns: List[str], 
                         timeframe: str, hour: int) -> Tuple[float, List[str]]:
        """
        Adjust confidence based on historical data
        
        Returns: (adjusted_confidence, adjustment_reasons)
        """
        adjusted = base_confidence
        reasons = []
        
        # Pattern performance adjustment
        pattern_stats = self.db.get_pattern_performance()
        for pattern in patterns:
            for stat in pattern_stats:
                if stat.pattern_name.lower() in pattern.lower():
                    if stat.total_signals >= 10:  # Enough data
                        if stat.win_rate > 70:
                            adjusted += 5
                            reasons.append(f"✅ {pattern} has {stat.win_rate:.0f}% historical win rate")
                        elif stat.win_rate < 50:
                            adjusted -= 5
                            reasons.append(f"⚠️ {pattern} has only {stat.win_rate:.0f}% win rate")
        
        # Timeframe adjustment
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT win_rate FROM timeframe_stats WHERE timeframe = ?
        """, (timeframe,))
        
        row = cursor.fetchone()
        if row and row[0]:
            tf_win_rate = row[0]
            if tf_win_rate > 70:
                adjusted += 3
                reasons.append(f"✅ {timeframe} timeframe: {tf_win_rate:.0f}% win rate")
            elif tf_win_rate < 50:
                adjusted -= 3
                reasons.append(f"⚠️ {timeframe} timeframe: {tf_win_rate:.0f}% win rate")
        
        # Hour adjustment
        cursor.execute("""
            SELECT win_rate FROM hourly_performance WHERE hour = ?
        """, (hour,))
        
        row = cursor.fetchone()
        if row and row[0]:
            hour_win_rate = row[0]
            if hour_win_rate > 65:
                adjusted += 2
                reasons.append(f"✅ Good trading hour ({hour}:00)")
            elif hour_win_rate < 45:
                adjusted -= 3
                reasons.append(f"⚠️ Poor trading hour ({hour}:00)")
        
        # Cap between 0-100
        adjusted = max(0, min(100, adjusted))
        
        return adjusted, reasons


# USAGE EXAMPLE
if __name__ == "__main__":
    
    # Initialize database
    db = TradingDatabase("pocket_option_trades.db")
    
    # Save a sample trade
    trade = TradeRecord(
        id=None,
        timestamp=datetime.now(),
        asset="EUR/USD",
        timeframe="1m",
        direction="UP",
        confidence=75.0,
        patterns=json.dumps(["BULLISH_ENGULFING", "HAMMER"]),
        trend="UPTREND",
        result=TradeResult.PENDING,
        entry_time=datetime.now(),
        chart_hash="abc123"
    )
    
    trade_id = db.save_trade(trade)
    print(f"✅ Trade saved with ID: {trade_id}")
    
    # Update result later
    # db.update_trade_result(trade_id, TradeResult.WIN)
    
    # Get statistics
    stats = db.get_overall_statistics()
    print(f"\n📊 Overall Stats: {stats}")
    
    # Get learning insights
    insights = db.get_learning_insights()
    print(f"\n🧠 AI Insights:")
    print(f"Total Trades: {insights.total_trades}")
    print(f"Win Rate: {insights.overall_win_rate:.1f}%")
    
    if insights.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in insights.recommendations:
            print(f"  {rec}")
    
    db.close()
