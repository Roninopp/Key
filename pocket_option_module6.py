"""
MODULE 6: COMPLETE INTEGRATION - POCKET OPTION TRADING BOT
Final production-ready bot with all modules integrated
CRITICAL: This is a REAL TRADING BOT - Handle with extreme care
"""

import logging
import os
import hashlib
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import json

# Import all modules
from module1 import PocketOptionChartAnalyzer
from module2 import TimingSystemIntegrator
from module3 import SignalGenerator, SignalDirection
from module5 import TradingDatabase, SmartConfidenceAdjuster, TradeRecord, TradeResult

# CONFIGURATION
BOT_TOKEN = "6506132532:AAGjfMXlSkefR5uldDwCRhxdk7YRES5385k"
AUTHORIZED_USER_ID = 6837532865

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompletePocketOptionBot:
    """
    Production-ready Pocket Option trading bot
    CRITICAL: Real money trading - maximum precision required
    """
    
    def __init__(self):
        self.bot_token = BOT_TOKEN
        self.authorized_user_id = AUTHORIZED_USER_ID
        
        # Initialize all components
        self.price_analyzer = PocketOptionChartAnalyzer()
        self.timing_system = TimingSystemIntegrator()
        self.signal_generator = SignalGenerator()
        self.database = TradingDatabase("pocket_option_trades.db")
        self.confidence_adjuster = SmartConfidenceAdjuster(self.database)
        
        # Temp storage
        self.temp_image_dir = "temp_charts"
        if not os.path.exists(self.temp_image_dir):
            os.makedirs(self.temp_image_dir)
        
        # Session tracking
        self.pending_trades = {}  # trade_id -> signal info
        
        logger.info("✅ Complete Pocket Option Trading Bot initialized")
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash to prevent duplicate analysis"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def is_authorized(self, user_id: int) -> bool:
        """Security check"""
        return user_id == self.authorized_user_id
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await update.message.reply_text(
                "❌ **UNAUTHORIZED ACCESS**\n\n"
                "This is a private trading bot.\n"
                f"Your ID: `{user_id}`\n\n"
                "⚠️ Access denied.",
                parse_mode='Markdown'
            )
            logger.warning(f"🚨 Unauthorized access attempt: {user_id}")
            return
        
        # Get statistics for welcome message
        stats = self.database.get_overall_statistics()
        
        welcome = (
            "🤖 **POCKET OPTION TRADING BOT**\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "⚠️ **REAL MONEY TRADING BOT**\n"
            "Use with caution and proper risk management.\n\n"
            "📊 **Current Statistics:**\n"
            f"   Total Trades: {stats['total_trades']}\n"
            f"   Win Rate: {stats['win_rate']:.1f}%\n"
            f"   Avg Confidence: {stats['avg_confidence']:.1f}%\n\n"
            "📸 **How to Use:**\n"
            "1. Take clear Pocket Option chart screenshot\n"
            "2. Send it to me\n"
            "3. Wait 2-3 seconds for analysis\n"
            "4. Get signal with timing\n"
            "5. Report result (WIN/LOSS)\n\n"
            "📝 **Commands:**\n"
            "/start - This message\n"
            "/stats - Detailed statistics\n"
            "/insights - AI learning insights\n"
            "/help - Full instructions\n\n"
            "⚠️ **IMPORTANT:**\n"
            "• Only trade signals 65%+ confidence\n"
            "• Follow entry timing exactly\n"
            "• Always report results\n"
            "• Never trade expired signals\n\n"
            "🎯 Ready to analyze. Send chart screenshot!"
        )
        
        await update.message.reply_text(welcome, parse_mode='Markdown')
        logger.info(f"✅ Bot started by authorized user: {user_id}")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Detailed statistics"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        stats = self.database.get_overall_statistics()
        recent = self.database.get_recent_trades(limit=10)
        
        msg = "📊 **TRADING STATISTICS**\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        msg += f"**Overall Performance:**\n"
        msg += f"   Total Trades: {stats['total_trades']}\n"
        msg += f"   Wins: {stats['wins']} ✅\n"
        msg += f"   Losses: {stats['losses']} ❌\n"
        msg += f"   Win Rate: {stats['win_rate']:.1f}%\n"
        msg += f"   Avg Confidence: {stats['avg_confidence']:.1f}%\n\n"
        
        if stats['total_trades'] > 0:
            # Calculate streak
            streak = 0
            streak_type = ""
            for trade in recent:
                if trade.result == TradeResult.WIN:
                    if streak >= 0:
                        streak += 1
                        streak_type = "WIN"
                    else:
                        break
                elif trade.result == TradeResult.LOSS:
                    if streak <= 0:
                        streak -= 1
                        streak_type = "LOSS"
                    else:
                        break
            
            if abs(streak) > 0:
                emoji = "🔥" if streak_type == "WIN" else "❄️"
                msg += f"**Current Streak:** {abs(streak)} {streak_type} {emoji}\n\n"
        
        msg += f"**Recent Trades (Last 10):**\n"
        if recent:
            for i, trade in enumerate(recent[:10], 1):
                result_emoji = "✅" if trade.result == TradeResult.WIN else "❌" if trade.result == TradeResult.LOSS else "⏳"
                msg += f"{i}. {trade.direction} {result_emoji} ({trade.confidence:.0f}%)\n"
        else:
            msg += "   No trades yet\n"
        
        msg += "\n💡 Use /insights for AI recommendations"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def insights_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """AI learning insights"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        insights = self.database.get_learning_insights()
        
        msg = "🧠 **AI LEARNING INSIGHTS**\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        msg += f"**Data Points:** {insights.total_trades} trades\n"
        msg += f"**Overall Win Rate:** {insights.overall_win_rate:.1f}%\n\n"
        
        if insights.best_patterns:
            msg += "**🎯 Best Patterns:**\n"
            for i, pattern in enumerate(insights.best_patterns[:3], 1):
                if pattern.total_signals >= 5:
                    msg += f"{i}. {pattern.pattern_name.replace('_', ' ').title()}\n"
                    msg += f"   Win Rate: {pattern.win_rate:.0f}% ({pattern.wins}/{pattern.total_signals})\n"
            msg += "\n"
        
        if insights.worst_patterns:
            msg += "**⚠️ Avoid These Patterns:**\n"
            for pattern in insights.worst_patterns[:2]:
                if pattern.total_signals >= 5 and pattern.win_rate < 50:
                    msg += f"• {pattern.pattern_name.replace('_', ' ').title()}\n"
                    msg += f"   Only {pattern.win_rate:.0f}% win rate\n"
            msg += "\n"
        
        if insights.best_timeframes:
            msg += "**⏰ Best Timeframes:**\n"
            for tf, wr in list(insights.best_timeframes.items())[:3]:
                msg += f"• {tf}: {wr:.0f}% win rate\n"
            msg += "\n"
        
        if insights.best_trading_hours:
            msg += "**🕐 Best Trading Hours:**\n"
            hours_str = ", ".join(f"{h}:00" for h in insights.best_trading_hours[:5])
            msg += f"   {hours_str}\n\n"
        
        if insights.recommendations:
            msg += "**💡 Recommendations:**\n"
            for rec in insights.recommendations[:5]:
                msg += f"   {rec}\n"
        
        if insights.total_trades < 20:
            msg += "\n⚠️ **Need more data for accurate insights**\n"
            msg += "Continue trading and reporting results."
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help instructions"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        help_text = (
            "❓ **HOW TO USE THIS BOT**\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "**Step 1: Capture Chart**\n"
            "• Open Pocket Option\n"
            "• Wait for clear pattern formation\n"
            "• Take screenshot (include 20+ candles)\n"
            "• Make sure timeframe is visible\n\n"
            "**Step 2: Send to Bot**\n"
            "• Send screenshot as photo (not file)\n"
            "• Wait 2-3 seconds\n\n"
            "**Step 3: Review Signal**\n"
            "• Check direction (UP/DOWN)\n"
            "• Check confidence %\n"
            "• Check entry timing\n"
            "• Read warnings\n\n"
            "**Step 4: Make Decision**\n"
            "• 65%+ confidence: Consider trading\n"
            "• 75%+ confidence: Good signal\n"
            "• 85%+ confidence: Strong signal\n"
            "• <65% confidence: Skip\n\n"
            "**Step 5: Report Result**\n"
            "• Click ✅ WIN or ❌ LOSS button\n"
            "• Bot learns from your results\n"
            "• Improves future predictions\n\n"
            "⚠️ **CRITICAL WARNINGS:**\n"
            "• Never trade expired signals (>15s lag)\n"
            "• Always follow entry timing\n"
            "• Never trade against warnings\n"
            "• Use proper risk management\n"
            "• This is NOT financial advice\n\n"
            "💰 **Risk Management:**\n"
            "• Risk max 1-2% per trade\n"
            "• Stop after 3 consecutive losses\n"
            "• Take breaks after big wins/losses\n"
            "• Never trade emotionally\n\n"
            "📊 Commands:\n"
            "/stats - View statistics\n"
            "/insights - AI recommendations\n"
            "/help - This message"
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Main photo analysis handler - CRITICAL FUNCTION"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await update.message.reply_text("❌ Unauthorized")
            return
        
        # Processing message
        processing = await update.message.reply_text(
            "🔍 **ANALYZING CHART...**\n\n"
            "⏳ Extracting price action...\n"
            "⏳ Calculating timing...\n"
            "⏳ Generating signal...",
            parse_mode='Markdown'
        )
        
        try:
            # Download image
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(self.temp_image_dir, f"chart_{timestamp}.jpg")
            await file.download_to_drive(image_path)
            
            # Calculate hash to prevent duplicate analysis
            chart_hash = self._calculate_image_hash(image_path)
            
            logger.info(f"📸 Chart received - Hash: {chart_hash[:8]}")
            
            # STEP 1: Analyze price action
            price_analysis = self.price_analyzer.analyze_chart(image_path)
            logger.info(f"✅ Price action analyzed - {len(price_analysis.detected_patterns)} patterns")
            
            # STEP 2: Analyze timing
            candle_completion = 90.0
            if price_analysis.candles:
                last = price_analysis.candles[-1]
                if last.total_range > 0:
                    candle_completion = (last.body_size / last.total_range) * 100
            
            timing_info = self.timing_system.analyze_chart_timing(image_path, candle_completion)
            logger.info(f"✅ Timing analyzed - Lag: {timing_info.time_lag:.1f}s")
            
            # STEP 3: Generate base signal
            signal = self.signal_generator.generate_signal(price_analysis, timing_info)
            logger.info(f"✅ Signal generated - {signal.direction.value} @ {signal.confidence:.0f}%")
            
            # STEP 4: Apply AI learning adjustment
            if signal.should_trade and self.database.get_overall_statistics()['total_trades'] >= 10:
                patterns_list = [p[0].name for p in price_analysis.detected_patterns]
                current_hour = datetime.now().hour
                
                adjusted_conf, adj_reasons = self.confidence_adjuster.adjust_confidence(
                    signal.confidence,
                    patterns_list,
                    timing_info.timeframe,
                    current_hour
                )
                
                if abs(adjusted_conf - signal.confidence) > 2:
                    logger.info(f"🧠 AI adjusted confidence: {signal.confidence:.0f}% → {adjusted_conf:.0f}%")
                    signal.confidence = adjusted_conf
                    signal.reasoning.extend(adj_reasons)
            
            # STEP 5: Save to database
            trade = TradeRecord(
                id=None,
                timestamp=datetime.now(),
                asset="Unknown",  # Could be extracted from chart
                timeframe=timing_info.timeframe,
                direction=signal.direction.value,
                confidence=signal.confidence,
                patterns=json.dumps([p[0].name for p in price_analysis.detected_patterns]),
                trend=price_analysis.trend.value,
                result=TradeResult.PENDING,
                entry_time=timing_info.next_candle_open,
                chart_hash=chart_hash
            )
            
            trade_id = self.database.save_trade(trade)
            
            if trade_id == -1:
                await processing.edit_text(
                    "⚠️ **DUPLICATE CHART DETECTED**\n\n"
                    "You already analyzed this exact chart.\n"
                    "Please send a new screenshot.",
                    parse_mode='Markdown'
                )
                os.remove(image_path)
                return
            
            logger.info(f"💾 Trade saved - ID: {trade_id}")
            
            # Store for feedback
            self.pending_trades[trade_id] = signal
            
            # Delete processing message
            await processing.delete()
            
            # STEP 6: Send formatted signal
            await self._send_complete_signal(update, signal, timing_info, trade_id)
            
            # Cleanup
            os.remove(image_path)
            
        except Exception as e:
            logger.error(f"❌ ERROR analyzing chart: {e}", exc_info=True)
            await processing.edit_text(
                f"❌ **ANALYSIS ERROR**\n\n"
                f"Error: {str(e)}\n\n"
                "Please:\n"
                "• Send clearer screenshot\n"
                "• Ensure full chart visible\n"
                "• Check image quality\n\n"
                "If problem persists, contact support.",
                parse_mode='Markdown'
            )
    
    async def _send_complete_signal(self, update: Update, signal, timing_info, trade_id: int):
        """Send complete trading signal with all information"""
        
        # Signal header
        if signal.direction == SignalDirection.UP:
            emoji = "🟢"
            color = "GREEN"
        elif signal.direction == SignalDirection.DOWN:
            emoji = "🔴"
            color = "RED"
        else:
            emoji = "⚪"
            color = "NEUTRAL"
        
        # Build message
        msg = f"{emoji} **TRADING SIGNAL #{trade_id}**\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        if signal.should_trade:
            # Confidence visualization
            bars = int(signal.confidence / 10)
            conf_bar = "█" * bars + "░" * (10 - bars)
            
            msg += f"**🎯 DIRECTION:** {signal.direction.value}\n"
            msg += f"**💪 CONFIDENCE:** {signal.confidence:.0f}%\n"
            msg += f"`{conf_bar}` {signal.confidence:.0f}%\n\n"
            
            msg += f"**⏰ TIMING:**\n"
            msg += f"   Timeframe: {timing_info.timeframe}\n"
            msg += f"   Entry: {timing_info.next_candle_open.strftime('%H:%M:%S')}\n"
            msg += f"   Countdown: {timing_info.seconds_until_entry:.0f}s\n"
            msg += f"   {timing_info.entry_message}\n\n"
            
            msg += f"**📊 ANALYSIS:**\n"
            for reason in signal.reasoning[:5]:
                msg += f"   {reason}\n"
            
            if signal.warnings:
                msg += f"\n**⚠️ WARNINGS:**\n"
                for warning in signal.warnings[:3]:
                    msg += f"   {warning}\n"
            
            msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
            
            # Trading recommendation
            if signal.confidence >= 85:
                msg += "✅ **STRONG SIGNAL**\n"
                msg += "💰 High confidence trade\n"
            elif signal.confidence >= 75:
                msg += "✅ **GOOD SIGNAL**\n"
                msg += "👍 Recommended trade\n"
            elif signal.confidence >= 65:
                msg += "⚠️ **MODERATE SIGNAL**\n"
                msg += "🤔 Trade with caution\n"
            else:
                msg += "⏸️ **WEAK SIGNAL**\n"
                msg += "❌ Consider skipping\n"
            
            msg += f"\n⏱️ **ENTRY: {timing_info.next_candle_open.strftime('%H:%M:%S')}**"
            
        else:
            msg += "⏸️ **NO TRADE RECOMMENDED**\n\n"
            msg += f"**Reason:**\n"
            msg += signal.reasoning[0] if signal.reasoning else "No clear setup"
            msg += "\n\n⏳ Wait for better opportunity"
        
        msg += "\n\n💡 _Not financial advice. Trade at your own risk._"
        
        # Feedback buttons
        keyboard = [
            [
                InlineKeyboardButton("✅ WIN", callback_data=f"result_win_{trade_id}"),
                InlineKeyboardButton("❌ LOSS", callback_data=f"result_loss_{trade_id}"),
                InlineKeyboardButton("⏭️ SKIP", callback_data=f"result_skip_{trade_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            msg,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle WIN/LOSS feedback - CRITICAL for learning"""
        query = update.callback_query
        await query.answer()
        
        if not self.is_authorized(query.from_user.id):
            return
        
        data = query.data
        parts = data.split('_')
        action = parts[1]  # win/loss/skip
        trade_id = int(parts[2])
        
        # Update database
        if action == "win":
            self.database.update_trade_result(trade_id, TradeResult.WIN)
            result_msg = "\n\n✅ **RESULT: WIN** 🎉"
            logger.info(f"✅ Trade #{trade_id} - WIN")
        elif action == "loss":
            self.database.update_trade_result(trade_id, TradeResult.LOSS)
            result_msg = "\n\n❌ **RESULT: LOSS** 😔"
            logger.info(f"❌ Trade #{trade_id} - LOSS")
        else:
            self.database.update_trade_result(trade_id, TradeResult.SKIP)
            result_msg = "\n\n⏭️ **SKIPPED**"
            logger.info(f"⏭️ Trade #{trade_id} - SKIPPED")
        
        # Update message
        await query.edit_message_text(
            query.message.text + result_msg,
            parse_mode='Markdown'
        )
        
        # Send follow-up statistics
        stats = self.database.get_overall_statistics()
        
        followup = (
            f"📊 **Updated Statistics:**\n"
            f"Total: {stats['total_trades']} | "
            f"Win Rate: {stats['win_rate']:.1f}%\n\n"
            "Thank you for feedback! 🙏"
        )
        
        await query.message.reply_text(followup, parse_mode='Markdown')
    
    async def handle_other(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle non-photo messages"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        await update.message.reply_text(
            "📸 **Please send chart screenshot**\n\n"
            "Send as photo (not file).\n"
            "Use /help for instructions.",
            parse_mode='Markdown'
        )
    
    def run(self):
        """Start the bot"""
        logger.info("🚀 Starting Complete Pocket Option Trading Bot...")
        
        application = Application.builder().token(self.bot_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("insights", self.insights_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        application.add_handler(CallbackQueryHandler(self.handle_feedback))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_other))
        
        # Start
        print("\n" + "="*60)
        print("🤖 POCKET OPTION TRADING BOT - PRODUCTION")
        print("="*60)
        print(f"✅ Status: RUNNING")
        print(f"🔐 Authorized User: {self.authorized_user_id}")
        print(f"📊 Database: Connected")
        print(f"🧠 AI Learning: Active")
        print("="*60)
        print("\n⚠️  REAL MONEY TRADING BOT")
        print("Use with proper risk management!\n")
        print("📱 Open Telegram and send /start to begin")
        print("="*60 + "\n")
        
        logger.info("✅ Bot is now running and accepting requests")
        
        # Run
        application.run_polling(allowed_updates=Update.ALL_TYPES)


# MAIN ENTRY
if __name__ == "__main__":
    try:
        bot = CompletePocketOptionBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
        print("\n\n👋 Trading bot stopped. Stay safe!")
    except Exception as e:
        logger.critical(f"💥 FATAL ERROR: {e}", exc_info=True)
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("Check trading_bot.log for details")
