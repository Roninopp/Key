"""
MODULE 4: TELEGRAM BOT INTEGRATION
Complete Telegram bot with image analysis and signal generation
"""

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import asyncio
from datetime import datetime
import os

# Import all previous modules
from module1 import PocketOptionChartAnalyzer
from module2 import TimingSystemIntegrator
from module3 import SignalGenerator, PocketOptionTradingBot, SignalDirection

# BOT CONFIGURATION
BOT_TOKEN = "6506132532:AAGjfMXlSkefR5uldDwCRhxdk7YRES5385k"
AUTHORIZED_USER_ID = 6837532865  # Only you can use this bot

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class PocketOptionTelegramBot:
    """
    Main Telegram bot class
    """
    
    def __init__(self):
        self.bot_token = BOT_TOKEN
        self.authorized_user_id = AUTHORIZED_USER_ID
        self.trading_bot = PocketOptionTradingBot()
        self.temp_image_dir = "temp_charts"
        
        # Create temp directory
        if not os.path.exists(self.temp_image_dir):
            os.makedirs(self.temp_image_dir)
        
        # Statistics
        self.stats = {
            'total_analyses': 0,
            'signals_sent': 0,
            'no_trade_signals': 0,
            'start_time': datetime.now()
        }
    
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        return user_id == self.authorized_user_id
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await update.message.reply_text(
                "❌ **Unauthorized Access**\n\n"
                "This bot is private and only accessible to authorized users.\n"
                f"Your ID: `{user_id}`",
                parse_mode='Markdown'
            )
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            return
        
        welcome_message = (
            "🤖 **Pocket Option Trading Bot**\n\n"
            "Welcome! I'm your personal trading assistant.\n\n"
            "📊 **How to use:**\n"
            "1. Take a screenshot of your Pocket Option chart\n"
            "2. Send it to me\n"
            "3. I'll analyze and send you a signal!\n\n"
            "⚡ **Features:**\n"
            "✓ Price action analysis\n"
            "✓ Pattern detection\n"
            "✓ Entry timing calculation\n"
            "✓ Confidence scoring\n"
            "✓ Risk warnings\n\n"
            "📝 **Commands:**\n"
            "/start - Show this message\n"
            "/stats - View statistics\n"
            "/help - Get help\n\n"
            "🎯 Ready to analyze! Send me a chart screenshot."
        )
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info(f"Bot started by user {user_id}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        help_text = (
            "❓ **HELP - How to Use**\n\n"
            "**Step 1:** Open Pocket Option app/web\n"
            "**Step 2:** Take a clear screenshot of the chart\n"
            "   - Make sure timeframe is visible\n"
            "   - Include at least 20 candles\n"
            "   - Clear visibility of patterns\n\n"
            "**Step 3:** Send screenshot to this bot\n\n"
            "**Step 4:** Wait 2-3 seconds for analysis\n\n"
            "**Step 5:** I'll send you:\n"
            "   • Direction (UP/DOWN)\n"
            "   • Confidence %\n"
            "   • Entry timing\n"
            "   • Reasoning\n"
            "   • Warnings\n\n"
            "💡 **Tips:**\n"
            "• Only trade signals with 65%+ confidence\n"
            "• Follow entry timing exactly\n"
            "• Don't trade expired signals\n"
            "• Respect warnings\n\n"
            "⚠️ **Risk Warning:**\n"
            "This bot provides signals based on technical analysis. "
            "Always trade responsibly and use proper risk management. "
            "Past performance does not guarantee future results."
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        uptime = datetime.now() - self.stats['start_time']
        hours = uptime.total_seconds() / 3600
        
        stats_text = (
            "📊 **BOT STATISTICS**\n\n"
            f"⏰ Uptime: {hours:.1f} hours\n"
            f"📈 Total Analyses: {self.stats['total_analyses']}\n"
            f"✅ Signals Sent: {self.stats['signals_sent']}\n"
            f"⏸️ No-Trade Signals: {self.stats['no_trade_signals']}\n\n"
            f"🤖 Status: Active ✅"
        )
        
        await update.message.reply_text(stats_text, parse_mode='Markdown')
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming chart images"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await update.message.reply_text("❌ Unauthorized access denied.")
            return
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🔍 **Analyzing chart...**\n\n"
            "⏳ Please wait 2-3 seconds...",
            parse_mode='Markdown'
        )
        
        try:
            # Download photo
            photo = update.message.photo[-1]  # Get highest resolution
            file = await context.bot.get_file(photo.file_id)
            
            # Save temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(self.temp_image_dir, f"chart_{timestamp}.jpg")
            await file.download_to_drive(image_path)
            
            logger.info(f"Image received from user {user_id}, saved to {image_path}")
            
            # Analyze chart
            signal = self.trading_bot.analyze_and_generate_signal(image_path)
            
            # Update statistics
            self.stats['total_analyses'] += 1
            if signal.should_trade:
                self.stats['signals_sent'] += 1
            else:
                self.stats['no_trade_signals'] += 1
            
            # Delete processing message
            await processing_msg.delete()
            
            # Send signal
            await self._send_signal(update, signal)
            
            # Clean up
            try:
                os.remove(image_path)
            except:
                pass
            
            logger.info(f"Analysis complete for user {user_id}: {signal.direction.value}")
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await processing_msg.edit_text(
                "❌ **Error Processing Chart**\n\n"
                f"Error: {str(e)}\n\n"
                "Please try:\n"
                "• Sending a clearer screenshot\n"
                "• Ensuring chart is fully visible\n"
                "• Checking image quality",
                parse_mode='Markdown'
            )
    
    async def _send_signal(self, update: Update, signal):
        """Send formatted signal to user"""
        
        # Choose emoji based on signal
        if signal.direction == SignalDirection.UP:
            emoji = "🟢"
        elif signal.direction == SignalDirection.DOWN:
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        # Format confidence bar
        confidence_bars = int(signal.confidence / 10)
        confidence_visual = "█" * confidence_bars + "░" * (10 - confidence_bars)
        
        # Build message
        msg = f"{emoji} **SIGNAL ANALYSIS**\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        if signal.should_trade:
            msg += f"**🎯 Direction:** {signal.direction.value}\n"
            msg += f"**💪 Confidence:** {signal.confidence:.0f}%\n"
            msg += f"`{confidence_visual}` {signal.confidence:.0f}%\n\n"
            
            msg += f"**📊 Level:** {signal.confidence_level.value}\n"
            msg += f"**⏰ Timeframe:** {signal.timeframe}\n"
            msg += f"**🕐 Entry:** {signal.entry_time.strftime('%H:%M:%S')}\n\n"
            
            msg += "**📈 Analysis:**\n"
            for reason in signal.reasoning[:4]:
                msg += f"{reason}\n"
            
            if signal.warnings:
                msg += "\n**⚠️ Warnings:**\n"
                for warning in signal.warnings[:3]:
                    msg += f"{warning}\n"
            
            msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
            
            if signal.confidence >= 80:
                msg += "✅ **STRONG SIGNAL**\n"
                msg += "💰 Recommended trade!"
            elif signal.confidence >= 70:
                msg += "✅ **GOOD SIGNAL**\n"
                msg += "👍 Trade with confidence"
            elif signal.confidence >= 60:
                msg += "⚠️ **MODERATE SIGNAL**\n"
                msg += "🤔 Trade with caution"
            else:
                msg += "⏸️ **WEAK SIGNAL**\n"
                msg += "❌ Consider skipping"
        
        else:
            msg += "⏸️ **NO TRADE**\n\n"
            msg += signal.reasoning[0] if signal.reasoning else "No clear setup detected"
            msg += "\n\n⏳ Wait for better opportunity"
        
        msg += "\n\n💡 _Trade at your own risk. Not financial advice._"
        
        # Create feedback buttons
        keyboard = [
            [
                InlineKeyboardButton("✅ WIN", callback_data=f"feedback_win_{signal.confidence}"),
                InlineKeyboardButton("❌ LOSS", callback_data=f"feedback_loss_{signal.confidence}"),
                InlineKeyboardButton("⏭️ SKIP", callback_data="feedback_skip")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            msg,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle user feedback on signals"""
        query = update.callback_query
        await query.answer()
        
        if not self.is_authorized(query.from_user.id):
            return
        
        data = query.data
        
        if data.startswith("feedback_win"):
            await query.edit_message_text(
                query.message.text + "\n\n✅ **Result: WIN** 🎉",
                parse_mode='Markdown'
            )
            logger.info(f"Feedback WIN received")
        
        elif data.startswith("feedback_loss"):
            await query.edit_message_text(
                query.message.text + "\n\n❌ **Result: LOSS** 😔",
                parse_mode='Markdown'
            )
            logger.info(f"Feedback LOSS received")
        
        elif data == "feedback_skip":
            await query.edit_message_text(
                query.message.text + "\n\n⏭️ **Skipped**",
                parse_mode='Markdown'
            )
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads (in case user sends image as file)"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        await update.message.reply_text(
            "📎 Please send the chart as a **photo**, not as a document.\n\n"
            "Tap the 📷 icon and select your screenshot.",
            parse_mode='Markdown'
        )
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        await update.message.reply_text(
            "📸 **Please send a chart screenshot**\n\n"
            "I can only analyze images. Send me a screenshot of your Pocket Option chart.\n\n"
            "Use /help for instructions.",
            parse_mode='Markdown'
        )
    
    def run(self):
        """Start the bot"""
        logger.info("Starting Pocket Option Trading Bot...")
        
        # Create application
        application = Application.builder().token(self.bot_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        application.add_handler(MessageHandler(filters.Document.IMAGE, self.handle_document))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        application.add_handler(CallbackQueryHandler(self.handle_feedback))
        
        # Start bot
        logger.info(f"Bot started! Authorized user: {self.authorized_user_id}")
        print("\n" + "="*50)
        print("🤖 POCKET OPTION TRADING BOT")
        print("="*50)
        print(f"✅ Status: RUNNING")
        print(f"🔐 Authorized User ID: {self.authorized_user_id}")
        print(f"📱 Bot Token: {self.bot_token[:20]}...")
        print("\n💡 Send /start to the bot to begin!")
        print("="*50 + "\n")
        
        # Run bot
        application.run_polling(allowed_updates=Update.ALL_TYPES)


# MAIN ENTRY POINT
if __name__ == "__main__":
    try:
        bot = PocketOptionTelegramBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("\n\n👋 Bot stopped. Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Error: {e}")


"""
INSTALLATION REQUIREMENTS:

pip install python-telegram-bot
pip install opencv-python
pip install pytesseract
pip install pillow
pip install numpy

Also install Tesseract OCR:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: sudo apt-get install tesseract-ocr
- Mac: brew install tesseract

USAGE:

1. Save all 4 modules in same directory:
   - module1.py (Price Action Analyzer)
   - module2.py (Timing System)
   - module3.py (Signal Generator)
   - module4.py (This file - Telegram Bot)

2. Run: python module4.py

3. Open Telegram and start chat with your bot

4. Send /start

5. Send chart screenshots!

FEATURES:
✅ Private bot (only you can access)
✅ Image analysis
✅ Signal generation
✅ Entry timing
✅ Confidence scoring
✅ Feedback system (WIN/LOSS)
✅ Statistics tracking
✅ Error handling
✅ Logging
"""
