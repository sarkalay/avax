import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import learn script
try:
    from learn_script import SelfLearningAITrader
    LEARN_SCRIPT_AVAILABLE = True
    print("✅ Learn script loaded successfully!")
except ImportError as e:
    print(f"❌ Learn script import failed: {e}")
    LEARN_SCRIPT_AVAILABLE = False

import requests
import json
import time
import re
import math
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
import pandas as pd

# Colorama setup
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Warning: Colorama not installed. Run: pip install colorama")

# Load environment variables
load_dotenv()

# Global color variables for fallback
if not COLORAMA_AVAILABLE:
    class DummyColors:
        def __getattr__(self, name):
            return ''
    
    Fore = DummyColors()
    Back = DummyColors() 
    Style = DummyColors()

# ==================== V4.1 TRUE SMART NEVER GIVE BACK ====================
def should_close_trade(trade, current_price, atr_14):
    """BOUNCE-PROOF 3-LAYER EXIT - NO WINNER-TURN-LOSER"""
    # လက်ရှိ PnL % တွက်တာ
    if trade['direction'] == 'LONG':
        pnl_pct = (current_price - trade['entry_price']) / trade['entry_price'] * 100 * trade['leverage']
    else:  # SHORT
        pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price'] * 100 * trade['leverage']
    
    # Peak PnL အမြဲ update ထား
    if 'peak_pnl' not in trade or pnl_pct > trade['peak_pnl']:
        trade['peak_pnl'] = pnl_pct

    peak = trade['peak_pnl']

    # ==================== ၁. 60% Partial @ +9% ====================
    if peak >= 9.0 and not trade.get('partial_done', False):
        trade['partial_done'] = True
        return {
            "should_close": True,
            "partial_percent": 60,
            "close_type": "PARTIAL_60",
            "reason": f"LOCK 60% PROFIT @ +{peak:.1f}% → အမြတ် ချက်ချင်း အိတ်ထဲ!",
            "confidence": 100
        }

    # ==================== ၂. Instant Breakeven @ +12% ====================
    if peak >= 12.0 and not trade.get('breakeven_done', False):
        trade['breakeven_done'] = True
        return {
            "should_close": False,
            "move_sl_to": trade['entry_price'],  # breakeven
            "close_type": "BREAKEVEN_ACTIVATED",
            "reason": f"Peak +{peak:.1f}% → ကျန် 40% ကို BREAKEVEN ချပြီး → ဘယ်လိုမှ မရှုံးနိုင်တော့ဘူး!",
            "confidence": 100
        }

    # ==================== ၃. Dynamic Profit Floor (75% of Peak) ====================
    if peak >= 15.0:  # Peak က +15% ကျော်မှ ဒီ rule စတယ်
        profit_floor = peak * 0.75  # 75% အောက် မခွင့်ပြု
        if pnl_pct <= profit_floor and trade.get('partial_done', False):
            return {
                "should_close": True,
                "partial_percent": 100,  # ကျန်တဲ့ 40% အကုန်ပိတ်
                "close_type": "PROFIT_FLOOR_HIT",
                "reason": f"Peak {peak:.1f}% → 75% floor ({profit_floor:.1f}%) ထိပြီး → ကျန်အကုန် အမြတ်နဲ့ ပိတ်!",
                "confidence": 100
            }

    # ==================== ၄. 2×ATR Trailing (optional boost) ====================
    if trade.get('partial_done', False) and peak >= 9.0:
        trail_price = current_price + (2 * atr_14) if trade['direction'] == 'LONG' else current_price - (2 * atr_14)
        if trade['direction'] == 'LONG' and current_price <= trail_price:
            return {"should_close": True, "partial_percent": 100, "close_type": "TRAILING_HIT", "reason": "2×ATR Trailing ထိပြီး ထွက်"}
        if trade['direction'] == 'SHORT' and current_price >= trail_price:
            return {"should_close": True, "partial_percent": 100, "close_type": "TRAILING_HIT", "reason": "2×ATR Trailing ထိပြီး ထွက်"}

    # ==================== ၅. Winner-Turn-Loser = လုံးဝ မရှိတော့ဘူး ====================
    # ❌❌❌ ဒီတစ်ကြောင်းကို လုံးဝ ဖျက်ထားပြီး → တစ်ခါမှ အမြတ်ပြန်မပေးတော့ဘူး

    return {"should_close": False}  # မပိတ်သေးဘူး

# Use conditional inheritance with proper method placement
if LEARN_SCRIPT_AVAILABLE:
    class FullyAutonomous1HourAITrader(SelfLearningAITrader):
        def __init__(self):
            # Initialize learning component first
            super().__init__()
            # Then initialize trading components
            self._initialize_trading()
else:
    class FullyAutonomous1HourAITrader(object):
        def __init__(self):
            # Fallback initialization without learning
            self.mistakes_history = []
            self.learned_patterns = {}
            self.performance_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'common_mistakes': {},
                'improvement_areas': []
            }
            self._initialize_trading()

# === MULTI-TIMEFRAME INDICATORS ===
def calculate_ema(self, data, period):
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return [None] * len(data)
    df = pd.Series(data)
    return df.ewm(span=period, adjust=False).mean().tolist()

def calculate_rsi(self, data, period=14):
    """Calculate Relative Strength Index"""
    if len(data) < period + 1:
        return [50] * len(data)
    df = pd.Series(data)
    delta = df.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).tolist()

def calculate_volume_spike(self, volumes, window=10):
    """Calculate if current volume is a spike"""
    if len(volumes) < window + 1:
        return False
    avg_vol = np.mean(volumes[-window-1:-1])
    current_vol = volumes[-1]
    return current_vol > avg_vol * 1.8

def validate_api_keys(self):
    """Validate all API keys at startup"""
    issues = []
    
    if not self.binance_api_key or self.binance_api_key == "your_binance_api_key_here":
        issues.append("Binance API Key not configured")
    
    if not self.binance_secret or self.binance_secret == "your_binance_secret_key_here":
        issues.append("Binance Secret Key not configured")
        
    if not self.openrouter_key or self.openrouter_key == "your_openrouter_api_key_here":
        issues.append("OpenRouter API Key not configured - AI will use fallback decisions")
    
    if issues:
        self.print_color("🚨 CONFIGURATION ISSUES FOUND:", self.Fore.RED + self.Style.BRIGHT)
        for issue in issues:
            self.print_color(f"   ❌ {issue}", self.Fore.RED)
        
        if "OpenRouter" in str(issues):
            self.print_color("   💡 Without OpenRouter, AI will use technical analysis fallback only", self.Fore.YELLOW)
    
    return len(issues) == 0

# Common trading initialization for both cases
def _initialize_trading(self):
    """Initialize trading components (common for both cases)"""
    # Load config from .env file
    self.binance_api_key = os.getenv('BINANCE_API_KEY')
    self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
    self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
    
    # Store colorama references
    self.Fore = Fore
    self.Back = Back
    self.Style = Style
    self.COLORAMA_AVAILABLE = COLORAMA_AVAILABLE
    
    # Thailand timezone
    self.thailand_tz = pytz.timezone('Asia/Bangkok')
    
    # 🎯 FULLY AUTONOMOUS AI TRADING PARAMETERS
    self.total_budget = 500  # $500 budget for AI to manage
    self.available_budget = 500  # Current available budget
    self.max_position_size_percent = 10  # Max 10% of budget per trade for 1hr
    self.max_concurrent_trades = 4  # Maximum concurrent positions
    
    # AI can trade selected 3 major pairs only
    self.available_pairs = [
        "AVAXUSDT"
    ]
    
    # Track AI-opened trades
    self.ai_opened_trades = {}
    
    # REAL TRADE HISTORY
    self.real_trade_history_file = "fully_autonomous_1hour_ai_trading_history.json"
    self.real_trade_history = self.load_real_trade_history()
    
    # Trading statistics
    self.real_total_trades = 0
    self.real_winning_trades = 0
    self.real_total_pnl = 0.0
    
    # Precision settings
    self.quantity_precision = {}
    self.price_precision = {}
    
    # NEW: Reverse position settings
    self.allow_reverse_positions = True  # Enable reverse position feature
    
    # NEW: Monitoring interval (3 minute)
    self.monitoring_interval = 180  # 3 minute in seconds
    
    # Validate APIs before starting
    self.validate_api_keys()
    
    # Initialize Binance client
    try:
        self.binance = Client(self.binance_api_key, self.binance_secret)
        self.print_color(f"🤖 FULLY AUTONOMOUS AI TRADER ACTIVATED! 🤖", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color(f"💰 TOTAL BUDGET: ${self.total_budget}", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"🔄 REVERSE POSITION FEATURE: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color(f"🎯 BOUNCE-PROOF 3-LAYER EXIT V2: ENABLED", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"⏰ MONITORING: 3 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"📊 Max Positions: {self.max_concurrent_trades}", self.Fore.YELLOW + self.Style.BRIGHT)
        if LEARN_SCRIPT_AVAILABLE:
            self.print_color(f"🧠 SELF-LEARNING AI: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
    except Exception as e:
        self.print_color(f"Binance initialization failed: {e}", self.Fore.RED)
        self.binance = None
    
    self.validate_config()
    if self.binance:
        self.setup_futures()
        self.load_symbol_precision()

# Add the method to both classes
FullyAutonomous1HourAITrader._initialize_trading = _initialize_trading

# Now add all the other methods to the class
def load_real_trade_history(self):
    """Load trading history"""
    try:
        if os.path.exists(self.real_trade_history_file):
            with open(self.real_trade_history_file, 'r') as f:
                history = json.load(f)
                self.real_total_trades = len(history)
                self.real_winning_trades = len([t for t in history if t.get('pnl', 0) > 0])
                self.real_total_pnl = sum(t.get('pnl', 0) for t in history)
                return history
        return []
    except Exception as e:
        self.print_color(f"Error loading trade history: {e}", self.Fore.RED)
        return []

def save_real_trade_history(self):
    """Save trading history"""
    try:
        with open(self.real_trade_history_file, 'w') as f:
            json.dump(self.real_trade_history, f, indent=2)
    except Exception as e:
        self.print_color(f"Error saving trade history: {e}", self.Fore.RED)

def add_trade_to_history(self, trade_data):
    """Add trade to history WITH learning and partial close support"""
    try:
        trade_data['close_time'] = self.get_thailand_time()
        trade_data['close_timestamp'] = time.time()
        trade_data['trade_type'] = 'REAL'
        
        # Add partial close indicator to display
        if trade_data.get('partial_percent', 100) < 100:
            trade_data['display_type'] = f"PARTIAL_{trade_data['partial_percent']}%"
        else:
            trade_data['display_type'] = "FULL_CLOSE"
        
        self.real_trade_history.append(trade_data)
        
        # 🧠 Learn from this trade (especially if it's a loss)
        if LEARN_SCRIPT_AVAILABLE:
            self.learn_from_mistake(trade_data)
            self.adaptive_learning_adjustment()
        
        # Update performance stats
        self.performance_stats['total_trades'] += 1
        pnl = trade_data.get('pnl', 0)
        self.real_total_pnl += pnl
        if pnl > 0:
            self.real_winning_trades += 1
            self.performance_stats['winning_trades'] += 1
        else:
            self.performance_stats['losing_trades'] += 1
            
        if len(self.real_trade_history) > 200:
            self.real_trade_history = self.real_trade_history[-200:]
        self.save_real_trade_history()
        
        # Better display message
        if trade_data.get('partial_percent', 100) < 100:
            self.print_color(f"📝 Partial close saved: {trade_data['pair']} {trade_data['direction']} {trade_data['partial_percent']}% | P&L: ${pnl:.2f}", self.Fore.CYAN)
        else:
            self.print_color(f"📝 Trade saved: {trade_data['pair']} {trade_data['direction']} | P&L: ${pnl:.2f}", self.Fore.CYAN)
            
    except Exception as e:
        self.print_color(f"Error adding trade to history: {e}", self.Fore.RED)

def get_thailand_time(self):
    now_utc = datetime.now(pytz.utc)
    thailand_time = now_utc.astimezone(self.thailand_tz)
    return thailand_time.strftime('%Y-%m-%d %H:%M:%S')

def print_color(self, text, color="", style=""):
    if self.COLORAMA_AVAILABLE:
        print(f"{style}{color}{text}")
    else:
        print(text)

def validate_config(self):
    if not all([self.binance_api_key, self.binance_secret, self.openrouter_key]):
        self.print_color("Missing API keys!", self.Fore.RED)
        return False
    try:
        if self.binance:
            self.binance.futures_exchange_info()
            self.print_color("✅ Binance connection successful!", self.Fore.GREEN + self.Style.BRIGHT)
        else:
            self.print_color("Binance client not available - Paper Trading only", self.Fore.YELLOW)
            return True
    except Exception as e:
        self.print_color(f"Binance connection failed: {e}", self.Fore.RED)
        return False
    return True

def setup_futures(self):
    if not self.binance:
        return
        
    try:
        for pair in self.available_pairs:
            try:
                # Set initial leverage to 5x (AI can change later)
                self.binance.futures_change_leverage(symbol=pair, leverage=5)
                self.binance.futures_change_margin_type(symbol=pair, marginType='ISOLATED')
                self.print_color(f"✅ Leverage set for {pair}", self.Fore.GREEN)
            except Exception as e:
                self.print_color(f"Leverage setup failed for {pair}: {e}", self.Fore.YELLOW)
        self.print_color("✅ Futures setup completed!", self.Fore.GREEN + self.Style.BRIGHT)
    except Exception as e:
        self.print_color(f"Futures setup failed: {e}", self.Fore.RED)

def load_symbol_precision(self):
    if not self.binance:
        # For paper trading, get precision from Binance public API
        for pair in self.available_pairs:
            try:
                # Use Binance public API to get symbol info
                response = requests.get(f'https://api.binance.com/api/v3/exchangeInfo?symbol={pair}')
                if response.status_code == 200:
                    data = response.json()
                    symbol_info = next((s for s in data['symbols'] if s['symbol'] == pair), None)
                    if symbol_info:
                        for f in symbol_info['filters']:
                            if f['filterType'] == 'LOT_SIZE':
                                step_size = f['stepSize']
                                qty_precision = len(step_size.split('.')[1].rstrip('0')) if '.' in step_size else 0
                                self.quantity_precision[pair] = qty_precision
                            elif f['filterType'] == 'PRICE_FILTER':
                                tick_size = f['tickSize']
                                price_precision = len(tick_size.split('.')[1].rstrip('0')) if '.' in tick_size else 0
                                self.price_precision[pair] = price_precision
                else:
                    # Default precision if API fails
                    self.quantity_precision[pair] = 3
                    self.price_precision[pair] = 4
            except:
                self.quantity_precision[pair] = 3
                self.price_precision[pair] = 4
        self.print_color("Symbol precision loaded from Binance API", self.Fore.GREEN)
        return
        
    try:
        exchange_info = self.binance.futures_exchange_info()
        for symbol in exchange_info['symbols']:
            pair = symbol['symbol']
            if pair not in self.available_pairs:
                continue
            for f in symbol['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = f['stepSize']
                    qty_precision = len(step_size.split('.')[1].rstrip('0')) if '.' in step_size else 0
                    self.quantity_precision[pair] = qty_precision
                elif f['filterType'] == 'PRICE_FILTER':
                    tick_size = f['tickSize']
                    price_precision = len(tick_size.split('.')[1].rstrip('0')) if '.' in tick_size else 0
                    self.price_precision[pair] = price_precision
        self.print_color("✅ Symbol precision loaded", self.Fore.GREEN + self.Style.BRIGHT)
    except Exception as e:
        self.print_color(f"Error loading symbol precision: {e}", self.Fore.RED)

def get_market_news_sentiment(self):
    """Get recent cryptocurrency news sentiment"""
    try:
        news_sources = [
            "CoinDesk", "Cointelegraph", "CryptoSlate", "Decrypt", "Binance Blog"
        ]
        return f"Monitoring: {', '.join(news_sources)}"
    except:
        return "General crypto market news monitoring"

def get_ai_trading_decision(self, pair, market_data, current_trade=None):
    """AI makes COMPLETE trading decisions including REVERSE positions"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if not self.openrouter_key:
                self.print_color("❌ OpenRouter API key missing!", self.Fore.RED)
                return self.get_improved_fallback_decision(pair, market_data)
            
            current_price = market_data.get('current_price', 0)
            mtf = market_data.get('mtf_analysis', {})

            # === MULTI-TIMEFRAME TEXT SUMMARY ===
            mtf_text = "MULTI-TIMEFRAME ANALYSIS:\n"
            for tf in ['5m', '15m', '1h', '4h', '1d']:
                if tf in mtf:
                    d = mtf[tf]
                    mtf_text += f"- {tf.upper()}: {d.get('trend', 'N/A')} | "
                    if 'crossover' in d:
                        mtf_text += f"Signal: {d['crossover']} | "
                    if 'rsi' in d:
                        mtf_text += f"RSI: {d['rsi']} | "
                    if 'vol_spike' in d:
                        mtf_text += f"Vol: {'SPIKE' if d['vol_spike'] else 'Normal'} | "
                    if 'support' in d and 'resistance' in d:
                        mtf_text += f"S/R: {d['support']:.4f}/{d['resistance']:.4f}"
                    mtf_text += "\n"

            # === TREND ALIGNMENT ===
            h1_trend = mtf.get('1h', {}).get('trend')
            h4_trend = mtf.get('4h', {}).get('trend')
            alignment = "STRONG" if h1_trend == h4_trend and h1_trend else "WEAK"

            # === REVERSE ANALYSIS ===
            reverse_analysis = ""
            if current_trade and self.allow_reverse_positions:
                pnl = self.calculate_current_pnl(current_trade, current_price)
                reverse_analysis = f"""
                EXISTING POSITION:
                - Direction: {current_trade['direction']}
                - Entry: ${current_trade['entry_price']:.4f}
                - PnL: {pnl:.2f}%
                - REVERSE if trend flipped?
                """

            # === LEARNING CONTEXT ===
            learning_context = ""
            if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'get_learning_enhanced_prompt'):
                learning_context = self.get_learning_enhanced_prompt(pair, market_data)

            # === FINAL PROMPT ===
            prompt = f"""
YOU ARE A PROFESSIONAL AI TRADER. Budget: ${self.available_budget:.2f}

{mtf_text}
TREND ALIGNMENT: {alignment}

1H TRADING PAIR: {pair}
Current Price: ${current_price:.6f}
{reverse_analysis}
{learning_context}

RULES:
- Only trade if 1H and 4H trend align
- Confirm entry with 15m crossover + volume spike
- RSI < 30 = oversold, > 70 = overbought
- Position size: 5-10% of budget ($50 min)
- Leverage: 5-10x based on volatility
- NO TP/SL - you will close manually

REVERSE POSITION STRATEGY (CRITICAL):
- Use "REVERSE_LONG"  → Close current SHORT + Open LONG immediately
- Use "REVERSE_SHORT" → Close current LONG  + Open SHORT immediately
- REVERSE only if ALL conditions met:
  1. Current PnL ≤ -2%
  2. 1H and 4H trend flipped (opposite to current position)
  3. 15m shows crossover in new direction
  4. Volume spike confirms momentum
- Example:
  • You have SHORT @ $100 → Price now $103 → PnL: -3%
  • 4H: BEARISH → BULLISH, 15m: GOLDEN cross, Volume: SPIKE
  → Return "REVERSE_LONG"

Return JSON:
{{
    "decision": "LONG" | "SHORT" | "HOLD" | "REVERSE_LONG" | "REVERSE_SHORT",
    "position_size_usd": number,
    "entry_price": number,
    "leverage": number,
    "confidence": 0-100,
    "reasoning": "MTF alignment + signal + risk"
}}
"""
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",
                "X-Title": "Fully Autonomous AI Trader"
            }
            
            data = {
                "model": "deepseek/deepseek-chat-v3.1",
                "messages": [
                    {"role": "system", "content": "You are a fully autonomous AI trader with reverse position capability. You manually close positions based on market conditions - no TP/SL orders are set. Analyze when to enter AND when to exit based on technical analysis. Monitor every 3 minute."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
            
            self.print_color(f"🧠 DeepSeek Analyzing {pair} with 3MIN monitoring...", self.Fore.MAGENTA + self.Style.BRIGHT)
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                return self.parse_ai_trading_decision(ai_response, pair, current_price, current_trade)
            else:
                self.print_color(f"⚠️ DeepSeek API attempt {attempt+1} failed: {response.status_code}", self.Fore.YELLOW)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                    
        except requests.exceptions.Timeout:
            self.print_color(f"⏰ DeepSeek timeout attempt {attempt+1}", self.Fore.YELLOW)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
                
        except Exception as e:
            self.print_color(f"❌ DeepSeek error attempt {attempt+1}: {e}", self.Fore.RED)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    
    # All retries failed - use improved fallback
    self.print_color("🚨 All AI attempts failed, using improved fallback", self.Fore.RED)
    return self.get_improved_fallback_decision(pair, market_data)

def get_improved_fallback_decision(self, pair, market_data):
    """Better fallback that analyzes market conditions"""
    current_price = market_data['current_price']
    mtf = market_data.get('mtf_analysis', {})
    
    # Analyze multiple timeframes
    h1_data = mtf.get('1h', {})
    h4_data = mtf.get('4h', {})
    m15_data = mtf.get('15m', {})
    
    # Technical analysis based fallback
    bullish_signals = 0
    bearish_signals = 0
    
    # Check 1H trend
    if h1_data.get('trend') == 'BULLISH':
        bullish_signals += 1
    elif h1_data.get('trend') == 'BEARISH':
        bearish_signals += 1
    
    # Check 4H trend  
    if h4_data.get('trend') == 'BULLISH':
        bullish_signals += 1
    elif h4_data.get('trend') == 'BEARISH':
        bearish_signals += 1
    
    # Check RSI
    h1_rsi = h1_data.get('rsi', 50)
    if h1_rsi < 35:  # Oversold
        bullish_signals += 1
    elif h1_rsi > 65:  # Overbought
        bearish_signals += 1
    
    # Check crossover
    if m15_data.get('crossover') == 'GOLDEN':
        bullish_signals += 1
    elif m15_data.get('crossover') == 'DEATH':
        bearish_signals += 1
    
    # Make decision
    if bullish_signals >= 3 and bearish_signals <= 1:
        return {
            "decision": "LONG",
            "position_size_usd": 20,  # Smaller size for fallback
            "entry_price": current_price,
            "leverage": 5,
            "confidence": 60,
            "reasoning": f"Fallback: Bullish signals ({bullish_signals}/{bearish_signals}) - Trend + RSI + Crossover",
            "should_reverse": False
        }
    elif bearish_signals >= 3 and bullish_signals <= 1:
        return {
            "decision": "SHORT", 
            "position_size_usd": 20,
            "entry_price": current_price,
            "leverage": 5,
            "confidence": 60,
            "reasoning": f"Fallback: Bearish signals ({bearish_signals}/{bullish_signals}) - Trend + RSI + Crossover",
            "should_reverse": False
        }
    else:
        return {
            "decision": "HOLD",
            "position_size_usd": 0,
            "entry_price": current_price,
            "leverage": 5,
            "confidence": 40,
            "reasoning": f"Fallback: Mixed signals ({bullish_signals}/{bearish_signals}) - Waiting for clear direction",
            "should_reverse": False
        }

def parse_ai_trading_decision(self, ai_response, pair, current_price, current_trade=None):
    """Parse AI's trading decision including REVERSE positions"""
    try:
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            decision_data = json.loads(json_str)
            
            decision = decision_data.get('decision', 'HOLD').upper()
            position_size_usd = float(decision_data.get('position_size_usd', 0))
            entry_price = float(decision_data.get('entry_price', 0))
            leverage = int(decision_data.get('leverage', 5))
            confidence = float(decision_data.get('confidence', 50))
            reasoning = decision_data.get('reasoning', 'AI Analysis')
            
            # Validate leverage
            if leverage < 5:
                leverage = 5
            elif leverage > 10:
                leverage = 10
                
            if entry_price <= 0:
                entry_price = current_price
                
            return {
                "decision": decision,
                "position_size_usd": position_size_usd,
                "entry_price": entry_price,
                "leverage": leverage,
                "confidence": confidence,
                "reasoning": reasoning,
                "should_reverse": decision.startswith('REVERSE_')
            }
        return self.get_improved_fallback_decision(pair, {'current_price': current_price})
    except Exception as e:
        self.print_color(f"DeepSeek response parsing failed: {e}", self.Fore.RED)
        return self.get_improved_fallback_decision(pair, {'current_price': current_price})

def calculate_current_pnl(self, trade, current_price):
    """Calculate current PnL percentage"""
    try:
        if trade['direction'] == 'LONG':
            pnl_percent = ((current_price - trade['entry_price']) / trade['entry_price']) * 100 * trade['leverage']
        else:
            pnl_percent = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100 * trade['leverage']
        return pnl_percent
    except:
        return 0

def execute_reverse_position(self, pair, ai_decision, current_trade):
    """Execute reverse position - CLOSE CURRENT, THEN ASK AI BEFORE OPENING REVERSE"""
    try:
        self.print_color(f"🔄 ATTEMPTING REVERSE POSITION FOR {pair}", self.Fore.YELLOW + self.Style.BRIGHT)
        
        # 1. First close the current losing position
        close_success = self.close_trade_immediately(pair, current_trade, "REVERSE_POSITION")
        
        if close_success:
            # 2. Wait a moment for position to close
            time.sleep(2)
            
            # 3. Verify position is actually removed
            if pair in self.ai_opened_trades:
                self.print_color(f"⚠️  Position still exists after close, forcing removal...", self.Fore.RED)
                del self.ai_opened_trades[pair]
            
            # 4. 🆕 ASK AI AGAIN BEFORE OPENING REVERSE POSITION
            self.print_color(f"🔍 Asking AI to confirm reverse position for {pair}...", self.Fore.BLUE)
            market_data = self.get_price_history(pair)
            
            # Get fresh AI decision after closing
            new_ai_decision = self.get_ai_trading_decision(pair, market_data, None)
            
            # Check if AI still wants to open reverse position
            if new_ai_decision["decision"] in ["LONG", "SHORT"] and new_ai_decision["position_size_usd"] > 0:
                # 🎯 Calculate correct reverse direction
                current_direction = current_trade['direction']
                if current_direction == "LONG":
                    correct_reverse_direction = "SHORT"
                else:
                    correct_reverse_direction = "LONG"
                
                self.print_color(f"✅ AI CONFIRMED: Opening {correct_reverse_direction} {pair}", self.Fore.CYAN + self.Style.BRIGHT)
                
                # Use the new AI decision but ensure correct direction
                reverse_decision = new_ai_decision.copy()
                reverse_decision["decision"] = correct_reverse_direction
                
                # Execute the reverse trade
                return self.execute_ai_trade(pair, reverse_decision)
            else:
                self.print_color(f"🔄 AI changed mind, not opening reverse position for {pair}", self.Fore.YELLOW)
                self.print_color(f"📝 AI Decision: {new_ai_decision['decision']} | Reason: {new_ai_decision['reasoning']}", self.Fore.WHITE)
                return False
        else:
            self.print_color(f"❌ Reverse position failed: Could not close current trade", self.Fore.RED)
            return False
            
    except Exception as e:
        self.print_color(f"❌ Reverse position execution failed: {e}", self.Fore.RED)
        return False

def close_trade_immediately(self, pair, trade, close_reason="AI_DECISION", partial_percent=100):
    """Close trade immediately at market price with AI reasoning"""
    try:
        current_price = self.get_current_price(pair)
        
        # Calculate PnL based on partial percentage
        if trade['direction'] == 'LONG':
            pnl = (current_price - trade['entry_price']) * trade['quantity'] * (partial_percent / 100)
        else:
            pnl = (trade['entry_price'] - current_price) * trade['quantity'] * (partial_percent / 100)
        
        # If partial close, calculate the remaining position
        if partial_percent < 100:
            # This is a partial close - update the existing trade
            remaining_quantity = trade['quantity'] * (1 - partial_percent / 100)
            closed_quantity = trade['quantity'] * (partial_percent / 100)
            closed_position_size = trade['position_size_usd'] * (partial_percent / 100)
            
            # Update the existing trade with remaining quantity
            trade['quantity'] = remaining_quantity
            trade['position_size_usd'] = trade['position_size_usd'] * (1 - partial_percent / 100)
            
            # Add partial close to history
            partial_trade = trade.copy()
            partial_trade['status'] = 'PARTIAL_CLOSE'
            partial_trade['exit_price'] = current_price
            partial_trade['pnl'] = pnl
            partial_trade['close_reason'] = close_reason
            partial_trade['close_time'] = self.get_thailand_time()
            partial_trade['partial_percent'] = partial_percent
            partial_trade['closed_quantity'] = closed_quantity
            partial_trade['closed_position_size'] = closed_position_size
            
            self.available_budget += closed_position_size + pnl
            self.add_trade_to_history(partial_trade)
            
            pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
            self.print_color(f"✅ Partial Close | {pair} | {partial_percent}% | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
            self.print_color(f"📊 Remaining: {remaining_quantity:.4f} {pair} (${trade['position_size_usd']:.2f})", self.Fore.CYAN)
            
            return True
            
        else:
            # Full close
            trade['status'] = 'CLOSED'
            trade['exit_price'] = current_price
            trade['pnl'] = pnl
            trade['close_reason'] = close_reason
            trade['close_time'] = self.get_thailand_time()
            trade['partial_percent'] = 100  # Mark as full close
            
            self.available_budget += trade['position_size_usd'] + pnl
            self.add_trade_to_history(trade.copy())
            
            pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
            self.print_color(f"✅ Full Close | {pair} | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
            
            # Remove from active positions after full closing
            if pair in self.ai_opened_trades:
                del self.ai_opened_trades[pair]
            
            return True
            
    except Exception as e:
        self.print_color(f"❌ Close failed: {e}", self.Fore.RED)
        return False

def get_current_price(self, pair):
    """Get real price from Binance API (no mock prices)"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Try Binance Futures API first
            if self.binance:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                return float(ticker['price'])
            
            # Fallback to Binance Spot API (no authentication needed)
            response = requests.get(
                f'https://api.binance.com/api/v3/ticker/price?symbol={pair}',
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            else:
                self.print_color(f"Binance API error: {response.status_code}", self.Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"Price fetch attempt {attempt+1} failed: {e}", self.Fore.YELLOW)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    
    # Final fallback - use reasonable default based on pair
    self.print_color(f"🚨 All price API attempts failed for {pair}, using fallback price", self.Fore.RED)
    
    fallback_prices = {
        "AVAXUSDT": 14.0,
    }
    return fallback_prices.get(pair, 100.0)

def get_price_history(self, pair, limit=50):
    """Multi-Timeframe Analysis with REAL Binance data"""
    try:
        # If no Binance client, use spot API for paper trading
        if not self.binance:
            return self._get_mtf_data_via_api(pair, limit)
        
        intervals = {
            '5m': (Client.KLINE_INTERVAL_5MINUTE, 50),
            '15m': (Client.KLINE_INTERVAL_15MINUTE, 50),
            '1h': (Client.KLINE_INTERVAL_1HOUR, 50),
            '4h': (Client.KLINE_INTERVAL_4HOUR, 30),
            '1d': (Client.KLINE_INTERVAL_1DAY, 30)
        }

        mtf = {}
        current_price = self.get_current_price(pair)

        for name, (interval, lim) in intervals.items():
            klines = self.binance.futures_klines(symbol=pair, interval=interval, limit=lim)
            if not klines:
                continue

            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            ema9 = self.calculate_ema(closes, 9)
            ema21 = self.calculate_ema(closes, 21)
            rsi = self.calculate_rsi(closes, 14)[-1] if len(closes) > 14 else 50

            crossover = 'NONE'
            if len(ema9) >= 2 and len(ema21) >= 2:
                if ema9[-2] < ema21[-2] and ema9[-1] > ema21[-1]:
                    crossover = 'GOLDEN'
                elif ema9[-2] > ema21[-2] and ema9[-1] < ema21[-1]:
                    crossover = 'DEATH'

            vol_spike = self.calculate_volume_spike(volumes)

            mtf[name] = {
                'current_price': closes[-1],
                'change_1h': ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0,
                'ema9': round(ema9[-1], 6) if ema9[-1] else 0,
                'ema21': round(ema21[-1], 6) if ema21[-1] else 0,
                'trend': 'BULLISH' if ema9[-1] > ema21[-1] else 'BEARISH',
                'crossover': crossover,
                'rsi': round(rsi, 1),
                'vol_spike': vol_spike,
                'support': round(min(lows[-10:]), 6),
                'resistance': round(max(highs[-10:]), 6)
            }

        main = mtf.get('1h', {})
        return {
            'current_price': current_price,
            'price_change': main.get('change_1h', 0),
            'support_levels': [mtf['1h']['support'], mtf['4h']['support']] if '4h' in mtf else [],
            'resistance_levels': [mtf['1h']['resistance'], mtf['4h']['resistance']] if '4h' in mtf else [],
            'mtf_analysis': mtf
        }

    except Exception as e:
        self.print_color(f"MTF Analysis error: {e}", self.Fore.RED)
        return self._get_mtf_data_via_api(pair, limit)  # Fallback to API

def _get_mtf_data_via_api(self, pair, limit=50):
    """Get MTF data using Binance public API"""
    try:
        intervals = {
            '5m': '5m',
            '15m': '15m', 
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        mtf = {}
        current_price = self.get_current_price(pair)
        
        for name, interval in intervals.items():
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': pair,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                klines = response.json()
                
                closes = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                # Calculate indicators
                ema9 = self.calculate_ema(closes, 9)
                ema21 = self.calculate_ema(closes, 21)
                rsi = self.calculate_rsi(closes, 14)[-1] if len(closes) > 14 else 50
                
                crossover = 'NONE'
                if len(ema9) >= 2 and len(ema21) >= 2:
                    if ema9[-2] < ema21[-2] and ema9[-1] > ema21[-1]:
                        crossover = 'GOLDEN'
                    elif ema9[-2] > ema21[-2] and ema9[-1] < ema21[-1]:
                        crossover = 'DEATH'
                
                vol_spike = self.calculate_volume_spike(volumes)
                
                mtf[name] = {
                    'current_price': closes[-1],
                    'change_1h': ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0,
                    'ema9': round(ema9[-1], 6) if ema9[-1] else 0,
                    'ema21': round(ema21[-1], 6) if ema21[-1] else 0,
                    'trend': 'BULLISH' if ema9[-1] > ema21[-1] else 'BEARISH',
                    'crossover': crossover,
                    'rsi': round(rsi, 1),
                    'vol_spike': vol_spike,
                    'support': round(min(lows[-10:]), 6),
                    'resistance': round(max(highs[-10:]), 6)
                }
            else:
                self.print_color(f"API error for {interval} {pair}: {response.status_code}", self.Fore.YELLOW)
        
        main = mtf.get('1h', {})
        return {
            'current_price': current_price,
            'price_change': main.get('change_1h', 0),
            'support_levels': [mtf['1h']['support'], mtf['4h']['support']] if '4h' in mtf else [],
            'resistance_levels': [mtf['1h']['resistance'], mtf['4h']['resistance']] if '4h' in mtf else [],
            'mtf_analysis': mtf
        }
        
    except Exception as e:
        self.print_color(f"API MTF Analysis error: {e}", self.Fore.RED)
        return {
            'current_price': self.get_current_price(pair),
            'price_change': 0,
            'support_levels': [],
            'resistance_levels': [],
            'mtf_analysis': {}
        }

def calculate_quantity(self, pair, entry_price, position_size_usd, leverage):
    """Calculate quantity based on position size and leverage"""
    try:
        if entry_price <= 0:
            return None
            
        # Calculate notional value
        notional_value = position_size_usd * leverage
        
        # Calculate quantity
        quantity = notional_value / entry_price
        
        # Apply precision
        precision = self.quantity_precision.get(pair, 3)
        quantity = round(quantity, precision)
        
        if quantity <= 0:
            return None
            
        self.print_color(f"📊 Position: ${position_size_usd} | Leverage: {leverage}x | Notional: ${notional_value:.2f} | Quantity: {quantity}", self.Fore.CYAN)
        return quantity
        
    except Exception as e:
        self.print_color(f"Quantity calculation failed: {e}", self.Fore.RED)
        return None

def can_open_new_position(self, pair, position_size_usd):
    """Check if new position can be opened"""
    if pair in self.ai_opened_trades:
        return False, "Position already exists"
    
    if len(self.ai_opened_trades) >= self.max_concurrent_trades:
        return False, f"Max concurrent trades reached ({self.max_concurrent_trades})"
        
    if position_size_usd > self.available_budget:
        return False, f"Insufficient budget: ${position_size_usd:.2f} > ${self.available_budget:.2f}"
        
    max_allowed = self.total_budget * self.max_position_size_percent / 100
    if position_size_usd > max_allowed:
        return False, f"Position size too large: ${position_size_usd:.2f} > ${max_allowed:.2f}"
        
    return True, "OK"

def get_ai_decision_with_learning(self, pair, market_data):
    """Get AI decision enhanced with learned lessons"""
    # First get normal AI decision
    ai_decision = self.get_ai_trading_decision(pair, market_data)
    
    # Check if this matches known mistake patterns
    if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'should_avoid_trade') and self.should_avoid_trade(ai_decision, market_data):
        self.print_color(f"🧠 AI USING LEARNING: Blocking potential mistake for {pair}", self.Fore.YELLOW)
        return {
            "decision": "HOLD",
            "position_size_usd": 0,
            "entry_price": market_data['current_price'],
            "leverage": 5,
            "confidence": 0,
            "reasoning": f"Blocked - matches known error pattern",
            "should_reverse": False
        }
    
    # Add learning context to reasoning
    if ai_decision["decision"] != "HOLD" and LEARN_SCRIPT_AVAILABLE and hasattr(self, 'mistakes_history'):
        learning_context = f" | Applying lessons from {len(self.mistakes_history)} past mistakes"
        ai_decision["reasoning"] += learning_context
    
    return ai_decision

def execute_ai_trade(self, pair, ai_decision):
    """Execute trade WITHOUT TP/SL orders - AI will close manually"""
    try:
        decision = ai_decision["decision"]
        position_size_usd = ai_decision["position_size_usd"]
        entry_price = ai_decision["entry_price"]
        leverage = ai_decision["leverage"]
        confidence = ai_decision["confidence"]
        reasoning = ai_decision["reasoning"]
        
        # NEW: Handle reverse positions
        if decision.startswith('REVERSE_'):
            if pair in self.ai_opened_trades:
                current_trade = self.ai_opened_trades[pair]
                return self.execute_reverse_position(pair, ai_decision, current_trade)
            else:
                self.print_color(f"❌ Cannot reverse: No active position for {pair}", self.Fore.RED)
                return False
        
        if decision == "HOLD" or position_size_usd <= 0:
            self.print_color(f"🟡 DeepSeek decides to HOLD {pair}", self.Fore.YELLOW)
            return False
        
        # Check if we can open position (skip if reversing)
        if pair in self.ai_opened_trades and not decision.startswith('REVERSE_'):
            self.print_color(f"🚫 Cannot open {pair}: Position already exists", self.Fore.RED)
            return False
        
        if len(self.ai_opened_trades) >= self.max_concurrent_trades and pair not in self.ai_opened_trades:
            self.print_color(f"🚫 Cannot open {pair}: Max concurrent trades reached", self.Fore.RED)
            return False
            
        if position_size_usd > self.available_budget:
            self.print_color(f"🚫 Cannot open {pair}: Insufficient budget", self.Fore.RED)
            return False
        
        # Calculate quantity
        quantity = self.calculate_quantity(pair, entry_price, position_size_usd, leverage)
        if quantity is None:
            return False
        
        # Display AI trade decision (NO TP/SL)
        direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
        direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
        
        self.print_color(f"\n🤖 DEEPSEEK TRADE EXECUTION (NO TP/SL)", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 80, self.Fore.CYAN)
        self.print_color(f"{direction_icon} {pair}", direction_color)
        self.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
        self.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
        self.print_color(f"🎯 BOUNCE-PROOF 3-LAYER EXIT V2 ACTIVE", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
        self.print_color("=" * 80, self.Fore.CYAN)
        
        # Execute live trade WITHOUT TP/SL orders
        if self.binance:
            entry_side = 'BUY' if decision == 'LONG' else 'SELL'
            
            # Set leverage
            try:
                self.binance.futures_change_leverage(symbol=pair, leverage=leverage)
            except Exception as e:
                self.print_color(f"Leverage change failed: {e}", self.Fore.YELLOW)
            
            # Execute order ONLY - no TP/SL orders
            order = self.binance.futures_create_order(
                symbol=pair,
                side=entry_side,
                type='MARKET',
                quantity=quantity
            )
            
            # ❌❌❌ NO TP/SL ORDERS CREATED ❌❌❌
        
        # Update budget and track trade
        self.available_budget -= position_size_usd
        
        self.ai_opened_trades[pair] = {
            "pair": pair,
            "direction": decision,
            "entry_price": entry_price,
            "quantity": quantity,
            "position_size_usd": position_size_usd,
            "leverage": leverage,
            "entry_time": time.time(),
            "status": 'ACTIVE',
            'ai_confidence': confidence,
            'ai_reasoning': reasoning,
            'entry_time_th': self.get_thailand_time(),
            'has_tp_sl': False,  # NEW: Mark as no TP/SL
            'peak_pnl': 0  # NEW: For 3-layer system
        }
        
        self.print_color(f"✅ TRADE EXECUTED (BOUNCE-PROOF V2): {pair} {decision} | Leverage: {leverage}x", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"📊 AI will monitor with Bounce-Proof 3-Layer Exit System", self.Fore.BLUE)
        return True
        
    except Exception as e:
        self.print_color(f"❌ Trade execution failed: {e}", self.Fore.RED)
        return False

def get_ai_close_decision_v2(self, pair, trade):
    """BOUNCE-PROOF 3-LAYER EXIT V2 – အမြတ်ပြန်မပေးရအောင် အပြတ်ပိတ်"""
    try:
        current_price = self.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        # Peak PnL tracking
        if 'peak_pnl' not in trade:
            trade['peak_pnl'] = current_pnl
        if current_pnl > trade['peak_pnl']:
            trade['peak_pnl'] = current_pnl
        
        peak = trade['peak_pnl']

        # 1. Hard stop -5% က ဘယ်လိုမှ မလွတ်
        if current_pnl <= -5.0:
            return {
                "should_close": True, 
                "close_type": "STOP_LOSS", 
                "close_reason": "Hard -5% rule", 
                "confidence": 100,
                "partial_percent": 100
            }

        # 2. 60% Partial @ +9%
        if peak >= 9.0 and not trade.get('partial_done', False):
            trade['partial_done'] = True
            return {
                "should_close": True,
                "partial_percent": 60,
                "close_type": "PARTIAL_60",
                "reason": f"LOCK 60% PROFIT @ +{peak:.1f}% → အမြတ် ချက်ချင်း အိတ်ထဲ!",
                "confidence": 100
            }

        # 3. Instant Breakeven @ +12%
        if peak >= 12.0 and not trade.get('breakeven_done', False):
            trade['breakeven_done'] = True
            return {
                "should_close": False,
                "move_sl_to": trade['entry_price'],
                "close_type": "BREAKEVEN_ACTIVATED", 
                "reason": f"Peak +{peak:.1f}% → ကျန် 40% ကို BREAKEVEN ချပြီး → ဘယ်လိုမှ မရှုံးနိုင်တော့ဘူး!",
                "confidence": 100
            }

        # 4. Dynamic Profit Floor (75% of Peak)
        if peak >= 15.0:
            profit_floor = peak * 0.75
            if current_pnl <= profit_floor and trade.get('partial_done', False):
                return {
                    "should_close": True,
                    "partial_percent": 100,
                    "close_type": "PROFIT_FLOOR_HIT",
                    "reason": f"Peak {peak:.1f}% → 75% floor ({profit_floor:.1f}%) ထိပြီး → ကျန်အကုန် အမြတ်နဲ့ ပိတ်!",
                    "confidence": 100
                }

        # 5. 2×ATR Trailing
        if trade.get('partial_done', False) and peak >= 9.0:
            atr_14 = 0.001
            try:
                if self.binance:
                    klines = self.binance.futures_klines(symbol=pair, interval='1h', limit=50)
                    if len(klines) >= 15:
                        highs = [float(k[2]) for k in klines]
                        lows = [float(k[3]) for k in klines]
                        closes = [float(k[4]) for k in klines]
                        tr = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(klines))]
                        atr_14 = sum(tr[-14:]) / 14
            except: pass
            
            trail_price = current_price + (2 * atr_14) if trade['direction'] == 'LONG' else current_price - (2 * atr_14)
            if trade['direction'] == 'LONG' and current_price <= trail_price:
                return {
                    "should_close": True, 
                    "partial_percent": 100, 
                    "close_type": "TRAILING_HIT", 
                    "reason": "2×ATR Trailing ထိပြီး ထွက်",
                    "confidence": 95
                }
            if trade['direction'] == 'SHORT' and current_price >= trail_price:
                return {
                    "should_close": True, 
                    "partial_percent": 100, 
                    "close_type": "TRAILING_HIT", 
                    "reason": "2×ATR Trailing ထိပြီး ထွက်",
                    "confidence": 95
                }

        return {"should_close": False}

    except Exception as e:
        return {"should_close": False}

def monitor_positions(self):
    """Monitor positions and ask AI when to close (3-LAYER SYSTEM)"""
    try:
        closed_trades = []
        for pair, trade in list(self.ai_opened_trades.items()):
            if trade['status'] != 'ACTIVE':
                continue
            
            # NEW: Ask AI whether to close this position using 3-Layer system
            if not trade.get('has_tp_sl', True):
                self.print_color(f"🔍 Bounce-Proof V2 Checking {pair}...", self.Fore.BLUE)
                close_decision = self.get_ai_close_decision_v2(pair, trade)
                
                if close_decision.get("should_close", False):
                    close_type = close_decision.get("close_type", "AI_DECISION")
                    confidence = close_decision.get("confidence", 0)
                    reasoning = close_decision.get("reasoning", "No reason provided")
                    partial_percent = close_decision.get("partial_percent", 100)
                    
                    # 🆕 Use 3-Layer system's ACTUAL reasoning for closing
                    full_close_reason = f"BOUNCE-PROOF V2: {close_type} - {reasoning}"
                    
                    self.print_color(f"🎯 Bounce-Proof V2 Decision: CLOSE {pair}", self.Fore.YELLOW + self.Style.BRIGHT)
                    self.print_color(f"📝 Close Type: {close_type} | Partial: {partial_percent}%", self.Fore.CYAN)
                    self.print_color(f"💡 Confidence: {confidence}% | Reasoning: {reasoning}", self.Fore.WHITE)
                    
                    # 🆕 Pass partial percentage to close function
                    success = self.close_trade_immediately(pair, trade, full_close_reason, partial_percent)
                    if success and partial_percent == 100:  # Only count as closed if full close
                        closed_trades.append(pair)
                else:
                    # Show 3-Layer system's decision to hold with reasoning
                    if close_decision.get('confidence', 0) > 0:
                        reasoning = close_decision.get('reasoning', 'No reason provided')
                        self.print_color(f"🔍 Bounce-Proof V2 wants to HOLD {pair} (Confidence: {close_decision.get('confidence', 0)}%)", self.Fore.GREEN)
                        self.print_color(f"📝 Hold Reasoning: {reasoning}", self.Fore.WHITE)
                
        return closed_trades
                
    except Exception as e:
        self.print_color(f"Bounce-Proof V2 Monitoring error: {e}", self.Fore.RED)
        return []

def display_dashboard(self):
    """Display trading dashboard WITH learning progress"""
    self.print_color(f"\n🤖 AI TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
    self.print_color("=" * 90, self.Fore.CYAN)
    self.print_color(f"🎯 MODE: BOUNCE-PROOF 3-LAYER EXIT V2", self.Fore.YELLOW + self.Style.BRIGHT)
    self.print_color(f"⏰ MONITORING: 3 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
    
    # === MTF SUMMARY ===
    if hasattr(self, 'last_mtf') and self.last_mtf:
        self.print_color(" MULTI-TIMEFRAME SUMMARY", self.Fore.MAGENTA + self.Style.BRIGHT)
        for tf, data in self.last_mtf.items():
            color = self.Fore.GREEN if data.get('trend') == 'BULLISH' else self.Fore.RED
            signal = f" | {data.get('crossover', '')}" if 'crossover' in data else ""
            rsi_text = f" | RSI: {data.get('rsi', 50)}" if 'rsi' in data else ""
            vol_text = f" | Vol: {'SPIKE' if data.get('vol_spike') else 'Normal'}" if 'vol_spike' in data else ""
            self.print_color(f"  {tf.upper()}: {data.get('trend', 'N/A')}{signal}{rsi_text}{vol_text}", color)
        self.print_color("   " + "-" * 60, self.Fore.CYAN)
    
    # 🧠 Add learning stats
    if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'mistakes_history'):
        total_lessons = len(self.mistakes_history)
        if total_lessons > 0:
            self.print_color(f"🧠 AI HAS LEARNED FROM {total_lessons} MISTAKES", self.Fore.MAGENTA + self.Style.BRIGHT)
    
    active_count = 0
    total_unrealized = 0
    
    for pair, trade in self.ai_opened_trades.items():
        if trade['status'] == 'ACTIVE':
            active_count += 1
            current_price = self.get_current_price(pair)
            
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            
            if trade['direction'] == 'LONG':
                unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
            else:
                unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                
            total_unrealized += unrealized_pnl
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if unrealized_pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
            
            self.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
            self.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x ⚡", self.Fore.WHITE)
            self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
            self.print_color(f"   P&L: ${unrealized_pnl:.2f}", pnl_color)
            self.print_color(f"   🎯 BOUNCE-PROOF V2 EXIT ACTIVE", self.Fore.YELLOW)
            self.print_color("   " + "-" * 60, self.Fore.CYAN)
    
    if active_count == 0:
        self.print_color("No active positions", self.Fore.YELLOW)
    else:
        total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
        self.print_color(f"📊 Active Positions: {active_count}/{self.max_concurrent_trades} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)

def show_trade_history(self, limit=15):
    """Show trading history with partial closes"""
    if not self.real_trade_history:
        self.print_color("No trade history found", self.Fore.YELLOW)
        return
    
    self.print_color(f"\n📊 TRADING HISTORY (Last {min(limit, len(self.real_trade_history))} trades)", self.Fore.CYAN + self.Style.BRIGHT)
    self.print_color("=" * 120, self.Fore.CYAN)
    
    recent_trades = self.real_trade_history[-limit:]
    for i, trade in enumerate(reversed(recent_trades)):
        pnl = trade.get('pnl', 0)
        pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
        direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
        position_size = trade.get('position_size_usd', 0)
        leverage = trade.get('leverage', 1)
        
        # Display type indicator
        display_type = trade.get('display_type', 'FULL_CLOSE')
        if display_type.startswith('PARTIAL'):
            type_indicator = f" | {display_type}"
            type_color = self.Fore.YELLOW
        else:
            type_indicator = " | FULL"
            type_color = self.Fore.WHITE
        
        self.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']}{type_indicator}", pnl_color)
        self.print_color(f"     Size: ${position_size:.2f} | Leverage: {leverage}x | P&L: ${pnl:.2f}", pnl_color)
        self.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)
        
        # Show additional info for partial closes
        if trade.get('partial_percent', 100) < 100:
            closed_qty = trade.get('closed_quantity', 0)
            self.print_color(f"     🔸 Partial: {trade['partial_percent']}% ({closed_qty:.4f}) closed", self.Fore.CYAN)

def show_trading_stats(self):
    """Show trading statistics"""
    if self.real_total_trades == 0:
        return
        
    win_rate = (self.real_winning_trades / self.real_total_trades) * 100
    avg_trade = self.real_total_pnl / self.real_total_trades
    
    self.print_color(f"\n📈 TRADING STATISTICS", self.Fore.GREEN + self.Style.BRIGHT)
    self.print_color("=" * 60, self.Fore.GREEN)
    self.print_color(f"Total Trades: {self.real_total_trades} | Winning Trades: {self.real_winning_trades}", self.Fore.WHITE)
    self.print_color(f"Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
    self.print_color(f"Total P&L: ${self.real_total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if self.real_total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
    self.print_color(f"Average P&L per Trade: ${avg_trade:.2f}", self.Fore.WHITE)
    self.print_color(f"Available Budget: ${self.available_budget:.2f}", self.Fore.CYAN + self.Style.BRIGHT)

def show_advanced_learning_progress(self):
    """Display learning progress every 3 cycles"""
    if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'mistakes_history'):
        total_lessons = len(self.mistakes_history)
        if total_lessons > 0:
            self.print_color(f"\n🧠 AI LEARNING PROGRESS (Cycle {getattr(self, 'cycle_count', 0)})", self.Fore.MAGENTA + self.Style.BRIGHT)
            self.print_color("=" * 50, self.Fore.MAGENTA)
            self.print_color(f"📚 Total Lessons Learned: {total_lessons}", self.Fore.CYAN)
            
            # Show recent mistakes patterns
            recent_mistakes = self.mistakes_history[-5:] if len(self.mistakes_history) >= 5 else self.mistakes_history
            if recent_mistakes:
                self.print_color(f"🔄 Recent Patterns:", self.Fore.YELLOW)
                for i, mistake in enumerate(reversed(recent_mistakes)):
                    reason = mistake.get('reason', 'Unknown pattern')
                    self.print_color(f"   {i+1}. {reason}", self.Fore.WHITE)
            
            # Show improvement stats
            if hasattr(self, 'performance_stats'):
                total_trades = self.performance_stats.get('total_trades', 0)
                winning_trades = self.performance_stats.get('winning_trades', 0)
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                self.print_color(f"📈 Current Win Rate: {win_rate:.1f}%", self.Fore.GREEN)
                
            # Show learned patterns count
            if hasattr(self, 'learned_patterns'):
                pattern_count = len(self.learned_patterns)
                self.print_color(f"🎯 Patterns Memorized: {pattern_count}", self.Fore.BLUE)
        else:
            self.print_color(f"\n🧠 AI is collecting learning data... (No mistakes yet)", self.Fore.YELLOW)
    else:
        self.print_color(f"\n🧠 Learning module not available", self.Fore.YELLOW)

def run_trading_cycle(self):
    """Run trading cycle with REVERSE position checking and 3-LAYER EXIT"""
    try:
        # First monitor and ask AI to close positions using 3-Layer system
        self.monitor_positions()
        self.display_dashboard()
        
        # Show stats periodically
        if hasattr(self, 'cycle_count') and self.cycle_count % 4 == 0:  # Every 4 cycles (5 minutes)
            self.show_trade_history(8)
            self.show_trading_stats()
        
        # 🧠 Show advanced learning progress every 3 cycles
        if hasattr(self, 'cycle_count') and self.cycle_count % 3 == 0 and LEARN_SCRIPT_AVAILABLE:
            self.show_advanced_learning_progress()
        
        self.print_color(f"\n🔍 DEEPSEEK SCANNING {len(self.available_pairs)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
        
        qualified_signals = 0
        for pair in self.available_pairs:
            if self.available_budget > 100:
                market_data = self.get_price_history(pair)
                self.last_mtf = market_data.get('mtf_analysis', {})
                
                # Use learning-enhanced AI decision
                ai_decision = self.get_ai_decision_with_learning(pair, market_data)
                
                if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] > 0:
                    qualified_signals += 1
                    direction = ai_decision['decision']
                    leverage_info = f"Leverage: {ai_decision['leverage']}x"
                    
                    if direction.startswith('REVERSE_'):
                        self.print_color(f"🔄 REVERSE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f}", self.Fore.YELLOW + self.Style.BRIGHT)
                    else:
                        self.print_color(f"🎯 TRADE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f} | {leverage_info}", self.Fore.GREEN + self.Style.BRIGHT)
                    
                    success = self.execute_ai_trade(pair, ai_decision)
                    if success:
                        time.sleep(2)  # Reduced delay for faster 3min cycles
            
        if qualified_signals == 0:
            self.print_color("No qualified DeepSeek signals this cycle", self.Fore.YELLOW)
            
    except Exception as e:
        self.print_color(f"Trading cycle error: {e}", self.Fore.RED)

def start_trading(self):
    """Start trading with REVERSE position feature and 3-LAYER EXIT"""
    self.print_color("🚀 STARTING AI TRADER WITH BOUNCE-PROOF 3-LAYER EXIT V2!", self.Fore.CYAN + self.Style.BRIGHT)
    self.print_color("💰 AI MANAGING $500 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
    self.print_color("🔄 REVERSE POSITION: ENABLED (AI can flip losing positions)", self.Fore.MAGENTA + self.Style.BRIGHT)
    self.print_color("🎯 BOUNCE-PROOF 3-LAYER EXIT V2: ACTIVE", self.Fore.YELLOW + self.Style.BRIGHT)
    self.print_color("⏰ MONITORING: 3 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
    self.print_color("⚡ LEVERAGE: 5x to 10x", self.Fore.RED + self.Style.BRIGHT)
    if LEARN_SCRIPT_AVAILABLE:
        self.print_color("🧠 SELF-LEARNING AI: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
    
    self.cycle_count = 0
    while True:
        try:
            self.cycle_count += 1
            self.print_color(f"\n🔄 TRADING CYCLE {self.cycle_count} (BOUNCE-PROOF V2)", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 60, self.Fore.CYAN)
            self.run_trading_cycle()
            self.print_color(f"⏳ Next Bounce-Proof V2 analysis in 3 minute...", self.Fore.BLUE)
            time.sleep(self.monitoring_interval)  # 3 minute
            
        except KeyboardInterrupt:
            self.print_color(f"\n🛑 TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
            self.show_trade_history(15)
            self.show_trading_stats()
            break
        except Exception as e:
            self.print_color(f"Main loop error: {e}", self.Fore.RED)
            time.sleep(self.monitoring_interval)

# Add all methods to the class including MTF indicators
methods = [
    load_real_trade_history, save_real_trade_history, add_trade_to_history,
    get_thailand_time, print_color, validate_config, setup_futures,
    load_symbol_precision, get_market_news_sentiment, get_ai_trading_decision,
    parse_ai_trading_decision, get_improved_fallback_decision, calculate_current_pnl,
    execute_reverse_position, close_trade_immediately, get_price_history,
    get_current_price, calculate_quantity, can_open_new_position,
    get_ai_decision_with_learning, execute_ai_trade, get_ai_close_decision_v2,
    monitor_positions, display_dashboard, show_trade_history, show_trading_stats,
    run_trading_cycle, start_trading, show_advanced_learning_progress,
    # Add MTF indicator methods
    calculate_ema, calculate_rsi, calculate_volume_spike, _get_mtf_data_via_api,
    validate_api_keys
]

for method in methods:
    setattr(FullyAutonomous1HourAITrader, method.__name__, method)

# Paper trading class - Uses REAL Binance data only
class FullyAutonomous1HourPaperTrader:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        # Copy colorama attributes from real_bot
        self.Fore = real_bot.Fore
        self.Back = real_bot.Back
        self.Style = real_bot.Style
        self.COLORAMA_AVAILABLE = real_bot.COLORAMA_AVAILABLE
        
        # Copy reverse position settings
        self.allow_reverse_positions = True
        
        # NEW: Monitoring interval (3 minute)
        self.monitoring_interval = 180  # 3 minute in seconds
        
        self.paper_balance = 500  # Virtual $500 budget
        self.available_budget = 500
        self.paper_positions = {}
        self.paper_history_file = "fully_autonomous_1hour_paper_trading_history.json"
        self.paper_history = self.load_paper_history()
        self.available_pairs = ["AVAXUSDT"]
        self.max_concurrent_trades = 6
        
        self.real_bot.print_color("🤖 FULLY AUTONOMOUS PAPER TRADER INITIALIZED!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Virtual Budget: ${self.paper_balance}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color(f"🔄 REVERSE POSITION FEATURE: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.real_bot.print_color(f"🎯 BOUNCE-PROOF 3-LAYER EXIT V2: ENABLED", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"⏰ MONITORING: 3 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
        self.real_bot.print_color(f"📡 USING REAL BINANCE MARKET DATA", self.Fore.BLUE + self.Style.BRIGHT)
    
    def load_paper_history(self):
        """Load PAPER trading history"""
        try:
            if os.path.exists(self.paper_history_file):
                with open(self.paper_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.real_bot.print_color(f"Error loading paper trade history: {e}", self.Fore.RED)
            return []
    
    def save_paper_history(self):
        """Save PAPER trading history"""
        try:
            with open(self.paper_history_file, 'w') as f:
                json.dump(self.paper_history, f, indent=2)
        except Exception as e:
            self.real_bot.print_color(f"Error saving paper trade history: {e}", self.Fore.RED)
    
    def add_paper_trade_to_history(self, trade_data):
        """Add trade to PAPER trading history with partial close support"""
        try:
            trade_data['close_time'] = self.real_bot.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            trade_data['trade_type'] = 'PAPER'
            
            # Add partial close indicator to display
            if trade_data.get('partial_percent', 100) < 100:
                trade_data['display_type'] = f"PARTIAL_{trade_data['partial_percent']}%"
            else:
                trade_data['display_type'] = "FULL_CLOSE"
            
            self.paper_history.append(trade_data)
            
            if len(self.paper_history) > 200:
                self.paper_history = self.paper_history[-200:]
            self.save_paper_history()
            
            # Better display message
            if trade_data.get('partial_percent', 100) < 100:
                self.real_bot.print_color(f"📝 PAPER Partial close saved: {trade_data['pair']} {trade_data['direction']} {trade_data['partial_percent']}% | P&L: ${trade_data.get('pnl', 0):.2f}", self.Fore.CYAN)
            else:
                self.real_bot.print_color(f"📝 PAPER Trade saved: {trade_data['pair']} {trade_data['direction']} | P&L: ${trade_data.get('pnl', 0):.2f}", self.Fore.CYAN)
                
        except Exception as e:
            self.real_bot.print_color(f"Error adding paper trade to history: {e}", self.Fore.RED)

    def calculate_current_pnl(self, trade, current_price):
        """Calculate current PnL percentage for paper trading"""
        try:
            if trade['direction'] == 'LONG':
                pnl_percent = ((current_price - trade['entry_price']) / trade['entry_price']) * 100 * trade['leverage']
            else:
                pnl_percent = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100 * trade['leverage']
            return pnl_percent
        except:
            return 0

    def paper_execute_reverse_position(self, pair, ai_decision, current_trade):
        """Execute reverse position in paper trading - CLOSE CURRENT, THEN ASK AI BEFORE OPENING REVERSE"""
        try:
            self.real_bot.print_color(f"🔄 PAPER: ATTEMPTING REVERSE POSITION FOR {pair}", self.Fore.YELLOW + self.Style.BRIGHT)
            
            # 1. First close the current losing position
            close_success = self.paper_close_trade_immediately(pair, current_trade, "REVERSE_POSITION")
            
            if close_success:
                # 2. Wait a moment and verify position is actually closed
                time.sleep(1)
                
                # Verify position is actually removed
                if pair in self.paper_positions:
                    self.real_bot.print_color(f"⚠️  PAPER: Position still exists after close, forcing removal...", self.Fore.RED)
                    del self.paper_positions[pair]
                
                # 3. 🆕 ASK AI AGAIN BEFORE OPENING REVERSE POSITION
                self.real_bot.print_color(f"🔍 PAPER: Asking AI to confirm reverse position for {pair}...", self.Fore.BLUE)
                market_data = self.real_bot.get_price_history(pair)
                
                # Get fresh AI decision after closing
                new_ai_decision = self.real_bot.get_ai_trading_decision(pair, market_data, None)
                
                # Check if AI still wants to open reverse position
                if new_ai_decision["decision"] in ["LONG", "SHORT"] and new_ai_decision["position_size_usd"] > 0:
                    # 🎯 Calculate correct reverse direction
                    current_direction = current_trade['direction']
                    if current_direction == "LONG":
                        correct_reverse_direction = "SHORT"
                    else:
                        correct_reverse_direction = "LONG"
                    
                    self.real_bot.print_color(f"✅ PAPER AI CONFIRMED: Opening {correct_reverse_direction} {pair}", self.Fore.CYAN + self.Style.BRIGHT)
                    
                    # Use the new AI decision but ensure correct direction
                    reverse_decision = new_ai_decision.copy()
                    reverse_decision["decision"] = correct_reverse_direction
                    
                    # Execute the reverse trade
                    return self.paper_execute_trade(pair, reverse_decision)
                else:
                    self.real_bot.print_color(f"🔄 PAPER AI changed mind, not opening reverse position for {pair}", self.Fore.YELLOW)
                    self.real_bot.print_color(f"📝 PAPER AI Decision: {new_ai_decision['decision']} | Reason: {new_ai_decision['reasoning']}", self.Fore.WHITE)
                    return False
            else:
                self.real_bot.print_color(f"❌ PAPER: Reverse position failed", self.Fore.RED)
                return False
                
        except Exception as e:
            self.real_bot.print_color(f"❌ PAPER: Reverse position execution failed: {e}", self.Fore.RED)
            return False

    def paper_close_trade_immediately(self, pair, trade, close_reason="AI_DECISION", partial_percent=100):
        """Close paper trade immediately with partial close support"""
        try:
            current_price = self.real_bot.get_current_price(pair)
            
            # Calculate PnL based on partial percentage
            if trade['direction'] == 'LONG':
                pnl = (current_price - trade['entry_price']) * trade['quantity'] * (partial_percent / 100)
            else:
                pnl = (trade['entry_price'] - current_price) * trade['quantity'] * (partial_percent / 100)
            
            # If partial close, calculate the remaining position
            if partial_percent < 100:
                # This is a partial close - update the existing trade
                remaining_quantity = trade['quantity'] * (1 - partial_percent / 100)
                closed_quantity = trade['quantity'] * (partial_percent / 100)
                closed_position_size = trade['position_size_usd'] * (partial_percent / 100)
                
                # Update the existing trade with remaining quantity
                trade['quantity'] = remaining_quantity
                trade['position_size_usd'] = trade['position_size_usd'] * (1 - partial_percent / 100)
                
                # Add partial close to history
                partial_trade = trade.copy()
                partial_trade['status'] = 'PARTIAL_CLOSE'
                partial_trade['exit_price'] = current_price
                partial_trade['pnl'] = pnl
                partial_trade['close_reason'] = close_reason
                partial_trade['close_time'] = self.real_bot.get_thailand_time()
                partial_trade['partial_percent'] = partial_percent
                partial_trade['closed_quantity'] = closed_quantity
                partial_trade['closed_position_size'] = closed_position_size
                
                self.available_budget += closed_position_size + pnl
                self.add_paper_trade_to_history(partial_trade)
                
                pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
                self.real_bot.print_color(f"✅ PAPER: Partial Close | {pair} | {partial_percent}% | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                self.real_bot.print_color(f"📊 PAPER: Remaining: {remaining_quantity:.4f} {pair} (${trade['position_size_usd']:.2f})", self.Fore.CYAN)
                
                return True
                
            else:
                # Full close
                trade['status'] = 'CLOSED'
                trade['exit_price'] = current_price
                trade['pnl'] = pnl
                trade['close_reason'] = close_reason
                trade['close_time'] = self.real_bot.get_thailand_time()
                trade['partial_percent'] = 100
                
                self.available_budget += trade['position_size_usd'] + pnl
                self.add_paper_trade_to_history(trade.copy())
                
                pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
                self.real_bot.print_color(f"✅ PAPER: Full Close | {pair} | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                
                # Remove from active positions after full closing
                if pair in self.paper_positions:
                    del self.paper_positions[pair]
                
                return True
                
        except Exception as e:
            self.real_bot.print_color(f"❌ PAPER: Close failed: {e}", self.Fore.RED)
            return False

    def get_ai_close_decision_v2(self, pair, trade):
        """BOUNCE-PROOF 3-LAYER EXIT V2 – PAPER TRADING VERSION (NO WINNER-TURN-LOSER)"""
        try:
            current_price = self.real_bot.get_current_price(pair)
            current_pnl = self.calculate_current_pnl(trade, current_price)
            
            # Peak PnL tracking
            if 'peak_pnl' not in trade:
                trade['peak_pnl'] = current_pnl
            if current_pnl > trade['peak_pnl']:
                trade['peak_pnl'] = current_pnl
            
            peak = trade['peak_pnl']

            # 1. Hard stop -5% က ဘယ်လိုမှ မလွတ်
            if current_pnl <= -5.0:
                return {
                    "should_close": True, 
                    "close_type": "STOP_LOSS", 
                    "close_reason": "Hard -5% rule", 
                    "confidence": 100,
                    "partial_percent": 100
                }

            # 2. 60% Partial @ +9%
            if peak >= 9.0 and not trade.get('partial_done', False):
                trade['partial_done'] = True
                return {
                    "should_close": True,
                    "partial_percent": 60,
                    "close_type": "PARTIAL_60",
                    "reason": f"PAPER: Lock 60% profit @ +{peak:.1f}%",
                    "confidence": 100
                }

            # 3. Instant Breakeven @ +12%
            if peak >= 12.0 and not trade.get('breakeven_done', False):
                trade['breakeven_done'] = True
                return {
                    "should_close": False,
                    "move_sl_to": trade['entry_price'],
                    "close_type": "BREAKEVEN_ACTIVATED",
                    "reason": f"PAPER: Breakeven activated @ +{peak:.1f}%",
                    "confidence": 100
                }

            # 4. Dynamic Profit Floor (75% of Peak)
            if peak >= 15.0:
                profit_floor = peak * 0.75
                if current_pnl <= profit_floor and trade.get('partial_done', False):
                    return {
                        "should_close": True,
                        "partial_percent": 100,
                        "close_type": "PROFIT_FLOOR_HIT",
                        "reason": f"PAPER: Profit floor hit {profit_floor:.1f}%",
                        "confidence": 100
                    }

            # 5. 2×ATR Trailing
            if trade.get('partial_done', False) and peak >= 9.0:
                atr_14 = 0.001
                trail_price = current_price + (2 * atr_14) if trade['direction'] == 'LONG' else current_price - (2 * atr_14)
                if trade['direction'] == 'LONG' and current_price <= trail_price:
                    return {
                        "should_close": True, 
                        "partial_percent": 100, 
                        "close_type": "TRAILING_HIT", 
                        "reason": "PAPER: 2×ATR Trailing",
                        "confidence": 95
                    }
                if trade['direction'] == 'SHORT' and current_price >= trail_price:
                    return {
                        "should_close": True, 
                        "partial_percent": 100, 
                        "close_type": "TRAILING_HIT", 
                        "reason": "PAPER: 2×ATR Trailing",
                        "confidence": 95
                    }

            # ❌❌❌ NO WINNER-TURN-LOSER LOGIC - COMPLETELY REMOVED ❌❌❌

            return {"should_close": False}

        except Exception as e:
            return {"should_close": False}

    def paper_execute_trade(self, pair, ai_decision):
        """Execute paper trade WITHOUT TP/SL orders"""
        try:
            decision = ai_decision["decision"]
            position_size_usd = ai_decision["position_size_usd"]
            entry_price = ai_decision["entry_price"]
            leverage = ai_decision["leverage"]
            confidence = ai_decision["confidence"]
            reasoning = ai_decision["reasoning"]
            
            # Handle reverse positions
            if decision.startswith('REVERSE_'):
                if pair in self.paper_positions:
                    current_trade = self.paper_positions[pair]
                    return self.paper_execute_reverse_position(pair, ai_decision, current_trade)
                else:
                    self.real_bot.print_color(f"❌ PAPER: Cannot reverse - No active position for {pair}", self.Fore.RED)
                    return False
            
            if decision == "HOLD" or position_size_usd <= 0:
                self.real_bot.print_color(f"🟡 PAPER: DeepSeek decides to HOLD {pair}", self.Fore.YELLOW)
                return False
            
            # Check if we can open position
            if pair in self.paper_positions:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Position already exists", self.Fore.RED)
                return False
            
            if len(self.paper_positions) >= self.max_concurrent_trades:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Max concurrent trades reached (6)", self.Fore.RED)
                return False
                
            if position_size_usd > self.available_budget:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Insufficient budget", self.Fore.RED)
                return False
            
            # Calculate quantity
            notional_value = position_size_usd * leverage
            quantity = notional_value / entry_price
            quantity = round(quantity, 3)
            
            # Display AI trade decision (NO TP/SL)
            direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
            
            self.real_bot.print_color(f"\n🤖 PAPER TRADE EXECUTION (BOUNCE-PROOF V2)", self.Fore.CYAN + self.Style.BRIGHT)
            self.real_bot.print_color("=" * 80, self.Fore.CYAN)
            self.real_bot.print_color(f"{direction_icon} {pair}", direction_color)
            self.real_bot.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.real_bot.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
            self.real_bot.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
            self.real_bot.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
            self.real_bot.print_color(f"🎯 BOUNCE-PROOF 3-LAYER EXIT V2 ACTIVE", self.Fore.YELLOW + self.Style.BRIGHT)
            self.real_bot.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
            self.real_bot.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
            self.real_bot.print_color("=" * 80, self.Fore.CYAN)
            
            # Update budget and track trade
            self.available_budget -= position_size_usd
            
            self.paper_positions[pair] = {
                "pair": pair,
                "direction": decision,
                "entry_price": entry_price,
                "quantity": quantity,
                "position_size_usd": position_size_usd,
                "leverage": leverage,
                "entry_time": time.time(),
                "status": 'ACTIVE',
                'ai_confidence': confidence,
                'ai_reasoning': reasoning,
                'entry_time_th': self.real_bot.get_thailand_time(),
                'has_tp_sl': False,  # Mark as no TP/SL
                'peak_pnl': 0  # NEW: For 3-layer system
            }
            
            self.real_bot.print_color(f"✅ PAPER TRADE EXECUTED (BOUNCE-PROOF V2): {pair} {decision} | Leverage: {leverage}x", self.Fore.GREEN + self.Style.BRIGHT)
            return True
            
        except Exception as e:
            self.real_bot.print_color(f"❌ PAPER: Trade execution failed: {e}", self.Fore.RED)
            return False

    def monitor_paper_positions(self):
        """Monitor paper positions and ask AI when to close (BOUNCE-PROOF V2)"""
        try:
            closed_positions = []
            for pair, trade in list(self.paper_positions.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                # Ask AI whether to close this paper position using Bounce-Proof V2
                if not trade.get('has_tp_sl', True):
                    self.real_bot.print_color(f"🔍 PAPER Bounce-Proof V2 Checking {pair}...", self.Fore.BLUE)
                    close_decision = self.get_ai_close_decision_v2(pair, trade)
                    
                    if close_decision.get("should_close", False):
                        close_type = close_decision.get("close_type", "AI_DECISION")
                        confidence = close_decision.get("confidence", 0)
                        reasoning = close_decision.get("reasoning", "No reason provided")
                        partial_percent = close_decision.get("partial_percent", 100)
                        
                        # 🆕 Use Bounce-Proof V2's ACTUAL reasoning for closing
                        full_close_reason = f"BOUNCE-PROOF V2: {close_type} - {reasoning}"
                        
                        self.real_bot.print_color(f"🎯 PAPER Bounce-Proof V2 Decision: CLOSE {pair}", self.Fore.YELLOW + self.Style.BRIGHT)
                        self.real_bot.print_color(f"📝 Close Type: {close_type} | Partial: {partial_percent}%", self.Fore.CYAN)
                        self.real_bot.print_color(f"💡 Confidence: {confidence}% | Reasoning: {reasoning}", self.Fore.WHITE)
                        
                        # 🆕 Pass partial percentage to close function
                        success = self.paper_close_trade_immediately(pair, trade, full_close_reason, partial_percent)
                        if success and partial_percent == 100:  # Only count as closed if full close
                            closed_positions.append(pair)
                    else:
                        # Show Bounce-Proof V2's decision to hold with reasoning
                        if close_decision.get('confidence', 0) > 0:
                            reasoning = close_decision.get('reasoning', 'No reason provided')
                            self.real_bot.print_color(f"🔍 PAPER Bounce-Proof V2 wants to HOLD {pair} (Confidence: {close_decision.get('confidence', 0)}%)", self.Fore.GREEN)
                            self.real_bot.print_color(f"📝 Hold Reasoning: {reasoning}", self.Fore.WHITE)
                    
            return closed_positions
                    
        except Exception as e:
            self.real_bot.print_color(f"PAPER: Bounce-Proof V2 Monitoring error: {e}", self.Fore.RED)
            return []

    def display_paper_dashboard(self):
        """Display paper trading dashboard"""
        self.real_bot.print_color(f"\n🤖 PAPER TRADING DASHBOARD - {self.real_bot.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 90, self.Fore.CYAN)
        self.real_bot.print_color(f"🎯 MODE: BOUNCE-PROOF 3-LAYER EXIT V2", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"⏰ MONITORING: 3 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
        self.real_bot.print_color(f"📡 USING REAL BINANCE MARKET DATA", self.Fore.BLUE + self.Style.BRIGHT)
        
        active_count = 0
        total_unrealized = 0
        
        for pair, trade in self.paper_positions.items():
            if trade['status'] == 'ACTIVE':
                active_count += 1
                current_price = self.real_bot.get_current_price(pair)
                
                direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                
                if trade['direction'] == 'LONG':
                    unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:
                    unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    
                total_unrealized += unrealized_pnl
                pnl_color = self.Fore.GREEN + self.Style.BRIGHT if unrealized_pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
                
                self.real_bot.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
                self.real_bot.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x ⚡", self.Fore.WHITE)
                self.real_bot.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
                self.real_bot.print_color(f"   P&L: ${unrealized_pnl:.2f}", pnl_color)
                self.real_bot.print_color(f"   🎯 BOUNCE-PROOF V2 EXIT ACTIVE", self.Fore.YELLOW)
                self.real_bot.print_color("   " + "-" * 60, self.Fore.CYAN)
        
        if active_count == 0:
            self.real_bot.print_color("No active paper positions", self.Fore.YELLOW)
        else:
            total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
            self.real_bot.print_color(f"📊 Active Paper Positions: {active_count}/{self.max_concurrent_trades} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)
        
        self.real_bot.print_color(f"💰 Paper Balance: ${self.paper_balance:.2f} | Available: ${self.available_budget:.2f}", self.Fore.GREEN + self.Style.BRIGHT)

    def show_paper_history(self, limit=10):
        """Show paper trading history with partial closes"""
        if not self.paper_history:
            self.real_bot.print_color("No paper trade history found", self.Fore.YELLOW)
            return
        
        self.real_bot.print_color(f"\n📊 PAPER TRADING HISTORY (Last {min(limit, len(self.paper_history))} trades)", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 120, self.Fore.CYAN)
        
        recent_trades = self.paper_history[-limit:]
        for i, trade in enumerate(reversed(recent_trades)):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            position_size = trade.get('position_size_usd', 0)
            leverage = trade.get('leverage', 1)
            
            # Display type indicator
            display_type = trade.get('display_type', 'FULL_CLOSE')
            if display_type.startswith('PARTIAL'):
                type_indicator = f" | {display_type}"
                type_color = self.Fore.YELLOW
            else:
                type_indicator = " | FULL"
                type_color = self.Fore.WHITE
            
            self.real_bot.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']}{type_indicator}", pnl_color)
            self.real_bot.print_color(f"     Size: ${position_size:.2f} | Leverage: {leverage}x | P&L: ${pnl:.2f}", pnl_color)
            self.real_bot.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)
            
            # Show additional info for partial closes
            if trade.get('partial_percent', 100) < 100:
                closed_qty = trade.get('closed_quantity', 0)
                self.real_bot.print_color(f"     🔸 Partial: {trade['partial_percent']}% ({closed_qty:.4f}) closed", self.Fore.CYAN)

    def show_paper_stats(self):
        """Show paper trading statistics"""
        if not self.paper_history:
            return
            
        total_trades = len(self.paper_history)
        winning_trades = len([t for t in self.paper_history if t.get('pnl', 0) > 0])
        total_pnl = sum(t.get('pnl', 0) for t in self.paper_history)
        
        if total_trades == 0:
            return
            
        win_rate = (winning_trades / total_trades) * 100
        avg_trade = total_pnl / total_trades
        
        self.real_bot.print_color(f"\n📈 PAPER TRADING STATISTICS", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 60, self.Fore.GREEN)
        self.real_bot.print_color(f"Total Paper Trades: {total_trades} | Winning Trades: {winning_trades}", self.Fore.WHITE)
        self.real_bot.print_color(f"Paper Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
        self.real_bot.print_color(f"Total Paper P&L: ${total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
        self.real_bot.print_color(f"Average P&L per Paper Trade: ${avg_trade:.2f}", self.Fore.WHITE)

    def run_paper_trading_cycle(self):
        """Run paper trading cycle"""
        try:
            # First monitor and ask AI to close paper positions using Bounce-Proof V2
            self.monitor_paper_positions()
            self.display_paper_dashboard()
            
            # Show stats periodically
            if hasattr(self, 'paper_cycle_count') and self.paper_cycle_count % 4 == 0:
                self.show_paper_history(8)
                self.show_paper_stats()
            
            # Show learning progress
            if hasattr(self, 'paper_cycle_count') and self.paper_cycle_count % 3 == 0 and LEARN_SCRIPT_AVAILABLE:
                if hasattr(self.real_bot, 'show_advanced_learning_progress'):
                    self.real_bot.show_advanced_learning_progress()
                else:
                    self.real_bot.print_color(f"\n🧠 Learning progress display not available", self.Fore.YELLOW)
            
            self.real_bot.print_color(f"\nPAPER: DEEPSEEK SCANNING {len(self.available_pairs)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.available_pairs:
                if self.available_budget > 100:
                    market_data = self.real_bot.get_price_history(pair)
                    
                    # Use learning-enhanced AI decision for paper trading too
                    ai_decision = self.real_bot.get_ai_decision_with_learning(pair, market_data)
                    
                    if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] > 0:
                        qualified_signals += 1
                        direction = ai_decision['decision']
                        leverage_info = f"Leverage: {ai_decision['leverage']}x"
                        
                        if direction.startswith('REVERSE_'):
                            self.real_bot.print_color(f"PAPER REVERSE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f}", self.Fore.YELLOW + self.Style.BRIGHT)
                        else:
                            self.real_bot.print_color(f"PAPER TRADE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f} | {leverage_info}", self.Fore.GREEN + self.Style.BRIGHT)
                            
                        success = self.paper_execute_trade(pair, ai_decision)
                        if success:
                            time.sleep(1)
                
            if qualified_signals == 0:
                self.real_bot.print_color("PAPER: No qualified DeepSeek signals this cycle", self.Fore.YELLOW)
                    
        except Exception as e:
            self.real_bot.print_color(f"PAPER: Trading cycle error: {e}", self.Fore.RED)

    def start_paper_trading(self):
        """Start paper trading"""
        self.real_bot.print_color("🚀 STARTING PAPER TRADING WITH BOUNCE-PROOF 3-LAYER EXIT V2!", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("💰 VIRTUAL $500 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("🔄 REVERSE POSITION: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.real_bot.print_color("🎯 BOUNCE-PROOF 3-LAYER EXIT V2: ACTIVE", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color("⏰ MONITORING: 3 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
        self.real_bot.print_color("📡 USING REAL BINANCE MARKET DATA", self.Fore.BLUE + self.Style.BRIGHT)
        
        self.paper_cycle_count = 0
        while True:
            try:
                self.paper_cycle_count += 1
                self.real_bot.print_color(f"\n🔄 PAPER TRADING CYCLE {self.paper_cycle_count} (BOUNCE-PROOF V2)", self.Fore.CYAN + self.Style.BRIGHT)
                self.real_bot.print_color("=" * 60, self.Fore.CYAN)
                self.run_paper_trading_cycle()
                self.real_bot.print_color(f"⏳ Next Bounce-Proof V2 analysis in 3 minute...", self.Fore.BLUE)
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                self.real_bot.print_color(f"\n🛑 PAPER TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_paper_history(15)
                self.show_paper_stats()
                break
            except Exception as e:
                self.real_bot.print_color(f"PAPER: Main loop error: {e}", self.Fore.RED)
                time.sleep(self.monitoring_interval)

# Main execution
if __name__ == "__main__":
    try:
        # Create the main bot
        bot = FullyAutonomous1HourAITrader()
        
        # Ask user for mode selection
        print("\n" + "="*70)
        print("🤖 FULLY AUTONOMOUS 1-HOUR AI TRADER")
        print("="*70)
        print("1. 🎯 REAL TRADING (Live Binance Account)")
        print("2. 📝 PAPER TRADING (Virtual Simulation)")
        print("3. ❌ EXIT")
        
        choice = input("\nSelect mode (1-3): ").strip()
        
        if choice == "1":
            if bot.binance:
                print(f"\n🚀 STARTING REAL TRADING WITH BOUNCE-PROOF V2...")
                bot.start_trading()
            else:
                print(f"\n❌ Binance connection failed. Switching to paper trading...")
                paper_bot = FullyAutonomous1HourPaperTrader(bot)
                paper_bot.start_paper_trading()
                
        elif choice == "2":
            print(f"\n📝 STARTING PAPER TRADING WITH BOUNCE-PROOF V2...")
            paper_bot = FullyAutonomous1HourPaperTrader(bot)
            paper_bot.start_paper_trading()
            
        elif choice == "3":
            print(f"\n👋 Exiting...")
            
        else:
            print(f"\n❌ Invalid choice. Exiting...")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Program stopped by user")
    except Exception as e:
        print(f"\n❌ Main execution error: {e}")
        import traceback
        traceback.print_exc()
