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

# ==================== V6.0 EVERY 3% AI CHECK SYSTEM ====================
class FullyAutonomous1HourAITrader:
    def __init__(self):
        self._initialize_trading()
        
        if LEARN_SCRIPT_AVAILABLE:
            self.learning_module = SelfLearningAITrader()
            self.mistakes_history = self.learning_module.mistakes_history
            self.learned_patterns = self.learning_module.learned_patterns
            self.performance_stats = self.learning_module.performance_stats
            print("🧠 Self-learning AI module loaded")
        else:
            self.mistakes_history = []
            self.learned_patterns = {}
            self.performance_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'common_mistakes': {},
                'improvement_areas': []
            }
        
        # 3% Increment System
        self.checked_3percent_levels = {}  # {pair: [checked_levels]}
        self.last_ai_check_time = {}
        
    def _initialize_trading(self):
        """Initialize trading components"""
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
        self.total_budget = 500
        self.available_budget = 500
        self.max_position_size_percent = 10
        self.max_concurrent_trades = 4
        
        # AI can trade selected 3 major pairs only
        self.available_pairs = [
            "SOLUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "HYPEUSDT"
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
        
        # 🆕 EVERY 3% AI CHECK SETTINGS
        self.exit_strategy_mode = "3PERCENT_AI_CHECK"  # "HARD_RULES", "AI_ONLY", "3PERCENT_AI_CHECK"
        self.min_check_level = 6  # Start checking at 6%
        self.percent_increment = 3  # Every 3%
        self.force_partial_at_milestones = True  # Force partial at 10%, 15%, 20%, etc
        self.time_based_check_minutes = 15  # Check every 15 minutes regardless of level
        
        # Reverse position settings
        self.allow_reverse_positions = True
        
        # Monitoring interval (3 minutes)
        self.monitoring_interval = 180
        
        # Validate APIs
        self.validate_api_keys()
        
        # Initialize Binance client
        try:
            self.binance = Client(self.binance_api_key, self.binance_secret)
            print("🤖 FULLY AUTONOMOUS AI TRADER V6.0 ACTIVATED!")
            print(f"💰 TOTAL BUDGET: ${self.total_budget}")
            print(f"🎯 EXIT STRATEGY: EVERY 3% AI CHECK")
            print(f"📊 Check starts at: {self.min_check_level}%")
            print(f"⏰ Additional checks: Every {self.time_based_check_minutes} minutes")
            print(f"📈 Force partial at: 10%, 15%, 20% milestones")
        except Exception as e:
            print(f"Binance initialization failed: {e}")
            self.binance = None
        
        self.validate_config()
        if self.binance:
            self.setup_futures()
            self.load_symbol_precision()
    
    # ==================== CORE FUNCTIONS ====================
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
        """Add trade to history"""
        try:
            trade_data['close_time'] = self.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            trade_data['trade_type'] = 'REAL'
            
            if 'exit_price' not in trade_data:
                current_price = self.get_current_price(trade_data['pair'])
                trade_data['exit_price'] = current_price
            
            if 'peak_pnl_pct' not in trade_data:
                if 'peak_pnl' in trade_data:
                    trade_data['peak_pnl_pct'] = trade_data['peak_pnl']
                else:
                    if trade_data['direction'] == 'LONG':
                        peak_pct = ((trade_data['exit_price'] - trade_data['entry_price']) / 
                                   trade_data['entry_price']) * 100 * trade_data.get('leverage', 1)
                    else:
                        peak_pct = ((trade_data['entry_price'] - trade_data['exit_price']) / 
                                   trade_data['entry_price']) * 100 * trade_data.get('leverage', 1)
                    trade_data['peak_pnl_pct'] = max(0, peak_pct)
            
            if trade_data.get('partial_percent', 100) < 100:
                trade_data['display_type'] = f"PARTIAL_{trade_data['partial_percent']}%"
            else:
                trade_data['display_type'] = "FULL_CLOSE"
            
            self.real_trade_history.append(trade_data)
            
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
            
            # Log for ML
            try:
                from data_collector import log_trade_for_ml
                log_trade_for_ml(trade_data)
            except:
                pass
            
            if trade_data.get('partial_percent', 100) < 100:
                self.print_color(f"📝 Partial close saved: {trade_data['pair']} {trade_data['partial_percent']}% | P&L: ${pnl:.2f}", self.Fore.CYAN)
            else:
                self.print_color(f"📝 Trade saved: {trade_data['pair']} {trade_data['direction']} | P&L: ${pnl:.2f}", self.Fore.CYAN)
                
        except Exception as e:
            self.print_color(f"Error adding trade to history: {e}", self.Fore.RED)
    
    def get_thailand_time(self):
        now_utc = datetime.now(pytz.utc)
        thailand_time = now_utc.astimezone(self.thailand_tz)
        return thailand_time.strftime('%Y-%m-%d %H:%M:%S')
    
    def print_color(self, text, color=""):
        if self.COLORAMA_AVAILABLE:
            print(f"{color}{text}")
        else:
            print(text)
    
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
        
        return len(issues) == 0
    
    def validate_config(self):
        if not all([self.binance_api_key, self.binance_secret]):
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
                    self.binance.futures_change_leverage(symbol=pair, leverage=5)
                    self.binance.futures_change_margin_type(symbol=pair, marginType='ISOLATED')
                except Exception as e:
                    self.print_color(f"Leverage setup failed for {pair}: {e}", self.Fore.YELLOW)
            self.print_color("✅ Futures setup completed!", self.Fore.GREEN + self.Style.BRIGHT)
        except Exception as e:
            self.print_color(f"Futures setup failed: {e}", self.Fore.RED)
    
    def load_symbol_precision(self):
        if not self.binance:
            for pair in self.available_pairs:
                try:
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
    
    # ==================== MARKET DATA ====================
    def get_current_price(self, pair):
        """Get real price from Binance API"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if self.binance:
                    ticker = self.binance.futures_symbol_ticker(symbol=pair)
                    return float(ticker['price'])
                
                response = requests.get(
                    f'https://api.binance.com/api/v3/ticker/price?symbol={pair}',
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return float(data['price'])
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        fallback_prices = {
            "SOLUSDT": 140.0, "XRPUSDT": 2.2, "AVAXUSDT": 15.0, 
            "LTCUSDT": 85.0, "HYPEUSDT": 35.0
        }
        return fallback_prices.get(pair, 100.0)
    
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
    
    def get_price_history(self, pair, limit=50):
        """Multi-Timeframe Analysis with REAL Binance data"""
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
            self.print_color(f"MTF Analysis error: {e}", self.Fore.RED)
            return {
                'current_price': self.get_current_price(pair),
                'price_change': 0,
                'support_levels': [],
                'resistance_levels': [],
                'mtf_analysis': {}
            }
    
    # ==================== AI DECISION MAKING ====================
    def get_ai_trading_decision(self, pair, market_data, current_trade=None):
        """AI makes trading decisions including REVERSE positions"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if not self.openrouter_key:
                    return self.get_improved_fallback_decision(pair, market_data)
                
                current_price = market_data.get('current_price', 0)
                mtf = market_data.get('mtf_analysis', {})
                
                prompt = f"""
YOU ARE A PROFESSIONAL AI TRADER. Budget: ${self.available_budget:.2f}

1H TRADING PAIR: {pair}
Current Price: ${current_price:.6f}

MULTI-TIMEFRAME ANALYSIS:
"""
                for tf in ['5m', '15m', '1h', '4h', '1d']:
                    if tf in mtf:
                        d = mtf[tf]
                        prompt += f"- {tf.upper()}: {d.get('trend', 'N/A')} | RSI: {d.get('rsi', 50)} | "
                        if 'crossover' in d:
                            prompt += f"Signal: {d['crossover']} | "
                        prompt += f"S/R: {d.get('support', 0):.4f}/{d.get('resistance', 0):.4f}\n"
                
                if current_trade and self.allow_reverse_positions:
                    pnl = self.calculate_current_pnl(current_trade, current_price)
                    prompt += f"""
EXISTING POSITION:
- Direction: {current_trade['direction']}
- Entry: ${current_trade['entry_price']:.4f}
- PnL: {pnl:.2f}%
- Consider REVERSE if trend flipped?
"""
                
                prompt += """
RULES:
- Only trade if 1H and 4H trend align
- Confirm entry with 15m crossover + volume spike
- Position size: 5-10% of budget ($50 min)
- Leverage: 5-10x based on volatility
- NO TP/SL - AI will close manually

Return JSON:
{
    "decision": "LONG" | "SHORT" | "HOLD" | "REVERSE_LONG" | "REVERSE_SHORT",
    "position_size_usd": number,
    "entry_price": number,
    "leverage": number,
    "confidence": 0-100,
    "reasoning": "MTF alignment + signal + risk"
}
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
                        {"role": "system", "content": "You are a fully autonomous AI trader with reverse position capability."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800
                }
                
                self.print_color(f"🧠 DeepSeek Analyzing {pair}...", self.Fore.MAGENTA + self.Style.BRIGHT)
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                       headers=headers, json=data, timeout=60)
                
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
        
        return self.get_improved_fallback_decision(pair, market_data)
    
    def get_ai_exit_decision_at_level(self, pair, trade, market_data, current_level, current_pnl):
        """Ask AI for exit decision at specific 3% level"""
        try:
            if not self.openrouter_key:
                return self.get_fallback_exit_decision_at_level(pair, trade, current_level, current_pnl)
            
            prompt = f"""
SPECIFIC 3% LEVEL CHECK: {pair} reached +{current_pnl:.1f}% (Level {current_level}%)

POSITION DETAILS:
- Direction: {trade['direction']}
- Entry Price: ${trade['entry_price']:.4f}
- Current Price: ${market_data['current_price']:.4f}
- Current PnL: +{current_pnl:.1f}%
- Leverage: {trade['leverage']}x
- Position Size: ${trade['position_size_usd']:.2f}

MARKET ANALYSIS AT THIS LEVEL:
"""
            mtf = market_data.get('mtf_analysis', {})
            for tf in ['15m', '1h', '4h']:
                if tf in mtf:
                    d = mtf[tf]
                    prompt += f"{tf.upper()}: Trend: {d.get('trend', 'N/A')}, RSI: {d.get('rsi', 50)}, "
                    prompt += f"Support: {d.get('support', 0):.4f}, Resistance: {d.get('resistance', 0):.4f}\n"
            
            prompt += f"""
SPECIFIC QUESTION: At this exact +{current_level}% profit level, what should we do?

OPTIONS:
1. HOLD - Wait for next level (+{current_level + self.percent_increment}%)
2. TAKE PARTIAL PROFIT - How much % to take?
3. CLOSE FULLY - Take all profit now

CONSIDER:
- Next 3% level is at +{current_level + self.percent_increment}%
- Risk/reward ratio at current level
- Market momentum and trend
- Time in trade: {((time.time() - trade.get('entry_time', time.time())) / 3600):.1f} hours

BE SPECIFIC about partial percentages if taking profit.

Return JSON:
{{
    "action": "HOLD_NEXT_LEVEL" | "TAKE_PARTIAL" | "CLOSE_FULL",
    "partial_percent": number (if taking partial, e.g., 20 for 20%),
    "next_check_at": {current_level + self.percent_increment},
    "confidence": 0-100,
    "reasoning": "Specific analysis for this {current_level}% level"
}}
"""
            
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek/deepseek-chat-v3.1",
                "messages": [
                    {"role": "system", "content": "You are an AI trader making specific exit decisions at each 3% profit level. Be precise about partial profit percentages."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                return self.parse_ai_level_exit_decision(ai_response, current_level)
                
        except Exception as e:
            self.print_color(f"AI Level Exit decision failed: {e}", self.Fore.YELLOW)
        
        return self.get_fallback_exit_decision_at_level(pair, trade, current_level, current_pnl)
    
    def parse_ai_trading_decision(self, ai_response, pair, current_price, current_trade=None):
        """Parse AI's trading decision"""
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
        except Exception as e:
            self.print_color(f"AI response parsing failed: {e}", self.Fore.RED)
        
        return self.get_improved_fallback_decision(pair, {'current_price': current_price})
    
    def parse_ai_level_exit_decision(self, ai_response, current_level):
        """Parse AI's level-based exit decision"""
        try:
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision_data = json.loads(json_str)
                
                action = decision_data.get('action', 'HOLD_NEXT_LEVEL')
                confidence = decision_data.get('confidence', 50)
                reasoning = decision_data.get('reasoning', 'No reason')
                
                if action == 'HOLD_NEXT_LEVEL':
                    return {
                        "should_close": False,
                        "next_level": decision_data.get('next_check_at', current_level + self.percent_increment),
                        "reasoning": reasoning,
                        "confidence": confidence
                    }
                
                elif action == 'TAKE_PARTIAL':
                    partial_percent = decision_data.get('partial_percent', 30)
                    
                    return {
                        "should_close": True,
                        "partial_percent": partial_percent,
                        "close_type": f"AI_3PERCENT_LEVEL_{current_level}",
                        "reasoning": reasoning,
                        "confidence": confidence
                    }
                
                elif action == 'CLOSE_FULL':
                    return {
                        "should_close": True,
                        "partial_percent": 100,
                        "close_type": f"AI_FULL_AT_{current_level}",
                        "reasoning": reasoning,
                        "confidence": confidence
                    }
        
        except Exception as e:
            self.print_color(f"Failed to parse AI level exit decision: {e}", self.Fore.RED)
        
        return {"should_close": False}
    
    def get_improved_fallback_decision(self, pair, market_data):
        """Better fallback decision"""
        current_price = market_data['current_price']
        mtf = market_data.get('mtf_analysis', {})
        
        bullish_signals = 0
        bearish_signals = 0
        
        h1_data = mtf.get('1h', {})
        h4_data = mtf.get('4h', {})
        m15_data = mtf.get('15m', {})
        
        if h1_data.get('trend') == 'BULLISH':
            bullish_signals += 1
        elif h1_data.get('trend') == 'BEARISH':
            bearish_signals += 1
        
        if h4_data.get('trend') == 'BULLISH':
            bullish_signals += 1
        elif h4_data.get('trend') == 'BEARISH':
            bearish_signals += 1
        
        h1_rsi = h1_data.get('rsi', 50)
        if h1_rsi < 35:
            bullish_signals += 1
        elif h1_rsi > 65:
            bearish_signals += 1
        
        if m15_data.get('crossover') == 'GOLDEN':
            bullish_signals += 1
        elif m15_data.get('crossover') == 'DEATH':
            bearish_signals += 1
        
        if bullish_signals >= 3 and bearish_signals <= 1:
            return {
                "decision": "LONG",
                "position_size_usd": 20,
                "entry_price": current_price,
                "leverage": 5,
                "confidence": 60,
                "reasoning": f"Fallback: Bullish signals ({bullish_signals}/{bearish_signals})",
                "should_reverse": False
            }
        elif bearish_signals >= 3 and bullish_signals <= 1:
            return {
                "decision": "SHORT", 
                "position_size_usd": 20,
                "entry_price": current_price,
                "leverage": 5,
                "confidence": 60,
                "reasoning": f"Fallback: Bearish signals ({bearish_signals}/{bullish_signals})",
                "should_reverse": False
            }
        else:
            return {
                "decision": "HOLD",
                "position_size_usd": 0,
                "entry_price": current_price,
                "leverage": 5,
                "confidence": 40,
                "reasoning": f"Fallback: Mixed signals ({bullish_signals}/{bearish_signals})",
                "should_reverse": False
            }
    
    def get_fallback_exit_decision_at_level(self, pair, trade, current_level, current_pnl):
        """Fallback exit decision at specific level"""
        # Progressive partial closing based on level
        if current_level >= 24:
            partial_percent = 70
        elif current_level >= 21:
            partial_percent = 60
        elif current_level >= 18:
            partial_percent = 50
        elif current_level >= 15:
            partial_percent = 40
        elif current_level >= 12:
            partial_percent = 30
        elif current_level >= 9:
            partial_percent = 20
        elif current_level >= 6:
            partial_percent = 10
        else:
            partial_percent = 0
        
        if partial_percent > 0:
            return {
                "should_close": True,
                "partial_percent": partial_percent,
                "close_type": f"FALLBACK_LEVEL_{current_level}",
                "reasoning": f"Fallback: Taking {partial_percent}% profit at +{current_level}%",
                "confidence": 70
            }
        
        return {"should_close": False}
    
    # ==================== 3% INCREMENT EXIT SYSTEM ====================
    def check_3percent_level(self, pair, trade):
        """Check if we hit a new 3% level and ask AI"""
        current_price = self.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        # Track peak
        if 'peak_pnl' not in trade:
            trade['peak_pnl'] = current_pnl
        elif current_pnl > trade['peak_pnl']:
            trade['peak_pnl'] = current_pnl
        
        # Calculate current 3% level
        current_level = math.floor(current_pnl / self.percent_increment) * self.percent_increment
        
        # Skip if below minimum check level
        if current_level < self.min_check_level:
            return {"should_close": False}
        
        # Initialize checked levels for this pair
        if pair not in self.checked_3percent_levels:
            self.checked_3percent_levels[pair] = []
        
        # Check if this level needs AI evaluation
        if current_level not in self.checked_3percent_levels[pair]:
            self.print_color(f"🎯 {pair} reached +{current_pnl:.1f}% (Level {current_level}%)", self.Fore.CYAN + self.Style.BRIGHT)
            
            # Add to checked levels
            self.checked_3percent_levels[pair].append(current_level)
            
            # Ask AI for decision at this level
            market_data = self.get_price_history(pair)
            ai_decision = self.get_ai_exit_decision_at_level(pair, trade, market_data, current_level, current_pnl)
            
            return ai_decision
        
        return {"should_close": False}
    
    def check_milestone_partial(self, pair, trade):
        """Force partial close at milestone levels (10%, 15%, 20%, etc)"""
        current_price = self.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if not self.force_partial_at_milestones:
            return {"should_close": False}
        
        milestone_levels = [10, 15, 20, 25, 30]
        
        for milestone in milestone_levels:
            if current_pnl >= milestone and current_pnl < milestone + 1:
                # Check if we already took partial at this milestone
                milestone_key = f"{pair}_milestone_{milestone}"
                if milestone_key not in trade:
                    trade[milestone_key] = True
                    
                    # Calculate partial percentage (more at higher milestones)
                    if milestone >= 25:
                        partial_percent = 40
                    elif milestone >= 20:
                        partial_percent = 30
                    elif milestone >= 15:
                        partial_percent = 20
                    else:  # 10%
                        partial_percent = 15
                    
                    return {
                        "should_close": True,
                        "partial_percent": partial_percent,
                        "close_type": f"MILESTONE_{milestone}_PARTIAL",
                        "reasoning": f"🎉 Milestone! Taking {partial_percent}% profit at +{milestone}%",
                        "confidence": 85
                    }
        
        return {"should_close": False}
    
    def check_time_based_exit(self, pair, trade):
        """Check exit based on time"""
        current_time = time.time()
        entry_time = trade.get('entry_time', current_time)
        
        # Calculate hours in trade
        hours_in_trade = (current_time - entry_time) / 3600
        
        # Check every 15 minutes
        last_check = self.last_ai_check_time.get(pair, 0)
        if current_time - last_check >= (self.time_based_check_minutes * 60):
            self.last_ai_check_time[pair] = current_time
            
            current_price = self.get_current_price(pair)
            current_pnl = self.calculate_current_pnl(trade, current_price)
            
            # Only ask AI if profit > 5%
            if current_pnl >= 5:
                market_data = self.get_price_history(pair)
                
                # Special prompt for time-based check
                prompt = f"""
TIME-BASED CHECK: {pair} has been open for {hours_in_trade:.1f} hours
Current PnL: +{current_pnl:.1f}%

POSITION:
- Direction: {trade['direction']}
- Entry: ${trade['entry_price']:.4f}
- Current: ${current_price:.4f}
- Leverage: {trade['leverage']}x

Considering time in trade, should we:
1. Continue holding?
2. Take partial profit?
3. Close fully?

Return JSON:
{{
    "action": "HOLD" | "TAKE_PARTIAL" | "CLOSE_FULL",
    "partial_percent": number (if taking partial),
    "reasoning": "Time-based analysis..."
}}
"""
                
                try:
                    headers = {
                        "Authorization": f"Bearer {self.openrouter_key}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": "deepseek/deepseek-chat-v3.1",
                        "messages": [
                            {"role": "system", "content": "You are an AI trader considering time-based exits."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 300
                    }
                    
                    response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                           headers=headers, json=data, timeout=20)
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_response = result['choices'][0]['message']['content']
                        
                        # Parse response
                        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            decision_data = json.loads(json_str)
                            
                            action = decision_data.get('action', 'HOLD')
                            
                            if action == 'TAKE_PARTIAL':
                                partial_percent = decision_data.get('partial_percent', 20)
                                return {
                                    "should_close": True,
                                    "partial_percent": partial_percent,
                                    "close_type": "TIME_BASED_PARTIAL",
                                    "reasoning": f"⏰ Time check: {decision_data.get('reasoning', '')}",
                                    "confidence": 75
                                }
                            
                            elif action == 'CLOSE_FULL':
                                return {
                                    "should_close": True,
                                    "partial_percent": 100,
                                    "close_type": "TIME_BASED_FULL",
                                    "reasoning": f"⏰ Time check: {decision_data.get('reasoning', '')}",
                                    "confidence": 75
                                }
                
                except Exception as e:
                    self.print_color(f"Time-based AI check failed: {e}", self.Fore.YELLOW)
            
            # Fallback: Small partial if trade is old and profitable
            if hours_in_trade >= 4 and current_pnl >= 10:
                return {
                    "should_close": True,
                    "partial_percent": 15,
                    "close_type": "TIME_FALLBACK_PARTIAL",
                    "reasoning": f"Trade open {hours_in_trade:.1f}h with +{current_pnl:.1f}% profit",
                    "confidence": 70
                }
        
        return {"should_close": False}
    
    def check_peak_drawdown_protection(self, pair, trade):
        """Protect against drawdowns from peak"""
        current_price = self.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if 'peak_pnl' not in trade:
            trade['peak_pnl'] = current_pnl
        
        peak = trade['peak_pnl']
        
        # Only check if we have a significant peak
        if peak < 6:
            return {"should_close": False}
        
        # Calculate drawdown from peak
        drawdown = peak - current_pnl
        
        # Drawdown protection rules
        if drawdown >= 6:  # Lost 6% from peak
            return {
                "should_close": True,
                "partial_percent": 100,
                "close_type": "PEAK_DRAWDOWN_6",
                "reasoning": f"🚨 Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 90
            }
        
        elif drawdown >= 4:  # Lost 4% from peak
            return {
                "should_close": True,
                "partial_percent": 50,
                "close_type": "PEAK_DRAWDOWN_4",
                "reasoning": f"⚠️ Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 80
            }
        
        elif drawdown >= 2 and peak >= 15:  # Lost 2% from 15%+ peak
            return {
                "should_close": True,
                "partial_percent": 30,
                "close_type": "PEAK_DRAWDOWN_2",
                "reasoning": f"Lost {drawdown:.1f}% from high peak {peak:.1f}%",
                "confidence": 75
            }
        
        return {"should_close": False}
    
    def get_3percent_exit_decision(self, pair, trade):
        """Main 3% increment exit decision system"""
        
        # Check emergency stops first (always highest priority)
        current_pnl = self.calculate_current_pnl(trade, self.get_current_price(pair))
        if current_pnl <= -5.0:
            return {
                "should_close": True,
                "partial_percent": 100,
                "close_type": "EMERGENCY_STOP_5",
                "reasoning": f"🚨 Emergency stop at -{abs(current_pnl):.1f}%",
                "confidence": 100
            }
        
        # Check peak drawdown protection
        drawdown_decision = self.check_peak_drawdown_protection(pair, trade)
        if drawdown_decision.get("should_close", False):
            return drawdown_decision
        
        # Check 3% level
        level_decision = self.check_3percent_level(pair, trade)
        if level_decision.get("should_close", False):
            return level_decision
        
        # Check milestone partials
        milestone_decision = self.check_milestone_partial(pair, trade)
        if milestone_decision.get("should_close", False):
            return milestone_decision
        
        # Check time-based exit
        time_decision = self.check_time_based_exit(pair, trade)
        if time_decision.get("should_close", False):
            return time_decision
        
        return {"should_close": False}
    
    # ==================== TRADE EXECUTION ====================
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
    
    def calculate_quantity(self, pair, entry_price, position_size_usd, leverage):
        """Calculate quantity based on position size and leverage"""
        try:
            if entry_price <= 0:
                return None
                
            notional_value = position_size_usd * leverage
            quantity = notional_value / entry_price
            
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
    
    def close_trade_immediately(self, pair, trade, close_reason="AI_DECISION", partial_percent=100):
        """Close trade immediately at market price"""
        try:
            current_price = self.get_current_price(pair)
            
            if trade['direction'] == 'LONG':
                pnl = (current_price - trade['entry_price']) * trade['quantity'] * (partial_percent / 100)
            else:
                pnl = (trade['entry_price'] - current_price) * trade['quantity'] * (partial_percent / 100)
            
            peak_pnl_pct = trade.get('peak_pnl', 0)
            
            if partial_percent < 100:
                remaining_quantity = trade['quantity'] * (1 - partial_percent / 100)
                closed_quantity = trade['quantity'] * (partial_percent / 100)
                closed_position_size = trade['position_size_usd'] * (partial_percent / 100)
                
                trade['quantity'] = remaining_quantity
                trade['position_size_usd'] = trade['position_size_usd'] * (1 - partial_percent / 100)
                
                partial_trade = trade.copy()
                partial_trade['status'] = 'PARTIAL_CLOSE'
                partial_trade['exit_price'] = current_price
                partial_trade['pnl'] = pnl
                partial_trade['close_reason'] = close_reason
                partial_trade['close_time'] = self.get_thailand_time()
                partial_trade['partial_percent'] = partial_percent
                partial_trade['closed_quantity'] = closed_quantity
                partial_trade['closed_position_size'] = closed_position_size
                partial_trade['peak_pnl_pct'] = round(peak_pnl_pct, 3)
                
                self.available_budget += closed_position_size + pnl
                self.add_trade_to_history(partial_trade)
                
                pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
                self.print_color(f"✅ Partial Close | {pair} | {partial_percent}% | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                self.print_color(f"📊 Remaining: {remaining_quantity:.4f} {pair} (${trade['position_size_usd']:.2f})", self.Fore.CYAN)
                
                return True
                
            else:
                trade['status'] = 'CLOSED'
                trade['exit_price'] = current_price
                trade['pnl'] = pnl
                trade['close_reason'] = close_reason
                trade['close_time'] = self.get_thailand_time()
                trade['partial_percent'] = 100
                trade['peak_pnl_pct'] = round(peak_pnl_pct, 3)
                
                self.available_budget += trade['position_size_usd'] + pnl
                self.add_trade_to_history(trade.copy())
                
                pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
                self.print_color(f"✅ Full Close | {pair} | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                
                if pair in self.ai_opened_trades:
                    del self.ai_opened_trades[pair]
                
                return True
                
        except Exception as e:
            self.print_color(f"❌ Close failed: {e}", self.Fore.RED)
            return False
    
    def execute_ai_trade(self, pair, ai_decision):
        """Execute trade WITHOUT TP/SL orders - AI will close manually"""
        try:
            decision = ai_decision["decision"]
            position_size_usd = ai_decision["position_size_usd"]
            entry_price = ai_decision["entry_price"]
            leverage = ai_decision["leverage"]
            confidence = ai_decision["confidence"]
            reasoning = ai_decision["reasoning"]
            
            if decision == "HOLD" or position_size_usd <= 0:
                self.print_color(f"🟡 DeepSeek decides to HOLD {pair}", self.Fore.YELLOW)
                return False
            
            if pair in self.ai_opened_trades:
                self.print_color(f"🚫 Cannot open {pair}: Position already exists", self.Fore.RED)
                return False
            
            if len(self.ai_opened_trades) >= self.max_concurrent_trades and pair not in self.ai_opened_trades:
                self.print_color(f"🚫 Cannot open {pair}: Max concurrent trades reached", self.Fore.RED)
                return False
                
            if position_size_usd > self.available_budget:
                self.print_color(f"🚫 Cannot open {pair}: Insufficient budget", self.Fore.RED)
                return False
            
            quantity = self.calculate_quantity(pair, entry_price, position_size_usd, leverage)
            if quantity is None:
                return False
            
            direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
            
            self.print_color(f"\n🤖 DEEPSEEK TRADE EXECUTION (3% AI CHECK SYSTEM)", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 80, self.Fore.CYAN)
            self.print_color(f"{direction_icon} {pair}", direction_color)
            self.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
            self.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
            self.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
            self.print_color(f"🎯 EXIT STRATEGY: EVERY 3% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"📊 Check starts at: {self.min_check_level}%", self.Fore.MAGENTA)
            self.print_color(f"⏰ Time checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
            self.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
            self.print_color("=" * 80, self.Fore.CYAN)
            
            # Execute live trade WITHOUT TP/SL orders
            if self.binance:
                entry_side = 'BUY' if decision == 'LONG' else 'SELL'
                
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=leverage)
                except Exception as e:
                    self.print_color(f"Leverage change failed: {e}", self.Fore.YELLOW)
                
                order = self.binance.futures_create_order(
                    symbol=pair,
                    side=entry_side,
                    type='MARKET',
                    quantity=quantity
                )
            
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
                'has_tp_sl': False,
                'peak_pnl': 0
            }
            
            # Initialize checked levels for this pair
            self.checked_3percent_levels[pair] = []
            
            self.print_color(f"✅ TRADE EXECUTED: {pair} {decision} | Leverage: {leverage}x", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"📊 AI will check at every 3% profit level", self.Fore.BLUE)
            return True
            
        except Exception as e:
            self.print_color(f"❌ Trade execution failed: {e}", self.Fore.RED)
            return False
    
    # ==================== MONITORING ====================
    def monitor_positions(self):
        """Monitor positions using 3% AI Check system"""
        try:
            closed_trades = []
            
            for pair, trade in list(self.ai_opened_trades.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                self.print_color(f"🔍 3% System Checking {pair}...", self.Fore.BLUE)
                
                # Get exit decision from 3% system
                exit_decision = self.get_3percent_exit_decision(pair, trade)
                
                if exit_decision.get("should_close", False):
                    close_type = exit_decision.get("close_type", "EXIT")
                    reasoning = exit_decision.get("reasoning", "No reason")
                    partial_percent = exit_decision.get("partial_percent", 100)
                    confidence = exit_decision.get("confidence", 0)
                    
                    self.print_color(f"🎯 3% System Decision for {pair}:", self.Fore.CYAN + self.Style.BRIGHT)
                    self.print_color(f"   Action: {'PARTIAL' if partial_percent < 100 else 'FULL'} CLOSE", self.Fore.YELLOW)
                    self.print_color(f"   Type: {close_type}", self.Fore.MAGENTA)
                    self.print_color(f"   Confidence: {confidence}%", self.Fore.GREEN if confidence > 70 else self.Fore.YELLOW)
                    self.print_color(f"   Reason: {reasoning}", self.Fore.WHITE)
                    
                    success = self.close_trade_immediately(pair, trade, f"{close_type}: {reasoning}", partial_percent)
                    if success and partial_percent == 100:
                        closed_trades.append(pair)
                        
                        # Clean up checked levels
                        if pair in self.checked_3percent_levels:
                            del self.checked_3percent_levels[pair]
            
            return closed_trades
                
        except Exception as e:
            self.print_color(f"Monitoring error: {e}", self.Fore.RED)
            return []
    
    # ==================== DASHBOARD ====================
    def display_dashboard(self):
        """Display trading dashboard"""
        self.print_color(f"\n🤖 AI TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 90, self.Fore.CYAN)
        self.print_color(f"🎯 EXIT STRATEGY: EVERY 3% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"📊 Check Levels: {self.min_check_level}%, {self.min_check_level+3}%, {self.min_check_level+6}%, etc.", self.Fore.MAGENTA)
        self.print_color(f"⏰ Time Checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
        self.print_color(f"💰 Milestone Partials: 10%, 15%, 20%, etc.", self.Fore.GREEN)
        self.print_color(f"📉 Drawdown Protection: Active", self.Fore.RED)
        
        active_count = 0
        total_unrealized = 0
        
        for pair, trade in self.ai_opened_trades.items():
            if trade['status'] == 'ACTIVE':
                active_count += 1
                current_price = self.get_current_price(pair)
                
                direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                current_pnl = self.calculate_current_pnl(trade, current_price)
                
                if trade['direction'] == 'LONG':
                    unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:
                    unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    
                total_unrealized += unrealized_pnl
                pnl_color = self.Fore.GREEN + self.Style.BRIGHT if unrealized_pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
                
                # Calculate current 3% level
                current_level = math.floor(current_pnl / self.percent_increment) * self.percent_increment
                next_level = current_level + self.percent_increment
                
                self.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
                self.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x ⚡", self.Fore.WHITE)
                self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
                self.print_color(f"   P&L: ${unrealized_pnl:.2f} ({current_pnl:.1f}%)", pnl_color)
                self.print_color(f"   📈 Next AI Check: +{next_level}%", self.Fore.CYAN)
                
                if 'peak_pnl' in trade:
                    peak = trade['peak_pnl']
                    self.print_color(f"   🏔️ Peak: {peak:.1f}% | Drawdown: {max(0, peak - current_pnl):.1f}%", 
                                   self.Fore.YELLOW if peak - current_pnl <= 2 else self.Fore.RED)
                
                self.print_color("   " + "-" * 60, self.Fore.CYAN)
        
        if active_count == 0:
            self.print_color("No active positions", self.Fore.YELLOW)
        else:
            total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
            self.print_color(f"📊 Active Positions: {active_count}/{self.max_concurrent_trades} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)
    
    def show_trade_history(self, limit=15):
        """Show trading history"""
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
            
            display_type = trade.get('display_type', 'FULL_CLOSE')
            if display_type.startswith('PARTIAL'):
                type_indicator = f" | {display_type}"
            else:
                type_indicator = " | FULL"
            
            self.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']}{type_indicator}", pnl_color)
            self.print_color(f"     Size: ${position_size:.2f} | Leverage: {leverage}x | P&L: ${pnl:.2f}", pnl_color)
            self.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)
            
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
    
    # ==================== MAIN TRADING LOOP ====================
    def run_trading_cycle(self):
        """Run trading cycle"""
        try:
            # Monitor and close positions
            self.monitor_positions()
            self.display_dashboard()
            
            # Show stats periodically
            if hasattr(self, 'cycle_count') and self.cycle_count % 4 == 0:
                self.show_trade_history(8)
                self.show_trading_stats()
            
            self.print_color(f"\n🔍 DEEPSEEK SCANNING {len(self.available_pairs)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.available_pairs:
                if self.available_budget > 100:
                    market_data = self.get_price_history(pair)
                    
                    ai_decision = self.get_ai_trading_decision(pair, market_data)
                    
                    if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] > 0:
                        qualified_signals += 1
                        direction = ai_decision['decision']
                        leverage_info = f"Leverage: {ai_decision['leverage']}x"
                        
                        self.print_color(f"🎯 TRADE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f} | {leverage_info}", self.Fore.GREEN + self.Style.BRIGHT)
                        
                        success = self.execute_ai_trade(pair, ai_decision)
                        if success:
                            time.sleep(2)
            
            if qualified_signals == 0:
                self.print_color("No qualified DeepSeek signals this cycle", self.Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"Trading cycle error: {e}", self.Fore.RED)
    
    def start_trading(self):
        """Start trading"""
        self.print_color("🚀 STARTING AI TRADER V6.0 WITH 3% AI CHECK SYSTEM!", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("💰 AI MANAGING $500 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"🎯 EXIT STRATEGY: EVERY 3% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"📊 Check Levels: {self.min_check_level}%, {self.min_check_level+3}%, {self.min_check_level+6}%, etc.", self.Fore.MAGENTA)
        self.print_color(f"⏰ Time Checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
        self.print_color(f"💰 Milestone Partials: 10%, 15%, 20%, etc.", self.Fore.GREEN)
        
        # Configuration options
        print("\n" + "="*60)
        print("3% AI Check Configuration:")
        print(f"1. Start checking at: {self.min_check_level}%")
        print(f"2. Check every: {self.percent_increment}%")
        print(f"3. Time checks every: {self.time_based_check_minutes} minutes")
        print(f"4. Force partial at milestones: {'ON' if self.force_partial_at_milestones else 'OFF'}")
        
        config_choice = input("\nConfigure settings? (y/N): ").strip().lower()
        if config_choice == 'y':
            try:
                min_level = input(f"Start checking at % (default {self.min_check_level}): ").strip()
                if min_level:
                    self.min_check_level = int(min_level)
                
                increment = input(f"Check every % (default {self.percent_increment}): ").strip()
                if increment:
                    self.percent_increment = int(increment)
                
                time_check = input(f"Time checks every minutes (default {self.time_based_check_minutes}): ").strip()
                if time_check:
                    self.time_based_check_minutes = int(time_check)
                
                milestone = input(f"Force partial at milestones? (y/N): ").strip().lower()
                self.force_partial_at_milestones = (milestone == 'y')
                
                self.print_color("✅ Configuration updated!", self.Fore.GREEN)
            except:
                self.print_color("⚠️ Invalid configuration, using defaults", self.Fore.YELLOW)
        
        self.cycle_count = 0
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🔄 TRADING CYCLE {self.cycle_count} (3% AI CHECK)", self.Fore.CYAN + self.Style.BRIGHT)
                self.print_color("=" * 60, self.Fore.CYAN)
                self.run_trading_cycle()
                self.print_color(f"⏳ Next analysis in 3 minutes...", self.Fore.BLUE)
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_trade_history(15)
                self.show_trading_stats()
                break
            except Exception as e:
                self.print_color(f"Main loop error: {e}", self.Fore.RED)
                time.sleep(self.monitoring_interval)


# ==================== PAPER TRADING CLASS ====================
class FullyAutonomous1HourPaperTrader:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        self.Fore = real_bot.Fore
        self.Back = real_bot.Back
        self.Style = real_bot.Style
        self.COLORAMA_AVAILABLE = real_bot.COLORAMA_AVAILABLE
        
        # Copy 3% system settings
        self.exit_strategy_mode = "3PERCENT_AI_CHECK"
        self.min_check_level = 6
        self.percent_increment = 3
        self.force_partial_at_milestones = True
        self.time_based_check_minutes = 15
        
        # Paper trading specific
        self.checked_3percent_levels = {}
        self.last_ai_check_time = {}
        
        # Paper trading settings
        self.monitoring_interval = 180
        self.paper_balance = 500
        self.available_budget = 500
        self.paper_positions = {}
        self.paper_history_file = "fully_autonomous_1hour_paper_trading_history.json"
        self.paper_history = self.load_paper_history()
        self.available_pairs = ["SOLUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "HYPEUSDT"]
        self.max_concurrent_trades = 6
        
        self.real_bot.print_color("🤖 FULLY AUTONOMOUS PAPER TRADER V6.0 INITIALIZED!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Virtual Budget: ${self.paper_balance}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color(f"🎯 EXIT STRATEGY: EVERY 3% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"📊 Check starts at: {self.min_check_level}%", self.Fore.MAGENTA)
    
    def load_paper_history(self):
        """Load paper trading history"""
        try:
            if os.path.exists(self.paper_history_file):
                with open(self.paper_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.real_bot.print_color(f"Error loading paper trade history: {e}", self.Fore.RED)
            return []
    
    def save_paper_history(self):
        """Save paper trading history"""
        try:
            with open(self.paper_history_file, 'w') as f:
                json.dump(self.paper_history, f, indent=2)
        except Exception as e:
            self.real_bot.print_color(f"Error saving paper trade history: {e}", self.Fore.RED)
    
    def add_paper_trade_to_history(self, trade_data):
        """Add trade to paper trading history"""
        try:
            trade_data['close_time'] = self.real_bot.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            trade_data['trade_type'] = 'PAPER'
            
            if 'exit_price' not in trade_data:
                current_price = self.real_bot.get_current_price(trade_data['pair'])
                trade_data['exit_price'] = current_price
            
            if 'peak_pnl_pct' not in trade_data:
                if 'peak_pnl' in trade_data:
                    trade_data['peak_pnl_pct'] = trade_data['peak_pnl']
                else:
                    if trade_data['direction'] == 'LONG':
                        peak_pct = ((trade_data['exit_price'] - trade_data['entry_price']) / 
                                   trade_data['entry_price']) * 100 * trade_data.get('leverage', 1)
                    else:
                        peak_pct = ((trade_data['entry_price'] - trade_data['exit_price']) / 
                                   trade_data['entry_price']) * 100 * trade_data.get('leverage', 1)
                    trade_data['peak_pnl_pct'] = max(0, peak_pct)
            
            if trade_data.get('partial_percent', 100) < 100:
                trade_data['display_type'] = f"PARTIAL_{trade_data['partial_percent']}%"
            else:
                trade_data['display_type'] = "FULL_CLOSE"
            
            self.paper_history.append(trade_data)
            
            if len(self.paper_history) > 200:
                self.paper_history = self.paper_history[-200:]
            self.save_paper_history()
            
            # Log for ML
            try:
                from data_collector import log_trade_for_ml
                log_trade_for_ml(trade_data)
            except:
                pass
            
            if trade_data.get('partial_percent', 100) < 100:
                self.real_bot.print_color(f"📝 PAPER Partial close: {trade_data['pair']} {trade_data['partial_percent']}% | P&L: ${trade_data.get('pnl', 0):.2f}", self.Fore.CYAN)
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
    
    def paper_check_3percent_level(self, pair, trade):
        """Paper version of 3% level check"""
        current_price = self.real_bot.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        # Track peak
        if 'peak_pnl' not in trade:
            trade['peak_pnl'] = current_pnl
        elif current_pnl > trade['peak_pnl']:
            trade['peak_pnl'] = current_pnl
        
        # Calculate current 3% level
        current_level = math.floor(current_pnl / self.percent_increment) * self.percent_increment
        
        # Skip if below minimum check level
        if current_level < self.min_check_level:
            return {"should_close": False}
        
        # Initialize checked levels for this pair
        if pair not in self.checked_3percent_levels:
            self.checked_3percent_levels[pair] = []
        
        # Check if this level needs evaluation
        if current_level not in self.checked_3percent_levels[pair]:
            self.real_bot.print_color(f"🎯 PAPER {pair} reached +{current_pnl:.1f}% (Level {current_level}%)", self.Fore.CYAN)
            
            # Add to checked levels
            self.checked_3percent_levels[pair].append(current_level)
            
            # Get AI decision at this level
            market_data = self.real_bot.get_price_history(pair)
            ai_decision = self.real_bot.get_ai_exit_decision_at_level(pair, trade, market_data, current_level, current_pnl)
            
            return ai_decision
        
        return {"should_close": False}
    
    def paper_check_milestone_partial(self, pair, trade):
        """Paper version of milestone partial"""
        current_price = self.real_bot.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if not self.force_partial_at_milestones:
            return {"should_close": False}
        
        milestone_levels = [10, 15, 20, 25, 30]
        
        for milestone in milestone_levels:
            if current_pnl >= milestone and current_pnl < milestone + 1:
                milestone_key = f"{pair}_milestone_{milestone}"
                if milestone_key not in trade:
                    trade[milestone_key] = True
                    
                    # Calculate partial percentage
                    if milestone >= 25:
                        partial_percent = 40
                    elif milestone >= 20:
                        partial_percent = 30
                    elif milestone >= 15:
                        partial_percent = 20
                    else:
                        partial_percent = 15
                    
                    return {
                        "should_close": True,
                        "partial_percent": partial_percent,
                        "close_type": f"PAPER_MILESTONE_{milestone}",
                        "reasoning": f"🎉 PAPER Milestone! Taking {partial_percent}% at +{milestone}%",
                        "confidence": 85
                    }
        
        return {"should_close": False}
    
    def paper_check_peak_drawdown_protection(self, pair, trade):
        """Paper version of drawdown protection"""
        current_price = self.real_bot.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if 'peak_pnl' not in trade:
            trade['peak_pnl'] = current_pnl
        
        peak = trade['peak_pnl']
        
        if peak < 6:
            return {"should_close": False}
        
        drawdown = peak - current_pnl
        
        if drawdown >= 6:
            return {
                "should_close": True,
                "partial_percent": 100,
                "close_type": "PAPER_PEAK_DRAWDOWN_6",
                "reasoning": f"🚨 PAPER Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 90
            }
        
        elif drawdown >= 4:
            return {
                "should_close": True,
                "partial_percent": 50,
                "close_type": "PAPER_PEAK_DRAWDOWN_4",
                "reasoning": f"⚠️ PAPER Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 80
            }
        
        elif drawdown >= 2 and peak >= 15:
            return {
                "should_close": True,
                "partial_percent": 30,
                "close_type": "PAPER_PEAK_DRAWDOWN_2",
                "reasoning": f"PAPER Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 75
            }
        
        return {"should_close": False}
    
    def paper_get_3percent_exit_decision(self, pair, trade):
        """Paper version of 3% exit decision"""
        
        # Check emergency stops
        current_pnl = self.calculate_current_pnl(trade, self.real_bot.get_current_price(pair))
        if current_pnl <= -5.0:
            return {
                "should_close": True,
                "partial_percent": 100,
                "close_type": "PAPER_EMERGENCY_STOP_5",
                "reasoning": f"🚨 PAPER Emergency stop at -{abs(current_pnl):.1f}%",
                "confidence": 100
            }
        
        # Check drawdown
        drawdown_decision = self.paper_check_peak_drawdown_protection(pair, trade)
        if drawdown_decision.get("should_close", False):
            return drawdown_decision
        
        # Check 3% level
        level_decision = self.paper_check_3percent_level(pair, trade)
        if level_decision.get("should_close", False):
            return level_decision
        
        # Check milestone
        milestone_decision = self.paper_check_milestone_partial(pair, trade)
        if milestone_decision.get("should_close", False):
            return milestone_decision
        
        # Time check (simplified for paper)
        current_time = time.time()
        last_check = self.last_ai_check_time.get(pair, 0)
        if current_time - last_check >= (self.time_based_check_minutes * 60):
            self.last_ai_check_time[pair] = current_time
            
            if current_pnl >= 10:
                # Small partial on time check
                return {
                    "should_close": True,
                    "partial_percent": 15,
                    "close_type": "PAPER_TIME_CHECK",
                    "reasoning": f"⏰ PAPER Time check at +{current_pnl:.1f}%",
                    "confidence": 70
                }
        
        return {"should_close": False}
    
    def paper_close_trade_immediately(self, pair, trade, close_reason="AI_DECISION", partial_percent=100):
        """Close paper trade immediately"""
        try:
            current_price = self.real_bot.get_current_price(pair)
            
            if trade['direction'] == 'LONG':
                pnl = (current_price - trade['entry_price']) * trade['quantity'] * (partial_percent / 100)
            else:
                pnl = (trade['entry_price'] - current_price) * trade['quantity'] * (partial_percent / 100)
            
            peak_pnl_pct = trade.get('peak_pnl', 0)
            
            if partial_percent < 100:
                remaining_quantity = trade['quantity'] * (1 - partial_percent / 100)
                closed_quantity = trade['quantity'] * (partial_percent / 100)
                closed_position_size = trade['position_size_usd'] * (partial_percent / 100)
                
                trade['quantity'] = remaining_quantity
                trade['position_size_usd'] = trade['position_size_usd'] * (1 - partial_percent / 100)
                
                partial_trade = trade.copy()
                partial_trade['status'] = 'PARTIAL_CLOSE'
                partial_trade['exit_price'] = current_price
                partial_trade['pnl'] = pnl
                partial_trade['close_reason'] = close_reason
                partial_trade['close_time'] = self.real_bot.get_thailand_time()
                partial_trade['partial_percent'] = partial_percent
                partial_trade['closed_quantity'] = closed_quantity
                partial_trade['closed_position_size'] = closed_position_size
                partial_trade['peak_pnl_pct'] = round(peak_pnl_pct, 3)
                
                self.available_budget += closed_position_size + pnl
                self.add_paper_trade_to_history(partial_trade)
                
                pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
                self.real_bot.print_color(f"✅ PAPER Partial Close | {pair} | {partial_percent}% | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                self.real_bot.print_color(f"📊 PAPER Remaining: {remaining_quantity:.4f} {pair} (${trade['position_size_usd']:.2f})", self.Fore.CYAN)
                
                return True
                
            else:
                trade['status'] = 'CLOSED'
                trade['exit_price'] = current_price
                trade['pnl'] = pnl
                trade['close_reason'] = close_reason
                trade['close_time'] = self.real_bot.get_thailand_time()
                trade['partial_percent'] = 100
                trade['peak_pnl_pct'] = round(peak_pnl_pct, 3)
                
                self.available_budget += trade['position_size_usd'] + pnl
                self.add_paper_trade_to_history(trade.copy())
                
                pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
                self.real_bot.print_color(f"✅ PAPER Full Close | {pair} | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                
                if pair in self.paper_positions:
                    del self.paper_positions[pair]
                
                return True
                
        except Exception as e:
            self.real_bot.print_color(f"❌ PAPER Close failed: {e}", self.Fore.RED)
            return False
    
    def paper_execute_trade(self, pair, ai_decision):
        """Execute paper trade"""
        try:
            decision = ai_decision["decision"]
            position_size_usd = ai_decision["position_size_usd"]
            entry_price = ai_decision["entry_price"]
            leverage = ai_decision["leverage"]
            confidence = ai_decision["confidence"]
            reasoning = ai_decision["reasoning"]
            
            if decision == "HOLD" or position_size_usd <= 0:
                self.real_bot.print_color(f"🟡 PAPER: DeepSeek decides to HOLD {pair}", self.Fore.YELLOW)
                return False
            
            if pair in self.paper_positions:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Position already exists", self.Fore.RED)
                return False
            
            if len(self.paper_positions) >= self.max_concurrent_trades:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Max concurrent trades reached (6)", self.Fore.RED)
                return False
                
            if position_size_usd > self.available_budget:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Insufficient budget", self.Fore.RED)
                return False
            
            notional_value = position_size_usd * leverage
            quantity = notional_value / entry_price
            quantity = round(quantity, 3)
            
            direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
            
            self.real_bot.print_color(f"\n🤖 PAPER TRADE EXECUTION (3% AI CHECK)", self.Fore.CYAN + self.Style.BRIGHT)
            self.real_bot.print_color("=" * 80, self.Fore.CYAN)
            self.real_bot.print_color(f"{direction_icon} {pair}", direction_color)
            self.real_bot.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.real_bot.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
            self.real_bot.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
            self.real_bot.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
            self.real_bot.print_color(f"🎯 EXIT STRATEGY: EVERY 3% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
            self.real_bot.print_color(f"📊 Check starts at: {self.min_check_level}%", self.Fore.MAGENTA)
            self.real_bot.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
            self.real_bot.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
            self.real_bot.print_color("=" * 80, self.Fore.CYAN)
            
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
                'has_tp_sl': False,
                'peak_pnl': 0
            }
            
            # Initialize checked levels
            self.checked_3percent_levels[pair] = []
            
            self.real_bot.print_color(f"✅ PAPER TRADE EXECUTED: {pair} {decision} | Leverage: {leverage}x", self.Fore.GREEN + self.Style.BRIGHT)
            return True
            
        except Exception as e:
            self.real_bot.print_color(f"❌ PAPER: Trade execution failed: {e}", self.Fore.RED)
            return False
    
    def monitor_paper_positions(self):
        """Monitor paper positions"""
        try:
            closed_positions = []
            
            for pair, trade in list(self.paper_positions.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                self.real_bot.print_color(f"🔍 PAPER 3% System Checking {pair}...", self.Fore.BLUE)
                
                # Get exit decision
                exit_decision = self.paper_get_3percent_exit_decision(pair, trade)
                
                if exit_decision.get("should_close", False):
                    close_type = exit_decision.get("close_type", "EXIT")
                    reasoning = exit_decision.get("reasoning", "No reason")
                    partial_percent = exit_decision.get("partial_percent", 100)
                    confidence = exit_decision.get("confidence", 0)
                    
                    self.real_bot.print_color(f"🎯 PAPER 3% System Decision for {pair}:", self.Fore.CYAN + self.Style.BRIGHT)
                    self.real_bot.print_color(f"   Action: {'PARTIAL' if partial_percent < 100 else 'FULL'} CLOSE", self.Fore.YELLOW)
                    self.real_bot.print_color(f"   Type: {close_type}", self.Fore.MAGENTA)
                    self.real_bot.print_color(f"   Confidence: {confidence}%", self.Fore.GREEN if confidence > 70 else self.Fore.YELLOW)
                    self.real_bot.print_color(f"   Reason: {reasoning}", self.Fore.WHITE)
                    
                    success = self.paper_close_trade_immediately(pair, trade, f"PAPER_{close_type}: {reasoning}", partial_percent)
                    if success and partial_percent == 100:
                        closed_positions.append(pair)
                        
                        # Clean up
                        if pair in self.checked_3percent_levels:
                            del self.checked_3percent_levels[pair]
            
            return closed_positions
                    
        except Exception as e:
            self.real_bot.print_color(f"PAPER Monitoring error: {e}", self.Fore.RED)
            return []
    
    def display_paper_dashboard(self):
        """Display paper trading dashboard"""
        self.real_bot.print_color(f"\n🤖 PAPER TRADING DASHBOARD - {self.real_bot.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 90, self.Fore.CYAN)
        self.real_bot.print_color(f"🎯 EXIT STRATEGY: EVERY 3% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"📊 Check Levels: {self.min_check_level}%, {self.min_check_level+3}%, {self.min_check_level+6}%, etc.", self.Fore.MAGENTA)
        self.real_bot.print_color(f"⏰ Time Checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
        self.real_bot.print_color(f"💰 Milestone Partials: 10%, 15%, 20%, etc.", self.Fore.GREEN)
        
        active_count = 0
        total_unrealized = 0
        
        for pair, trade in self.paper_positions.items():
            if trade['status'] == 'ACTIVE':
                active_count += 1
                current_price = self.real_bot.get_current_price(pair)
                
                direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                current_pnl = self.calculate_current_pnl(trade, current_price)
                
                if trade['direction'] == 'LONG':
                    unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:
                    unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    
                total_unrealized += unrealized_pnl
                pnl_color = self.Fore.GREEN + self.Style.BRIGHT if unrealized_pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
                
                current_level = math.floor(current_pnl / self.percent_increment) * self.percent_increment
                next_level = current_level + self.percent_increment
                
                self.real_bot.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
                self.real_bot.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x ⚡", self.Fore.WHITE)
                self.real_bot.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
                self.real_bot.print_color(f"   P&L: ${unrealized_pnl:.2f} ({current_pnl:.1f}%)", pnl_color)
                self.real_bot.print_color(f"   📈 Next Check: +{next_level}%", self.Fore.CYAN)
                
                if 'peak_pnl' in trade:
                    peak = trade['peak_pnl']
                    self.real_bot.print_color(f"   🏔️ Peak: {peak:.1f}% | Drawdown: {max(0, peak - current_pnl):.1f}%", 
                                           self.Fore.YELLOW if peak - current_pnl <= 2 else self.Fore.RED)
                
                self.real_bot.print_color("   " + "-" * 60, self.Fore.CYAN)
        
        if active_count == 0:
            self.real_bot.print_color("No active paper positions", self.Fore.YELLOW)
        else:
            total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
            self.real_bot.print_color(f"📊 Active Paper Positions: {active_count}/{self.max_concurrent_trades} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)
        
        self.real_bot.print_color(f"💰 Paper Balance: ${self.paper_balance:.2f} | Available: ${self.available_budget:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
    
    def show_paper_history(self, limit=10):
        """Show paper trading history"""
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
            
            display_type = trade.get('display_type', 'FULL_CLOSE')
            if display_type.startswith('PARTIAL'):
                type_indicator = f" | {display_type}"
            else:
                type_indicator = " | FULL"
            
            self.real_bot.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']}{type_indicator}", pnl_color)
            self.real_bot.print_color(f"     Size: ${position_size:.2f} | Leverage: {leverage}x | P&L: ${pnl:.2f}", pnl_color)
            self.real_bot.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)
            
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
            self.monitor_paper_positions()
            self.display_paper_dashboard()
            
            if hasattr(self, 'paper_cycle_count') and self.paper_cycle_count % 4 == 0:
                self.show_paper_history(8)
                self.show_paper_stats()
            
            self.real_bot.print_color(f"\nPAPER: DEEPSEEK SCANNING {len(self.available_pairs)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.available_pairs:
                if self.available_budget > 100:
                    market_data = self.real_bot.get_price_history(pair)
                    
                    ai_decision = self.real_bot.get_ai_trading_decision(pair, market_data)
                    
                    if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] > 0:
                        qualified_signals += 1
                        direction = ai_decision['decision']
                        leverage_info = f"Leverage: {ai_decision['leverage']}x"
                        
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
        self.real_bot.print_color("🚀 STARTING PAPER TRADING V6.0 WITH 3% AI CHECK SYSTEM!", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("💰 VIRTUAL $500 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"🎯 EXIT STRATEGY: EVERY 3% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"📊 Check starts at: {self.min_check_level}%", self.Fore.MAGENTA)
        self.real_bot.print_color(f"⏰ Time checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
        
        # Configuration for paper trading
        print("\n" + "="*60)
        print("PAPER 3% AI Check Configuration:")
        print(f"1. Start checking at: {self.min_check_level}%")
        print(f"2. Check every: {self.percent_increment}%")
        print(f"3. Time checks every: {self.time_based_check_minutes} minutes")
        
        config_choice = input("\nConfigure paper settings? (y/N): ").strip().lower()
        if config_choice == 'y':
            try:
                min_level = input(f"Start checking at % (default {self.min_check_level}): ").strip()
                if min_level:
                    self.min_check_level = int(min_level)
                
                increment = input(f"Check every % (default {self.percent_increment}): ").strip()
                if increment:
                    self.percent_increment = int(increment)
                
                time_check = input(f"Time checks every minutes (default {self.time_based_check_minutes}): ").strip()
                if time_check:
                    self.time_based_check_minutes = int(time_check)
                
                self.real_bot.print_color("✅ PAPER Configuration updated!", self.Fore.GREEN)
            except:
                self.real_bot.print_color("⚠️ Invalid configuration, using defaults", self.Fore.YELLOW)
        
        self.paper_cycle_count = 0
        while True:
            try:
                self.paper_cycle_count += 1
                self.real_bot.print_color(f"\n🔄 PAPER TRADING CYCLE {self.paper_cycle_count} (3% AI CHECK)", self.Fore.CYAN + self.Style.BRIGHT)
                self.real_bot.print_color("=" * 60, self.Fore.CYAN)
                self.run_paper_trading_cycle()
                self.real_bot.print_color(f"⏳ Next paper analysis in 3 minutes...", self.Fore.BLUE)
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                self.real_bot.print_color(f"\n🛑 PAPER TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_paper_history(15)
                self.show_paper_stats()
                break
            except Exception as e:
                self.real_bot.print_color(f"PAPER: Main loop error: {e}", self.Fore.RED)
                time.sleep(self.monitoring_interval)


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    try:
        bot = FullyAutonomous1HourAITrader()
        
        print("\n" + "="*70)
        print("🤖 FULLY AUTONOMOUS 1-HOUR AI TRADER V6.0")
        print("EVERY 3% AI CHECK SYSTEM")
        print("="*70)
        print("1. 🎯 REAL TRADING (Live Binance Account)")
        print("2. 📝 PAPER TRADING (Virtual Simulation)")
        print("3. ❌ EXIT")
        
        choice = input("\nSelect mode (1-3): ").strip()
        
        if choice == "1":
            if bot.binance:
                bot.start_trading()
            else:
                print(f"\n❌ Binance connection failed. Switching to paper trading...")
                paper_bot = FullyAutonomous1HourPaperTrader(bot)
                paper_bot.start_paper_trading()
                
        elif choice == "2":
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
