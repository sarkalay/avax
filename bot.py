"""
FIXED TRADING BOT - Correct Partial Close System
Binance Cross Mode + $25 Min Position + Real Partial Close Fix
"""

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
from collections import Counter

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

# ==================== CONFIGURATION FILE ====================
CONFIG_FILE = "config_3percent.json"
DEFAULT_CONFIG = {
    "exit_strategy": "3PERCENT_AI_CHECK",
    "min_check_level": 6,
    "percent_increment": 3,
    "force_milestone_partials": True,
    "milestone_levels": [10, 15, 20, 25, 30],
    "time_check_minutes": 15,
    "emergency_stop": -5,
    "min_position_size": 25,  # ✅ $25 minimum position
    "position_mode": "CROSS",  # ✅ CROSS mode
    "drawdown_protection": {
        "from_peak_6": 6,
        "from_peak_4": 4,
        "from_peak_2": 2
    },
    "paper_trading": {
        "virtual_balance": 500,
        "max_positions": 6
    },
    "auto_calibration": {
        "enabled": True,
        "volatility_threshold_high": 2.0,
        "volatility_threshold_low": 1.0
    }
}

# ==================== FIXED PERFORMANCE ANALYTICS ====================
class PerformanceAnalytics:
    def __init__(self):
        self.analytics_file = "trading_analytics.json"
        self.analytics_data = self.load_analytics()
    
    def load_analytics(self):
        """Load analytics data"""
        try:
            if os.path.exists(self.analytics_file):
                with open(self.analytics_file, 'r') as f:
                    return json.load(f)
            return {
                "level_decisions": [],
                "trade_history": [],
                "calibration_history": []
            }
        except Exception as e:
            print(f"Error loading analytics: {e}")
            return {
                "level_decisions": [],
                "trade_history": [],
                "calibration_history": []
            }
    
    def save_analytics(self):
        """Save analytics data"""
        try:
            with open(self.analytics_file, 'w') as f:
                json.dump(self.analytics_data, f, indent=2)
        except Exception as e:
            print(f"Error saving analytics: {e}")
    
    def log_level_decision(self, pair, level, ai_decision, result_pnl):
        """Log every 3% level decision - FIXED VERSION"""
        # Determine action from decision
        action = "UNKNOWN"
        
        if "action" in ai_decision:
            action = ai_decision["action"]
        elif "close_type" in ai_decision:
            close_type = ai_decision["close_type"]
            if "FALLBACK" in close_type or "MILESTONE" in close_type or "TIME" in close_type:
                action = "TAKE_PARTIAL"
            elif "DRAWDOWN" in close_type or "EMERGENCY" in close_type:
                action = "CLOSE_FULL"
            elif "AI_FULL" in close_type:
                action = "CLOSE_FULL"
            elif "AI_" in close_type and "LEVEL" in close_type:
                action = "TAKE_PARTIAL"
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "level": level,
            "ai_decision": action,
            "partial_percent": ai_decision.get("partial_percent", 0),
            "result_pnl": result_pnl,
            "confidence": ai_decision.get("confidence", 0),
            "close_type": ai_decision.get("close_type", "")
        }
        
        if "level_decisions" not in self.analytics_data:
            self.analytics_data["level_decisions"] = []
        
        self.analytics_data["level_decisions"].append(entry)
        
        # Keep only last 1000 entries
        if len(self.analytics_data["level_decisions"]) > 1000:
            self.analytics_data["level_decisions"] = self.analytics_data["level_decisions"][-1000:]
        
        self.save_analytics()
    
    def analyze_best_levels(self):
        """Which 3% levels give best results?"""
        if "level_decisions" not in self.analytics_data:
            return {}
        
        level_stats = {}
        for entry in self.analytics_data["level_decisions"]:
            level = entry["level"]
            if level not in level_stats:
                level_stats[level] = {
                    "count": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                    "decisions": {"HOLD_NEXT_LEVEL": 0, "TAKE_PARTIAL": 0, "CLOSE_FULL": 0}
                }
            
            level_stats[level]["count"] += 1
            level_stats[level]["total_pnl"] += entry["result_pnl"]
            
            # Map decisions properly
            ai_decision = entry["ai_decision"]
            if ai_decision == "HOLD" or ai_decision == "HOLD_NEXT_LEVEL":
                level_stats[level]["decisions"]["HOLD_NEXT_LEVEL"] += 1
            elif ai_decision == "TAKE_PARTIAL":
                level_stats[level]["decisions"]["TAKE_PARTIAL"] += 1
            elif ai_decision == "CLOSE_FULL":
                level_stats[level]["decisions"]["CLOSE_FULL"] += 1
            else:
                # Try to infer from partial_percent
                if entry.get("partial_percent", 0) > 0 and entry.get("partial_percent", 0) < 100:
                    level_stats[level]["decisions"]["TAKE_PARTIAL"] += 1
                elif entry.get("partial_percent", 0) == 100:
                    level_stats[level]["decisions"]["CLOSE_FULL"] += 1
                else:
                    level_stats[level]["decisions"]["HOLD_NEXT_LEVEL"] += 1
        
        # Calculate averages
        for level, stats in level_stats.items():
            if stats["count"] > 0:
                stats["avg_pnl"] = stats["total_pnl"] / stats["count"]
        
        return level_stats
    
    def show_analytics_dashboard(self):
        """Show analytics dashboard"""
        stats = self.analyze_best_levels()
        
        print("\n📊 3% LEVEL ANALYTICS (FIXED)")
        print("=" * 80)
        
        if not stats:
            print("No analytics data available yet.")
            return
        
        print(f"{'Level':<10} {'Count':<8} {'Avg P&L':<12} {'HOLD':<8} {'PARTIAL':<10} {'FULL':<8}")
        print("-" * 80)
        
        for level in sorted(stats.keys()):
            s = stats[level]
            if s["count"] >= 1:  # Changed from 3 to 1 to show all data
                print(f"+{level}%:    {s['count']:<8} ${s['avg_pnl']:<11.2f} "
                      f"{s['decisions'].get('HOLD_NEXT_LEVEL', 0):<8} "
                      f"{s['decisions'].get('TAKE_PARTIAL', 0):<10} "
                      f"{s['decisions'].get('CLOSE_FULL', 0):<8}")

# ==================== FIXED AUTO-CALIBRATING SYSTEM ====================
class AutoCalibrating3PercentSystem:
    def __init__(self, config):
        self.performance_history = []
        self.config = config
        self.optimal_increment = config.get("percent_increment", 3)
        self.optimal_start_level = config.get("min_check_level", 6)
        self.calibration_history = []
    
    def calibrate_based_on_market(self, volatility):
        """Market volatility အလိုက် auto calibrate"""
        auto_calibration = self.config.get("auto_calibration", {})
        
        if not auto_calibration.get("enabled", False):
            return
        
        volatility_high = auto_calibration.get("volatility_threshold_high", 2.0)
        volatility_low = auto_calibration.get("volatility_threshold_low", 1.0)
        
        # High volatility = wider increments
        # Low volatility = tighter increments
        
        if volatility > volatility_high:  # High volatility
            self.optimal_increment = 4  # Every 4%
            self.optimal_start_level = 8  # Start at 8%
            calibration_type = "HIGH_VOLATILITY"
        elif volatility > volatility_low:  # Medium volatility
            self.optimal_increment = 3  # Every 3%
            self.optimal_start_level = 6  # Start at 6%
            calibration_type = "MEDIUM_VOLATILITY"
        else:  # Low volatility
            self.optimal_increment = 2  # Every 2%
            self.optimal_start_level = 4  # Start at 4%
            calibration_type = "LOW_VOLATILITY"
        
        # Log calibration
        self.calibration_history.append({
            "timestamp": datetime.now().isoformat(),
            "volatility": volatility,
            "increment": self.optimal_increment,
            "start_level": self.optimal_start_level,
            "type": calibration_type
        })
        
        # Keep only last 100 calibrations
        if len(self.calibration_history) > 100:
            self.calibration_history = self.calibration_history[-100:]
    
    def calibrate_based_on_performance(self):
        """Past performance အလိုက် calibrate"""
        if len(self.performance_history) < 10:
            return
        
        # Analyze which levels worked best
        successful_levels = []
        for trade in self.performance_history:
            if trade.get('pnl', 0) > 0:
                # Which levels were hit in profitable trades?
                for level in trade.get('levels_hit', []):
                    successful_levels.append(level)
        
        if successful_levels:
            # Find most common successful level
            level_counts = Counter(successful_levels)
            most_common_level = level_counts.most_common(1)[0][0]
            
            # Adjust start level
            self.optimal_start_level = most_common_level - 3
            if self.optimal_start_level < 3:
                self.optimal_start_level = 3
            
            # Log calibration
            self.calibration_history.append({
                "timestamp": datetime.now().isoformat(),
                "most_common_level": most_common_level,
                "start_level": self.optimal_start_level,
                "increment": self.optimal_increment,
                "type": "PERFORMANCE_BASED"
            })
            
            if len(self.calibration_history) > 100:
                self.calibration_history = self.calibration_history[-100:]

# ==================== FIXED V6.0 EVERY 3% AI CHECK SYSTEM ====================
class FullyAutonomous1HourAITrader:
    def __init__(self):
        # Load configuration
        self.config = self.load_config()
        
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
        
        # Performance Analytics
        self.analytics = PerformanceAnalytics()
        
        # Auto-calibration
        self.calibrator = AutoCalibrating3PercentSystem(self.config)
        
        # 3% Increment System
        self.checked_3percent_levels = {}  # {pair: [checked_levels]}
        self.last_ai_check_time = {}
        
    def load_config(self):
        """Load configuration from file or create default"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    print("✅ Configuration loaded from file")
                    return config
            else:
                # Create default config file
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
                print("✅ Default configuration created")
                return DEFAULT_CONFIG
        except Exception as e:
            print(f"❌ Error loading config: {e}, using defaults")
            return DEFAULT_CONFIG
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
            print("✅ Configuration saved")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def update_config(self, updates):
        """Update configuration"""
        self.config.update(updates)
        self.save_config()
        
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
        self.max_position_size_percent = 6

        self.max_concurrent_trades = 4
        
        # ✅ NEW: Minimum position size from config
        self.min_position_size = self.config.get("min_position_size", 25)  # $25 minimum
        
        # AI can trade selected 3 major pairs only
        self.available_pairs = [
            "SOLUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "LINKUSDT"
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
        
        # 🆕 EVERY 3% AI CHECK SETTINGS FROM CONFIG
        self.exit_strategy_mode = self.config.get("exit_strategy", "3PERCENT_AI_CHECK")
        self.min_check_level = self.config.get("min_check_level", 6)
        self.percent_increment = self.config.get("percent_increment", 3)
        self.force_partial_at_milestones = self.config.get("force_milestone_partials", True)
        self.milestone_levels = self.config.get("milestone_levels", [10, 15, 20, 25, 30])
        self.time_based_check_minutes = self.config.get("time_check_minutes", 15)
        self.emergency_stop = self.config.get("emergency_stop", -5)
        
        # Drawdown protection settings
        self.drawdown_protection = self.config.get("drawdown_protection", {
            "from_peak_8": 8,
            "from_peak_6": 6,
            "from_peak_4": 4
        })
        
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
            print(f"🎯 EXIT STRATEGY: EVERY {self.percent_increment}% AI CHECK")
            print(f"📊 Check starts at: {self.min_check_level}%")
            print(f"✅ MINIMUM POSITION SIZE: ${self.min_position_size}")
            print(f"✅ POSITION MODE: {self.config.get('position_mode', 'CROSS')}")
            print(f"⏰ Additional checks: Every {self.time_based_check_minutes} minutes")
            print(f"📈 Force partial at milestones: {'ON' if self.force_partial_at_milestones else 'OFF'}")
        except Exception as e:
            print(f"Binance initialization failed: {e}")
            self.binance = None
        
        self.validate_config()
        if self.binance:
            self.setup_futures()
            self.load_symbol_precision()
    
    # ==================== FIXED SETUP FUTURES ====================
    def setup_futures(self):
        """Setup futures trading with error handling"""
        if not self.binance:
            self.print_color("⚠️ Binance client not available - paper trading only", self.Fore.YELLOW)
            return
        
        try:
            self.print_color("🔄 Setting up futures trading...", self.Fore.BLUE)
            
            # Get position mode from config
            position_mode = self.config.get("position_mode", "CROSS")
            margin_type = 'CROSS' if position_mode == "CROSS" else 'ISOLATED'
            
            # 1. Set position mode (Hedge/One-way)
            try:
                # Get current position mode
                current_mode = self.binance.futures_get_position_mode()
                is_dual_mode = current_mode.get('dualSidePosition', False)
                
                # Determine if we need to change
                need_dual = (position_mode == "ISOLATED")
                
                if is_dual_mode != need_dual:
                    self.binance.futures_change_position_mode(dualSidePosition=need_dual)
                    self.print_color(f"✅ Position mode changed to {position_mode}", self.Fore.GREEN)
                else:
                    self.print_color(f"✅ Position mode already {position_mode}", self.Fore.GREEN)
            except Exception as e:
                if "No need to change" in str(e):
                    self.print_color(f"✅ Position mode already set", self.Fore.GREEN)
                else:
                    self.print_color(f"⚠️ Position mode setup: {str(e)[:100]}", self.Fore.YELLOW)
            
            # 2. Setup each trading pair
            successful_setups = 0
            for pair in self.available_pairs:
                try:
                    # Set leverage
                    self.binance.futures_change_leverage(symbol=pair, leverage=5)
                    
                    # Set margin type with multiple attempts
                    margin_set = False
                    
                    # Try different parameter names
                    param_variations = [
                        {'symbol': pair, 'marginType': margin_type},
                        {'symbol': pair, 'margin_type': margin_type},
                        {'symbol': pair, 'margintype': margin_type}
                    ]
                    
                    for params in param_variations:
                        try:
                            self.binance.futures_change_margin_type(**params)
                            margin_set = True
                            break
                        except:
                            continue
                    
                    if margin_set:
                        successful_setups += 1
                        self.print_color(f"✅ {pair}: 5x leverage, {margin_type} margin", self.Fore.GREEN)
                    else:
                        self.print_color(f"⚠️ {pair}: Margin type setup failed (but continuing)", self.Fore.YELLOW)
                        
                except Exception as e:
                    error_msg = str(e)
                    if "No need to change" in error_msg or "already set" in error_msg.lower():
                        successful_setups += 1
                        self.print_color(f"✅ {pair}: Already setup correctly", self.Fore.GREEN)
                    else:
                        self.print_color(f"⚠️ {pair}: {error_msg[:80]}", self.Fore.YELLOW)
            
            if successful_setups > 0:
                self.print_color(f"✅ Futures setup: {successful_setups}/{len(self.available_pairs)} pairs ready", 
                               self.Fore.GREEN + self.Style.BRIGHT)
            else:
                self.print_color("⚠️ Futures setup had issues but continuing...", self.Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"⚠️ Futures setup error (continuing anyway): {e}", self.Fore.YELLOW)
    
    # ==================== FIXED ERROR HANDLING ====================
    def robust_ai_exit_decision(self, pair, trade, market_data, current_level):
        """ပိုပြီး robust ဖြစ်တဲ့ AI decision - FIXED"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try main AI
                decision = self.get_ai_exit_decision_at_level(pair, trade, market_data, current_level)
                # Ensure action key exists
                if "action" not in decision:
                    if decision.get("should_close", False):
                        if decision.get("partial_percent", 100) < 100:
                            decision["action"] = "TAKE_PARTIAL"
                        else:
                            decision["action"] = "CLOSE_FULL"
                    else:
                        decision["action"] = "HOLD_NEXT_LEVEL"
                return decision
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return self.get_smart_fallback_decision(pair, trade, current_level)
            except Exception as e:
                self.print_color(f"AI Error (attempt {attempt+1}): {e}", self.Fore.YELLOW)
                time.sleep(1)
        
        return self.get_smart_fallback_decision(pair, trade, current_level)
    
    def get_smart_fallback_decision(self, pair, trade, current_level):
        """AI fail ရင် smart fallback - FIXED"""
        
        # Progressive fallback based on level
        fallback_rules = {
            6: {"partial": 10, "reason": "First profit level"},
            9: {"partial": 15, "reason": "Initial profit target"},
            12: {"partial": 20, "reason": "Good profit level"},
            15: {"partial": 25, "reason": "Strong profit level"},
            18: {"partial": 30, "reason": "Very good profit"},
            21: {"partial": 35, "reason": "Excellent profit"},
            24: {"partial": 40, "reason": "Outstanding profit"},
            27: {"partial": 50, "reason": "Amazing profit"},
            30: {"partial": 60, "reason": "Maximum profit level"}
        }
        
        if current_level in fallback_rules:
            rule = fallback_rules[current_level]
            return {
                "should_close": True,
                "action": "TAKE_PARTIAL",  # ✅ Added action key
                "partial_percent": rule["partial"],
                "close_type": f"FALLBACK_LEVEL_{current_level}",
                "reasoning": f"AI failed: {rule['reason']} at +{current_level}%",
                "confidence": 75
            }
        
        # Default fallback
        return {
            "should_close": True,
            "action": "TAKE_PARTIAL",  # ✅ Added action key
            "partial_percent": min(30, current_level * 2),
            "close_type": "DEFAULT_FALLBACK",
            "reasoning": f"Taking profit at +{current_level}% (AI unavailable)",
            "confidence": 70
        }
    
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
            "LTCUSDT": 85.0, "LINKUSDT": 13.
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
    
    def calculate_market_volatility(self, pair):
        """Calculate market volatility for auto-calibration"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': pair,
                'interval': '1h',
                'limit': 24  # Last 24 hours
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                klines = response.json()
                closes = [float(k[4]) for k in klines]
                
                # Calculate daily volatility (standard deviation of returns)
                returns = []
                for i in range(1, len(closes)):
                    ret = (closes[i] - closes[i-1]) / closes[i-1]
                    returns.append(ret)
                
                if returns:
                    volatility = np.std(returns) * 100  # Convert to percentage
                    return volatility
                
        except Exception as e:
            self.print_color(f"Volatility calculation error: {e}", self.Fore.YELLOW)
        
        return 1.0  # Default medium volatility
    
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
    
    # ==================== FIXED AI DECISION MAKING ====================
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
1. 1H and 4H trend must be identical (both bullish or both bearish)
2. On 15-minute chart: EMA9 must have cleanly crossed EMA34 at least 3 candles ago (not just touching)
3. 15-minute volume must be ≥ 2.8× the average of the last 30 candles
4. 15-minute RSI:
   - LONG  → between 48 and 70
   - SHORT → between 30 and 52
5. Current price must be above 1H EMA200 (for LONG) or below 1H EMA200 (for SHORT)
6. No high-impact news within the last 30 minutes and next 30 minutes
7. Your own confidence must be 85 or higher. If confidence < 85 → HOLD
8. If you are not 100% sure → answer HOLD only
   Missing 10 good trades is better than being wrong once.
9. MINIMUM POSITION SIZE: $25 (important!)
10. Position size: 5-7% of budget (min $25)
11. Leverage: 2-5x based on volatility
12. NO TP/SL - AI will close manually

Return JSON:
{
    "decision": "LONG" | "SHORT" | "HOLD" | "REVERSE_LONG" | "REVERSE_SHORT",
    "position_size_usd": number (minimum $25),
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
                        {"role": "system", "content": "You are a fully autonomous AI trader with reverse position capability. Minimum position size is $25."},
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
    
    def get_ai_exit_decision_at_level(self, pair, trade, market_data, current_level):
        """Ask AI for exit decision at specific 3% level - FIXED"""
        try:
            if not self.openrouter_key:
                decision = self.get_fallback_exit_decision_at_level(pair, trade, current_level)
                # Ensure action key exists
                if "action" not in decision:
                    if decision.get("should_close", False):
                        if decision.get("partial_percent", 100) < 100:
                            decision["action"] = "TAKE_PARTIAL"
                        else:
                            decision["action"] = "CLOSE_FULL"
                    else:
                        decision["action"] = "HOLD_NEXT_LEVEL"
                return decision
            
            current_price = market_data['current_price']
            current_pnl = self.calculate_current_pnl(trade, current_price)
            
            prompt = f"""
SPECIFIC {self.percent_increment}% LEVEL CHECK: {pair} reached +{current_pnl:.1f}% (Level {current_level}%)

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
1. HOLD_NEXT_LEVEL - Wait for next level (+{current_level + self.percent_increment}%)
2. TAKE_PARTIAL - How much % to take? (e.g., 20 for 20%)
3. CLOSE_FULL - Take all profit now

CONSIDER:
- Next {self.percent_increment}% level is at +{current_level + self.percent_increment}%
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
                    {"role": "system", "content": f"You are an AI trader making specific exit decisions at each {self.percent_increment}% profit level. Be precise about partial profit percentages."},
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
                
                # Parse AI response
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    decision_data = json.loads(json_str)
                    
                    action = decision_data.get('action', 'HOLD_NEXT_LEVEL').upper()
                    confidence = decision_data.get('confidence', 50)
                    reasoning = decision_data.get('reasoning', 'No reason')
                    
                    # Ensure action is one of the expected values
                    if action not in ['HOLD_NEXT_LEVEL', 'TAKE_PARTIAL', 'CLOSE_FULL']:
                        if 'HOLD' in action:
                            action = 'HOLD_NEXT_LEVEL'
                        elif 'PARTIAL' in action or 'TAKE' in action:
                            action = 'TAKE_PARTIAL'
                        elif 'CLOSE' in action or 'FULL' in action:
                            action = 'CLOSE_FULL'
                        else:
                            action = 'HOLD_NEXT_LEVEL'
                    
                    if action == 'HOLD_NEXT_LEVEL':
                        result_decision = {
                            "should_close": False,
                            "action": "HOLD_NEXT_LEVEL",
                            "next_level": decision_data.get('next_check_at', current_level + self.percent_increment),
                            "reasoning": reasoning,
                            "confidence": confidence
                        }
                    
                    elif action == 'TAKE_PARTIAL':
                        partial_percent = decision_data.get('partial_percent', 30)
                        
                        # Log to analytics
                        self.analytics.log_level_decision(pair, current_level, decision_data, 0)
                        
                        result_decision = {
                            "should_close": True,
                            "action": "TAKE_PARTIAL",
                            "partial_percent": partial_percent,
                            "close_type": f"AI_{self.percent_increment}PERCENT_LEVEL_{current_level}",
                            "reasoning": reasoning,
                            "confidence": confidence
                        }
                    
                    elif action == 'CLOSE_FULL':
                        # Log to analytics
                        self.analytics.log_level_decision(pair, current_level, decision_data, 0)
                        
                        result_decision = {
                            "should_close": True,
                            "action": "CLOSE_FULL",
                            "partial_percent": 100,
                            "close_type": f"AI_FULL_AT_{current_level}",
                            "reasoning": reasoning,
                            "confidence": confidence
                        }
                    
                    return result_decision
                    
            # If AI fails, use fallback
            return self.get_fallback_exit_decision_at_level(pair, trade, current_level)
                
        except Exception as e:
            self.print_color(f"AI Level Exit decision failed: {e}", self.Fore.YELLOW)
            return self.get_fallback_exit_decision_at_level(pair, trade, current_level)
    
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
                
                # ✅ ENFORCE MINIMUM POSITION SIZE
                if position_size_usd < self.min_position_size:
                    position_size_usd = self.min_position_size
                    self.print_color(f"⚠️ Position size increased to minimum ${self.min_position_size}", self.Fore.YELLOW)
                
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
                "position_size_usd": max(25, self.min_position_size),  # ✅ Minimum $25
                "entry_price": current_price,
                "leverage": 5,
                "confidence": 60,
                "reasoning": f"Fallback: Bullish signals ({bullish_signals}/{bearish_signals})",
                "should_reverse": False
            }
        elif bearish_signals >= 3 and bullish_signals <= 1:
            return {
                "decision": "SHORT", 
                "position_size_usd": max(25, self.min_position_size),  # ✅ Minimum $25
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
    
    def get_fallback_exit_decision_at_level(self, pair, trade, current_level):
        """Fallback exit decision at specific level - FIXED"""
        # Progressive partial closing based on level
        if current_level >= 24:
            partial_percent = 70
            action_type = "TAKE_PARTIAL"
        elif current_level >= 21:
            partial_percent = 60
            action_type = "TAKE_PARTIAL"
        elif current_level >= 18:
            partial_percent = 50
            action_type = "TAKE_PARTIAL"
        elif current_level >= 15:
            partial_percent = 40
            action_type = "TAKE_PARTIAL"
        elif current_level >= 12:
            partial_percent = 30
            action_type = "TAKE_PARTIAL"
        elif current_level >= 9:
            partial_percent = 20
            action_type = "TAKE_PARTIAL"
        elif current_level >= 6:
            partial_percent = 10
            action_type = "TAKE_PARTIAL"
        else:
            partial_percent = 0
            action_type = "HOLD_NEXT_LEVEL"
        
        if partial_percent > 0:
            return {
                "should_close": True,
                "action": action_type,
                "partial_percent": partial_percent,
                "close_type": f"FALLBACK_LEVEL_{current_level}",
                "reasoning": f"Fallback: Taking {partial_percent}% profit at +{current_level}%",
                "confidence": 70
            }
        
        return {
            "should_close": False,
            "action": "HOLD_NEXT_LEVEL",
            "reasoning": f"Fallback: Holding at +{current_level}%",
            "confidence": 60
        }
    
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
            return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
        
        # Initialize checked levels for this pair
        if pair not in self.checked_3percent_levels:
            self.checked_3percent_levels[pair] = []
        
        # Check if this level needs AI evaluation
        if current_level not in self.checked_3percent_levels[pair]:
            self.print_color(f"🎯 {pair} reached +{current_pnl:.1f}% (Level {current_level}%)", self.Fore.CYAN + self.Style.BRIGHT)
            
            # Add to checked levels
            self.checked_3percent_levels[pair].append(current_level)
            
            # Ask AI for decision at this level (with robust error handling)
            market_data = self.get_price_history(pair)
            ai_decision = self.robust_ai_exit_decision(pair, trade, market_data, current_level)
            
            return ai_decision
        
        return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
    
    def check_milestone_partial(self, pair, trade):
        """Force partial close at milestone levels (10%, 15%, 20%, etc)"""
        current_price = self.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if not self.force_partial_at_milestones:
            return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
        
        for milestone in self.milestone_levels:
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
                        "action": "TAKE_PARTIAL",
                        "partial_percent": partial_percent,
                        "close_type": f"MILESTONE_{milestone}_PARTIAL",
                        "reasoning": f"🎉 Milestone! Taking {partial_percent}% profit at +{milestone}%",
                        "confidence": 85
                    }
        
        return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
    
    def check_time_based_exit(self, pair, trade):
        """Check exit based on time"""
        current_time = time.time()
        entry_time = trade.get('entry_time', current_time)
        
        # Calculate hours in trade
        hours_in_trade = (current_time - entry_time) / 3600
        
        # Check every X minutes (from config)
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
    "action": "HOLD_NEXT_LEVEL" | "TAKE_PARTIAL" | "CLOSE_FULL",
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
                            
                            action = decision_data.get('action', 'HOLD_NEXT_LEVEL')
                            
                            if action == 'TAKE_PARTIAL':
                                partial_percent = decision_data.get('partial_percent', 20)
                                return {
                                    "should_close": True,
                                    "action": "TAKE_PARTIAL",
                                    "partial_percent": partial_percent,
                                    "close_type": "TIME_BASED_PARTIAL",
                                    "reasoning": f"⏰ Time check: {decision_data.get('reasoning', '')}",
                                    "confidence": 75
                                }
                            
                            elif action == 'CLOSE_FULL':
                                return {
                                    "should_close": True,
                                    "action": "CLOSE_FULL",
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
                    "action": "TAKE_PARTIAL",
                    "partial_percent": 15,
                    "close_type": "TIME_FALLBACK_PARTIAL",
                    "reasoning": f"Trade open {hours_in_trade:.1f}h with +{current_pnl:.1f}% profit",
                    "confidence": 70
                }
        
        return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
    
    def check_peak_drawdown_protection(self, pair, trade):
        """Protect against drawdowns from peak"""
        current_price = self.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if 'peak_pnl' not in trade:
            trade['peak_pnl'] = current_pnl
        
        peak = trade['peak_pnl']
        
        # Only check if we have a significant peak
        if peak < 6:
            return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
        
        # Calculate drawdown from peak
        drawdown = peak - current_pnl
        
        # Drawdown protection rules from config
        dd6 = self.drawdown_protection.get("from_peak_8", 8)
        dd4 = self.drawdown_protection.get("from_peak_6", 6)
        dd2 = self.drawdown_protection.get("from_peak_4", 4)
        
        if drawdown >= dd6:  # Lost X% from peak
            return {
                "should_close": True,
                "action": "CLOSE_FULL",
                "partial_percent": 100,
                "close_type": f"PEAK_DRAWDOWN_{dd6}",
                "reasoning": f"🚨 Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 90
            }
        
        elif drawdown >= dd4:  # Lost X% from peak
            return {
                "should_close": True,
                "action": "TAKE_PARTIAL",
                "partial_percent": 50,
                "close_type": f"PEAK_DRAWDOWN_{dd4}",
                "reasoning": f"⚠️ Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 80
            }
        
        elif drawdown >= dd2 and peak >= 15:  # Lost X% from 15%+ peak
            return {
                "should_close": True,
                "action": "TAKE_PARTIAL",
                "partial_percent": 30,
                "close_type": f"PEAK_DRAWDOWN_{dd2}",
                "reasoning": f"Lost {drawdown:.1f}% from high peak {peak:.1f}%",
                "confidence": 75
            }
        
        return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
    
    def get_3percent_exit_decision(self, pair, trade):
        """MAIN EXIT SYSTEM - MAX 4 PARTIALS ONLY - ALL PROTECTIONS STILL ACTIVE"""
        
        # === 1. ဘ Partial close ဘယ်နှစ်ခါ လုပ်ပြီးပြီလဲ ရေတွက် ===
        if 'partial_close_count' not in trade:
            trade['partial_close_count'] = 0
        
        partial_count = trade['partial_close_count']
        max_partials = self.config.get("max_partial_closes_per_trade", 3)
        partial_percentages = self.config.get("partial_percentages", [50, 30, 20])
        
        # အများဆုံး ၄ ခါ ရောက်ပြီဆိုရင် ကျန်တာ အကုန်ထွက်
        if partial_count >= max_partials:
            trade['partial_close_count'] = max_partials  # cap it
            return {
                "should_close": True,
                "action": "CLOSE_FULL",
                "partial_percent": 100,
                "close_type": "MAX_PARTIALS_REACHED",
                "reasoning": f"Reached maximum {max_partials} partial closes - closing remaining position",
                "confidence": 95
            }
        
        # === 2. Emergency stop & Drawdown protection (မဖျက်ဘူး အလုပ်လုပ်နေမယ်) ===
        current_price = self.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if current_pnl <= self.emergency_stop:
            return {
                "should_close": True,
                "action": "CLOSE_FULL",
                "partial_percent": 100,
                "close_type": f"EMERGENCY_STOP_{abs(self.emergency_stop)}",
                "reasoning": f"Emergency stop triggered at {current_pnl:.1f}%",
                "confidence": 100
            }
        
        # Drawdown protection အကုန်လုံး အလုပ်လုပ်နေမယ်
        drawdown_decision = self.check_peak_drawdown_protection(pair, trade)
        if drawdown_decision.get("should_close", False):
            return drawdown_decision
        
        # === 3. Normal 3% AI level check ===
        level_decision = self.check_3percent_level(pair, trade)
        if level_decision.get("should_close", False):
            if level_decision["action"] == "TAKE_PARTIAL":
                # ကိုယ့် config အတိုင်း ရာခိုင်နှုန်း သတ်မှတ်ပေး
                if partial_count < len(partial_percentages):
                    percent_to_close = partial_percentages[partial_count]
                else:
                    percent_to_close = partial_percentages[-1]
                level_decision["partial_percent"] = percent_to_close
                level_decision["close_type"] = f"AI_LEVEL_{percent_to_close}PCT"
                level_decision["reasoning"] = f"AI decided partial at +{current_pnl:.1f}% - taking {percent_to_close}% (close #{partial_count+1})"
                
                # ရေတွက်တိုးပေး
                trade['partial_close_count'] += 1
            return level_decision
        
        # === 4. Time check (3 မိနစ်တစ်ခါ မင်းလိုချင်တဲ့အတိုင်း) ===
        time_decision = self.check_time_based_exit(pair, trade)
        if time_decision.get("should_close", False):
            if time_decision["action"] == "TAKE_PARTIAL":
                percent_to_close = partial_percentages[partial_count]
                time_decision["partial_percent"] = percent_to_close
                time_decision["close_type"] = f"TIME_CHECK_{percent_to_close}PCT"
                trade['partial_close_count'] += 1
            return time_decision
        
        return {
            "should_close": False,
            "action": "HOLD_NEXT_LEVEL",
            "reasoning": "No exit conditions met",
            "confidence": 50
        }
    
    # ==================== FIXED TRADE EXECUTION ====================
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
        """
        ✅ FIXED: Close trade immediately at market price with REAL Binance execution
        partial_percent = 100 → full close
        partial_percent < 100 → partial close (correct & clean logic)
        """
        try:
            current_price = self.get_current_price(pair)

            # ====================== INITIAL VALUES ======================
            initial_quantity      = trade['quantity']
            initial_margin_usd    = trade['position_size_usd']   # actual used margin
            initial_leverage      = trade['leverage']
            entry_price           = trade['entry_price']
            direction             = trade['direction']

            if initial_quantity <= 0:
                self.print_color(f"Warning: Zero quantity for {pair}, skipping close", self.Fore.YELLOW)
                return False

            # ====================== CALCULATE CLOSE RATIO ======================
            close_ratio = partial_percent / 100.0
            close_ratio = min(1.0, max(0.0, close_ratio))  # clamp between 0-1

            closed_quantity = initial_quantity * close_ratio
            closed_margin   = initial_margin_usd * close_ratio

            # Binance precision
            precision = self.quantity_precision.get(pair, 3)
            closed_quantity = round(closed_quantity, precision)
            closed_margin   = round(closed_margin, 3)

            # ====================== P&L FOR CLOSED PORTION ======================
            if direction == 'LONG':
                pnl = (current_price - entry_price) * closed_quantity
            else:  # SHORT
                pnl = (entry_price - current_price) * closed_quantity

            pnl = round(pnl, 4)

            # ====================== REAL BINANCE EXECUTION ======================
            if self.binance and closed_quantity > 0:
                side = 'SELL' if direction == 'LONG' else 'BUY'
                try:
                    # ✅ REAL ORDER SENT TO BINANCE
                    order = self.binance.futures_create_order(
                        symbol=pair,
                        side=side,
                        type='MARKET',
                        quantity=closed_quantity,
                        reduceOnly=True
                    )
                    self.print_color(f"✅ Binance order executed: {closed_quantity} {pair} at market", self.Fore.GREEN)
                except BinanceAPIException as e:
                    self.print_color(f"❌ Binance order failed: {e}", self.Fore.RED)
                    # Try with adjusted quantity
                    try:
                        # Reduce quantity by 1% and retry
                        adjusted_qty = round(closed_quantity * 0.99, precision)
                        order = self.binance.futures_create_order(
                            symbol=pair,
                            side=side,
                            type='MARKET',
                            quantity=adjusted_qty,
                            reduceOnly=True
                        )
                        closed_quantity = adjusted_qty
                        self.print_color(f"✅ Binance order executed with adjusted quantity: {adjusted_qty}", self.Fore.GREEN)
                    except Exception as retry_err:
                        self.print_color(f"❌ Binance order retry failed: {retry_err}", self.Fore.RED)
                        return False
                except Exception as order_err:
                    self.print_color(f"❌ Binance order error: {order_err}", self.Fore.RED)
                    return False

            # ====================== PARTIAL CLOSE (remaining > 0) ======================
            if partial_percent < 100 and closed_quantity < initial_quantity:
                remaining_quantity = initial_quantity - closed_quantity
                remaining_margin   = initial_margin_usd - closed_margin

                # Dust protection - if remaining is too small, close fully
                if remaining_quantity < (0.1 ** precision) or remaining_margin < 0.01:
                    self.print_color(f"Dust position detected, forcing full close", self.Fore.YELLOW)
                    # Close remaining
                    if self.binance and remaining_quantity > 0:
                        side = 'SELL' if direction == 'LONG' else 'BUY'
                        try:
                            self.binance.futures_create_order(
                                symbol=pair,
                                side=side,
                                type='MARKET',
                                quantity=remaining_quantity,
                                reduceOnly=True
                            )
                            remaining_quantity = 0
                            remaining_margin = 0
                        except:
                            pass
                    
                    # Update to full close
                    trade['quantity'] = 0
                    trade['position_size_usd'] = 0
                    
                    # Calculate total P&L
                    if direction == 'LONG':
                        total_pnl = (current_price - entry_price) * initial_quantity
                    else:
                        total_pnl = (entry_price - current_price) * initial_quantity
                    
                    trade.update({
                        'status': 'CLOSED',
                        'exit_price': current_price,
                        'pnl': total_pnl,
                        'close_reason': close_reason + " (DUST_CLEANUP)",
                        'close_time': self.get_thailand_time(),
                        'partial_percent': 100
                    })
                    
                    self.available_budget += initial_margin_usd + total_pnl
                    self.add_trade_to_history(trade.copy())
                    
                    # Clean up
                    self.ai_opened_trades.pop(pair, None)
                    self.checked_3percent_levels.pop(pair, None)
                    
                    self.print_color(f"✅ FULL CLOSE (dust cleanup): {pair} | P&L: ${total_pnl:+.2f}", self.Fore.GREEN if total_pnl > 0 else self.Fore.RED)
                    return True

                # ✅ Update live trade with REAL remaining position
                trade['quantity'] = remaining_quantity
                trade['position_size_usd'] = remaining_margin

                # Record partial close
                partial_trade = trade.copy()
                partial_trade.update({
                    'status': 'PARTIAL_CLOSE',
                    'exit_price': current_price,
                    'pnl': pnl,
                    'close_reason': close_reason,
                    'close_time': self.get_thailand_time(),
                    'partial_percent': partial_percent,
                    'closed_quantity': closed_quantity,
                    'closed_position_size': closed_margin,
                    'peak_pnl_pct': round(trade.get('peak_pnl', 0), 3),
                    'initial_position_size': initial_margin_usd,
                    'remaining_position_size': remaining_margin,
                    'binance_order_id': order.get('orderId', 'SIMULATED') if self.binance else 'SIMULATED'
                })

                self.available_budget += closed_margin + pnl
                self.add_trade_to_history(partial_trade)

                # Calibrator update
                self.calibrator.performance_history.append({
                    'pair': pair,
                    'pnl': pnl,
                    'levels_hit': self.checked_3percent_levels.get(pair, []),
                    'partial_percent': partial_percent
                })

                color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
                self.print_color(f"✅ REAL Partial Close | {pair} | {partial_percent}% | "
                                 f"Closed: {closed_quantity:.6f} (${closed_margin:.2f}) | "
                                 f"P&L: ${pnl:+.2f} | {close_reason}", color)
                self.print_color(f"Remaining: {remaining_quantity:.6f} (${remaining_margin:.2f} margin)", self.Fore.CYAN)

                return True

            # ====================== FULL CLOSE ======================
            else:
                # Final P&L for whole position
                if direction == 'LONG':
                    final_pnl = (current_price - entry_price) * initial_quantity
                else:
                    final_pnl = (entry_price - current_price) * initial_quantity

                final_pnl = round(final_pnl, 4)

                # Update trade record
                trade.update({
                    'status': 'CLOSED',
                    'exit_price': current_price,
                    'pnl': final_pnl,
                    'close_reason': close_reason,
                    'close_time': self.get_thailand_time(),
                    'partial_percent': 100,
                    'peak_pnl_pct': round(trade.get('peak_pnl', 0), 3),
                    'binance_order_id': order.get('orderId', 'SIMULATED') if self.binance else 'SIMULATED'
                })

                self.available_budget += initial_margin_usd + final_pnl
                self.add_trade_to_history(trade.copy())

                # Calibrator
                self.calibrator.performance_history.append({
                    'pair': pair,
                    'pnl': final_pnl,
                    'levels_hit': self.checked_3percent_levels.get(pair, []),
                    'partial_percent': 100
                })

                color = self.Fore.GREEN if final_pnl > 0 else self.Fore.RED
                self.print_color(f"✅ REAL FULL CLOSE | {pair} | Qty: {initial_quantity:.6f} | "
                                 f"P&L: ${final_pnl:+.2f} | {close_reason}", color)

                # Clean up active trade lists
                self.ai_opened_trades.pop(pair, None)
                self.checked_3percent_levels.pop(pair, None)

                return True

        except Exception as e:
            self.print_color(f"❌ Close failed for {pair}: {e}", self.Fore.RED)
            import traceback
            traceback.print_exc()
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
            
            self.print_color(f"\n🤖 DEEPSEEK TRADE EXECUTION ({self.percent_increment}% AI CHECK SYSTEM)", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 80, self.Fore.CYAN)
            self.print_color(f"{direction_icon} {pair}", direction_color)
            self.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
            self.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
            self.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
            self.print_color(f"🎯 EXIT STRATEGY: EVERY {self.percent_increment}% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"📊 Check starts at: {self.min_check_level}%", self.Fore.MAGENTA)
            self.print_color(f"⏰ Additional checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
            self.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
            self.print_color("=" * 80, self.Fore.CYAN)
            
            # Execute live trade WITHOUT TP/SL orders
            order_id = None
            if self.binance:
                entry_side = 'BUY' if decision == 'LONG' else 'SELL'
                
                try:
                    # Set leverage
                    self.binance.futures_change_leverage(symbol=pair, leverage=leverage)
                    
                    # Place MARKET order
                    order = self.binance.futures_create_order(
                        symbol=pair,
                        side=entry_side,
                        type='MARKET',
                        quantity=quantity
                    )
                    order_id = order.get('orderId')
                    self.print_color(f"✅ Binance order placed: {order_id}", self.Fore.GREEN)
                    
                except Exception as e:
                    self.print_color(f"❌ Binance order failed: {e}", self.Fore.RED)
                    return False
            
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
                'peak_pnl': 0,
                'initial_position_size': position_size_usd,
                'binance_order_id': order_id
            }
            
            # Initialize checked levels for this pair
            self.checked_3percent_levels[pair] = []
            
            self.print_color(f"✅ TRADE EXECUTED: {pair} {decision} | Leverage: {leverage}x", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"📊 AI will check at every {self.percent_increment}% profit level", self.Fore.BLUE)
            return True
            
        except Exception as e:
            self.print_color(f"❌ Trade execution failed: {e}", self.Fore.RED)
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== MONITORING ====================
    def monitor_positions(self):
        """Monitor positions using 3% AI Check system"""
        try:
            closed_trades = []
            
            for pair, trade in list(self.ai_opened_trades.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                self.print_color(f"🔍 {self.percent_increment}% System Checking {pair}...", self.Fore.BLUE)
                
                # Get exit decision from 3% system
                exit_decision = self.get_3percent_exit_decision(pair, trade)
                
                if exit_decision.get("should_close", False):
                    close_type = exit_decision.get("close_type", "EXIT")
                    reasoning = exit_decision.get("reasoning", "No reason")
                    partial_percent = exit_decision.get("partial_percent", 100)
                    confidence = exit_decision.get("confidence", 0)
                    action = exit_decision.get("action", "TAKE_PARTIAL")
                    
                    self.print_color(f"🎯 {self.percent_increment}% System Decision for {pair}:", self.Fore.CYAN + self.Style.BRIGHT)
                    self.print_color(f"   Action: {action}", self.Fore.YELLOW)
                    self.print_color(f"   Partial %: {partial_percent}%", self.Fore.MAGENTA)
                    self.print_color(f"   Type: {close_type}", self.Fore.MAGENTA)
                    self.print_color(f"   Confidence: {confidence}%", self.Fore.GREEN if confidence > 70 else self.Fore.YELLOW)
                    self.print_color(f"   Reason: {reasoning}", self.Fore.WHITE)
                    
                    success = self.close_trade_immediately(pair, trade, f"{close_type}: {reasoning}", partial_percent)
                    if success and partial_percent == 100:
                        closed_trades.append(pair)
            
            return closed_trades
                
        except Exception as e:
            self.print_color(f"Monitoring error: {e}", self.Fore.RED)
            return []
    
    # ==================== DASHBOARD ====================
    def display_dashboard(self):
        """Display trading dashboard"""
        self.print_color(f"\n🤖 AI TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 90, self.Fore.CYAN)
        self.print_color(f"🎯 EXIT STRATEGY: EVERY {self.percent_increment}% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"✅ MINIMUM POSITION: ${self.min_position_size}", self.Fore.GREEN)
        self.print_color(f"✅ POSITION MODE: {self.config.get('position_mode', 'CROSS')}", self.Fore.GREEN)
        self.print_color(f"📊 Check Levels: {self.min_check_level}%, {self.min_check_level+self.percent_increment}%, {self.min_check_level+(self.percent_increment*2)}%, etc.", self.Fore.MAGENTA)
        self.print_color(f"⏰ Time Checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
        self.print_color(f"💰 Milestone Partials: {', '.join(map(str, self.milestone_levels))}%", self.Fore.GREEN)
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
                    drawdown = max(0, peak - current_pnl)
                    drawdown_color = self.Fore.YELLOW if drawdown <= 2 else self.Fore.RED
                    self.print_color(f"   🏔️ Peak: {peak:.1f}% | Drawdown: {drawdown:.1f}%", drawdown_color)
                
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
                initial_size = trade.get('initial_position_size', position_size)
                remaining_size = trade.get('remaining_position_size', 0)
                self.print_color(f"     🔸 Partial: {trade['partial_percent']}% ({closed_qty:.4f}) closed | Remaining: ${remaining_size:.2f} of ${initial_size:.2f}", self.Fore.CYAN)
    
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
        self.print_color(f"✅ Minimum Position Size: ${self.min_position_size}", self.Fore.GREEN)
    
    def show_analytics_menu(self):
        """Show analytics menu"""
        while True:
            print("\n" + "="*60)
            print("📊 ANALYTICS MENU")
            print("="*60)
            print("1. Show 3% Level Analytics")
            print("2. Show Configuration")
            print("3. Show Auto-calibration Status")
            print("4. Back to Main Menu")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                self.analytics.show_analytics_dashboard()
                input("\nPress Enter to continue...")
            
            elif choice == "2":
                print("\n🔧 CURRENT CONFIGURATION:")
                print(json.dumps(self.config, indent=2))
                input("\nPress Enter to continue...")
            
            elif choice == "3":
                print("\n⚙️ AUTO-CALIBRATION STATUS:")
                print(f"Optimal increment: {self.calibrator.optimal_increment}%")
                print(f"Optimal start level: {self.calibrator.optimal_start_level}%")
                print(f"Performance history: {len(self.calibrator.performance_history)} trades")
                print(f"Calibration history: {len(self.calibrator.calibration_history)} entries")
                
                if self.calibrator.calibration_history:
                    print("\nRecent calibrations:")
                    for i, cal in enumerate(self.calibrator.calibration_history[-5:]):
                        print(f"  {i+1}. {cal['timestamp']} - {cal['type']} (Increment: {cal['increment']}%, Start: {cal['start_level']}%)")
                
                input("\nPress Enter to continue...")
            
            elif choice == "4":
                break
    
    # ==================== MAIN TRADING LOOP ====================
    def run_trading_cycle(self):
        """Run trading cycle"""
        try:
            # Run auto-calibration
            self.run_auto_calibration()
            
            # Monitor and close positions
            self.monitor_positions()
            self.display_dashboard()
            
            # Show stats periodically
            if hasattr(self, 'cycle_count') and self.cycle_count % 4 == 0:
                self.show_trade_history(8)
                self.show_trading_stats()
                
                # Show analytics every 8 cycles
                if self.cycle_count % 8 == 0:
                    self.analytics.show_analytics_dashboard()
            
            self.print_color(f"\n🔍 DEEPSEEK SCANNING {len(self.available_pairs)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.available_pairs:
                if self.available_budget > self.min_position_size:  # ✅ Check against min position
                    market_data = self.get_price_history(pair)
                    
                    ai_decision = self.get_ai_trading_decision(pair, market_data)
                    
                    if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] >= self.min_position_size:
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
    
    def run_auto_calibration(self):
        """Run auto-calibration"""
        try:
            # Calculate average market volatility
            volatilities = []
            for pair in self.available_pairs[:3]:  # Check first 3 pairs
                vol = self.calculate_market_volatility(pair)
                volatilities.append(vol)
            
            avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 1.0
            
            # Calibrate based on market volatility
            self.calibrator.calibrate_based_on_market(avg_volatility)
            
            # Calibrate based on performance
            self.calibrator.calibrate_based_on_performance()
            
            # Update config if calibration changed parameters
            if (self.calibrator.optimal_increment != self.percent_increment or 
                self.calibrator.optimal_start_level != self.min_check_level):
                
                old_increment = self.percent_increment
                old_start = self.min_check_level
                
                self.percent_increment = self.calibrator.optimal_increment
                self.min_check_level = self.calibrator.optimal_start_level
                
                # Update config
                self.config['percent_increment'] = self.percent_increment
                self.config['min_check_level'] = self.min_check_level
                self.save_config()
                
                self.print_color(f"🔄 Auto-calibration updated: {old_increment}%->{self.percent_increment}%, Start {old_start}%->{self.min_check_level}%", self.Fore.CYAN)
                
        except Exception as e:
            self.print_color(f"Auto-calibration error: {e}", self.Fore.YELLOW)
    
    def start_trading(self):
        """Start trading"""
        self.print_color("🚀 STARTING AI TRADER V6.0 WITH ENHANCED 3% AI CHECK SYSTEM!", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("💰 AI MANAGING $500 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"🎯 EXIT STRATEGY: EVERY {self.percent_increment}% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"✅ MINIMUM POSITION SIZE: ${self.min_position_size}", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"✅ POSITION MODE: {self.config.get('position_mode', 'CROSS')}", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"📊 Check Levels: {self.min_check_level}%, {self.min_check_level+self.percent_increment}%, {self.min_check_level+(self.percent_increment*2)}%, etc.", self.Fore.MAGENTA)
        self.print_color(f"⏰ Time Checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
        self.print_color(f"💰 Milestone Partials: {', '.join(map(str, self.milestone_levels))}%", self.Fore.GREEN)
        self.print_color(f"⚙️ Auto-calibration: {'ON' if self.config.get('auto_calibration', {}).get('enabled', False) else 'OFF'}", self.Fore.CYAN)
        
        # Configuration options
        print("\n" + "="*70)
        print("ENHANCED 3% AI CHECK CONFIGURATION:")
        print(f"1. Start checking at: {self.min_check_level}%")
        print(f"2. Check every: {self.percent_increment}%")
        print(f"3. Time checks every: {self.time_based_check_minutes} minutes")
        print(f"4. Force partial at milestones: {'ON' if self.force_partial_at_milestones else 'OFF'}")
        print(f"5. Emergency stop at: {self.emergency_stop}%")
        print(f"6. Minimum Position Size: ${self.min_position_size}")
        print(f"7. Position Mode: {self.config.get('position_mode', 'CROSS')}")
        print(f"8. Auto-calibration: {'ON' if self.config.get('auto_calibration', {}).get('enabled', False) else 'OFF'}")
        print(f"9. View Analytics Dashboard")
        print(f"0. Start Trading")
        
        config_choice = input("\nConfigure settings? (enter number or 0 to start): ").strip()
        
        if config_choice == "9":
            self.show_analytics_menu()
            return self.start_trading()  # Return to main menu
        
        elif config_choice != "0" and config_choice != "":
            try:
                if config_choice == "1":
                    min_level = input(f"Start checking at % (default {self.min_check_level}): ").strip()
                    if min_level:
                        self.min_check_level = int(min_level)
                        self.config['min_check_level'] = self.min_check_level
                
                elif config_choice == "2":
                    increment = input(f"Check every % (default {self.percent_increment}): ").strip()
                    if increment:
                        self.percent_increment = int(increment)
                        self.config['percent_increment'] = self.percent_increment
                
                elif config_choice == "3":
                    time_check = input(f"Time checks every minutes (default {self.time_based_check_minutes}): ").strip()
                    if time_check:
                        self.time_based_check_minutes = int(time_check)
                        self.config['time_check_minutes'] = self.time_based_check_minutes
                
                elif config_choice == "4":
                    milestone = input(f"Force partial at milestones? (y/N): ").strip().lower()
                    self.force_partial_at_milestones = (milestone == 'y')
                    self.config['force_milestone_partials'] = self.force_partial_at_milestones
                
                elif config_choice == "5":
                    emergency = input(f"Emergency stop at % (default {self.emergency_stop}): ").strip()
                    if emergency:
                        self.emergency_stop = int(emergency)
                        self.config['emergency_stop'] = self.emergency_stop
                
                elif config_choice == "6":
                    min_pos = input(f"Minimum Position Size $ (default {self.min_position_size}): ").strip()
                    if min_pos:
                        self.min_position_size = int(min_pos)
                        self.config['min_position_size'] = self.min_position_size
                
                elif config_choice == "7":
                    mode = input(f"Position Mode (CROSS/ISOLATED) (default {self.config.get('position_mode', 'CROSS')}): ").strip().upper()
                    if mode in ['CROSS', 'ISOLATED']:
                        self.config['position_mode'] = mode
                        # Update Binance if connected
                        if self.binance:
                            try:
                                self.binance.futures_change_position_mode(dualSidePosition=(mode == 'ISOLATED'))
                                for pair in self.available_pairs:
                                    self.binance.futures_change_margin_type(symbol=pair, marginType=mode)
                                self.print_color(f"✅ Position mode updated to {mode}", self.Fore.GREEN)
                            except Exception as e:
                                self.print_color(f"Failed to update position mode: {e}", self.Fore.YELLOW)
                
                elif config_choice == "8":
                    auto_cal = input(f"Enable auto-calibration? (y/N): ").strip().lower()
                    if 'auto_calibration' not in self.config:
                        self.config['auto_calibration'] = {}
                    self.config['auto_calibration']['enabled'] = (auto_cal == 'y')
                
                self.save_config()
                self.print_color("✅ Configuration updated and saved!", self.Fore.GREEN)
                return self.start_trading()  # Show updated config
                
            except:
                self.print_color("⚠️ Invalid configuration, using defaults", self.Fore.YELLOW)
        
        self.cycle_count = 0
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🔄 TRADING CYCLE {self.cycle_count} ({self.percent_increment}% AI CHECK)", self.Fore.CYAN + self.Style.BRIGHT)
                self.print_color("=" * 70, self.Fore.CYAN)
                self.run_trading_cycle()
                self.print_color(f"⏳ Next analysis in 3 minutes...", self.Fore.BLUE)
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_trade_history(15)
                self.show_trading_stats()
                self.show_analytics_menu()
                break
            except Exception as e:
                self.print_color(f"Main loop error: {e}", self.Fore.RED)
                time.sleep(self.monitoring_interval)


# ==================== FIXED PAPER TRADING CLASS ====================
class FullyAutonomous1HourPaperTrader:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        self.Fore = real_bot.Fore
        self.Back = real_bot.Back
        self.Style = real_bot.Style
        self.COLORAMA_AVAILABLE = real_bot.COLORAMA_AVAILABLE
        
        # Load paper config from main config
        paper_config = real_bot.config.get('paper_trading', {})
        
        # Copy 3% system settings
        self.exit_strategy_mode = "3PERCENT_AI_CHECK"
        self.min_check_level = real_bot.min_check_level
        self.percent_increment = real_bot.percent_increment
        self.force_partial_at_milestones = real_bot.force_partial_at_milestones
        self.milestone_levels = real_bot.milestone_levels
        self.time_based_check_minutes = real_bot.time_based_check_minutes
        self.emergency_stop = real_bot.emergency_stop
        self.drawdown_protection = real_bot.drawdown_protection
        
        # ✅ Copy minimum position size
        self.min_position_size = real_bot.min_position_size
        
        # Paper trading specific
        self.checked_3percent_levels = {}
        self.last_ai_check_time = {}
        
        # Paper trading settings from config
        self.monitoring_interval = 180
        self.paper_balance = paper_config.get('virtual_balance', 500)
        self.available_budget = self.paper_balance
        self.paper_positions = {}
        self.paper_history_file = "fully_autonomous_1hour_paper_trading_history.json"
        self.paper_history = self.load_paper_history()
        self.available_pairs = ["SOLUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "LINKUSDT"]
        self.max_concurrent_trades = paper_config.get('max_positions', 6)
        
        # Analytics for paper trading
        self.paper_analytics_file = "paper_trading_analytics.json"
        self.paper_analytics = self.load_paper_analytics()
        
        self.real_bot.print_color("🤖 ENHANCED PAPER TRADER V6.0 INITIALIZED!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Virtual Budget: ${self.paper_balance}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color(f"🎯 EXIT STRATEGY: EVERY {self.percent_increment}% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"✅ MINIMUM POSITION SIZE: ${self.min_position_size}", self.Fore.GREEN)
        self.real_bot.print_color(f"📊 Check starts at: {self.min_check_level}%", self.Fore.MAGENTA)
        self.real_bot.print_color(f"⚙️ All features from real trading available!", self.Fore.BLUE)
    
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
    
    def load_paper_analytics(self):
        """Load paper trading analytics"""
        try:
            if os.path.exists(self.paper_analytics_file):
                with open(self.paper_analytics_file, 'r') as f:
                    data = json.load(f)
                    # Clean UNKNOWN entries
                    if 'level_decisions' in data:
                        for entry in data['level_decisions']:
                            if entry.get('ai_decision') == 'UNKNOWN':
                                if entry.get('partial_percent', 0) > 0 and entry.get('partial_percent', 0) < 100:
                                    entry['ai_decision'] = 'TAKE_PARTIAL'
                                elif entry.get('partial_percent', 0) == 100:
                                    entry['ai_decision'] = 'CLOSE_FULL'
                                else:
                                    entry['ai_decision'] = 'HOLD_NEXT_LEVEL'
                    return data
            return {"level_decisions": [], "trade_history": []}
        except Exception as e:
            self.real_bot.print_color(f"Error loading paper analytics: {e}", self.Fore.YELLOW)
            return {"level_decisions": [], "trade_history": []}
    
    def save_paper_history(self):
        """Save paper trading history"""
        try:
            with open(self.paper_history_file, 'w') as f:
                json.dump(self.paper_history, f, indent=2)
        except Exception as e:
            self.real_bot.print_color(f"Error saving paper trade history: {e}", self.Fore.RED)
    
    def save_paper_analytics(self):
        """Save paper trading analytics"""
        try:
            with open(self.paper_analytics_file, 'w') as f:
                json.dump(self.paper_analytics, f, indent=2)
        except Exception as e:
            self.real_bot.print_color(f"Error saving paper analytics: {e}", self.Fore.YELLOW)
    
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
            
            # Log to paper analytics
            if 'level_decisions' not in self.paper_analytics:
                self.paper_analytics['level_decisions'] = []
            
            self.paper_analytics['trade_history'].append({
                'timestamp': datetime.now().isoformat(),
                'pair': trade_data['pair'],
                'pnl': trade_data.get('pnl', 0),
                'partial_percent': trade_data.get('partial_percent', 100)
            })
            
            self.save_paper_analytics()
            
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
            return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
        
        # Initialize checked levels for this pair
        if pair not in self.checked_3percent_levels:
            self.checked_3percent_levels[pair] = []
        
        # Check if this level needs evaluation
        if current_level not in self.checked_3percent_levels[pair]:
            self.real_bot.print_color(f"🎯 PAPER {pair} reached +{current_pnl:.1f}% (Level {current_level}%)", self.Fore.CYAN)
            
            # Add to checked levels
            self.checked_3percent_levels[pair].append(current_level)
            
            # Get AI decision at this level (using robust error handling)
            market_data = self.real_bot.get_price_history(pair)
            ai_decision = self.real_bot.robust_ai_exit_decision(pair, trade, market_data, current_level)
            
            # Ensure action key exists
            if "action" not in ai_decision:
                if ai_decision.get("should_close", False):
                    if ai_decision.get("partial_percent", 100) < 100:
                        ai_decision["action"] = "TAKE_PARTIAL"
                    else:
                        ai_decision["action"] = "CLOSE_FULL"
                else:
                    ai_decision["action"] = "HOLD_NEXT_LEVEL"
            
            # Log to paper analytics
            if 'level_decisions' not in self.paper_analytics:
                self.paper_analytics['level_decisions'] = []
            
            self.paper_analytics['level_decisions'].append({
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'level': current_level,
                'ai_decision': ai_decision.get("action", "HOLD_NEXT_LEVEL"),
                'partial_percent': ai_decision.get("partial_percent", 0),
                'confidence': ai_decision.get("confidence", 0),
                'close_type': ai_decision.get("close_type", "")
            })
            
            if len(self.paper_analytics['level_decisions']) > 1000:
                self.paper_analytics['level_decisions'] = self.paper_analytics['level_decisions'][-1000:]
            
            self.save_paper_analytics()
            
            return ai_decision
        
        return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
    
    def paper_check_milestone_partial(self, pair, trade):
        """Paper version of milestone partial"""
        current_price = self.real_bot.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if not self.force_partial_at_milestones:
            return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
        
        for milestone in self.milestone_levels:
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
                        "action": "TAKE_PARTIAL",
                        "partial_percent": partial_percent,
                        "close_type": f"PAPER_MILESTONE_{milestone}",
                        "reasoning": f"🎉 PAPER Milestone! Taking {partial_percent}% at +{milestone}%",
                        "confidence": 85
                    }
        
        return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
    
    def paper_check_peak_drawdown_protection(self, pair, trade):
        """Paper version of drawdown protection"""
        current_price = self.real_bot.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if 'peak_pnl' not in trade:
            trade['peak_pnl'] = current_pnl
        
        peak = trade['peak_pnl']
        
        if peak < 6:
            return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
        
        drawdown = peak - current_pnl
        
        dd6 = self.drawdown_protection.get("from_peak_8", 8)
        dd4 = self.drawdown_protection.get("from_peak_6", 6)
        dd2 = self.drawdown_protection.get("from_peak_4", 4)
        
        if drawdown >= dd6:
            return {
                "should_close": True,
                "action": "CLOSE_FULL",
                "partial_percent": 100,
                "close_type": f"PAPER_PEAK_DRAWDOWN_{dd6}",
                "reasoning": f"🚨 PAPER Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 90
            }
        
        elif drawdown >= dd4:
            return {
                "should_close": True,
                "action": "TAKE_PARTIAL",
                "partial_percent": 50,
                "close_type": f"PAPER_PEAK_DRAWDOWN_{dd4}",
                "reasoning": f"⚠️ PAPER Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 80
            }
        
        elif drawdown >= dd2 and peak >= 15:
            return {
                "should_close": True,
                "action": "TAKE_PARTIAL",
                "partial_percent": 30,
                "close_type": f"PAPER_PEAK_DRAWDOWN_{dd2}",
                "reasoning": f"PAPER Lost {drawdown:.1f}% from peak {peak:.1f}%",
                "confidence": 75
            }
        
        return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
    
    def paper_check_time_based_exit(self, pair, trade):
        """Paper version of time-based exit"""
        current_time = time.time()
        entry_time = trade.get('entry_time', current_time)
        
        # Calculate hours in trade
        hours_in_trade = (current_time - entry_time) / 3600
        
        # Check every X minutes (from config)
        last_check = self.last_ai_check_time.get(pair, 0)
        if current_time - last_check >= (self.time_based_check_minutes * 60):
            self.last_ai_check_time[pair] = current_time
            
            current_price = self.real_bot.get_current_price(pair)
            current_pnl = self.calculate_current_pnl(trade, current_price)
            
            # Only check if profit > 5%
            if current_pnl >= 5:
                # Fallback: Small partial if trade is old and profitable
                if hours_in_trade >= 4 and current_pnl >= 10:
                    return {
                        "should_close": True,
                        "action": "TAKE_PARTIAL",
                        "partial_percent": 15,
                        "close_type": "PAPER_TIME_FALLBACK_PARTIAL",
                        "reasoning": f"PAPER Trade open {hours_in_trade:.1f}h with +{current_pnl:.1f}% profit",
                        "confidence": 70
                    }
        
        return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
    
    def paper_get_3percent_exit_decision(self, pair, trade):
        """Paper version of the main exit system"""
        
        # 1. Count partial closes
        if 'partial_close_count' not in trade:
            trade['partial_close_count'] = 0
        
        partial_count = trade['partial_close_count']
        max_partials = self.real_bot.config.get("max_partial_closes_per_trade", 3)
        partial_percentages = self.real_bot.config.get("partial_percentages", [50, 30, 20])
        
        # Maximum 4 partials reached, close remaining
        if partial_count >= max_partials:
            trade['partial_close_count'] = max_partials
            return {
                "should_close": True,
                "action": "CLOSE_FULL",
                "partial_percent": 100,
                "close_type": "PAPER_MAX_PARTIALS_REACHED",
                "reasoning": f"PAPER Reached maximum {max_partials} partial closes - closing remaining position",
                "confidence": 95
            }
        
        # 2. Emergency stop & Drawdown protection
        current_price = self.real_bot.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if current_pnl <= self.emergency_stop:
            return {
                "should_close": True,
                "action": "CLOSE_FULL",
                "partial_percent": 100,
                "close_type": f"PAPER_EMERGENCY_STOP_{abs(self.emergency_stop)}",
                "reasoning": f"PAPER Emergency stop triggered at {current_pnl:.1f}%",
                "confidence": 100
            }
        
        # Drawdown protection
        drawdown_decision = self.paper_check_peak_drawdown_protection(pair, trade)
        if drawdown_decision.get("should_close", False):
            return drawdown_decision
        
        # 3. Normal 3% AI level check
        level_decision = self.paper_check_3percent_level(pair, trade)
        if level_decision.get("should_close", False):
            if level_decision["action"] == "TAKE_PARTIAL":
                if partial_count < len(partial_percentages):
                    percent_to_close = partial_percentages[partial_count]
                else:
                    percent_to_close = partial_percentages[-1]
                level_decision["partial_percent"] = percent_to_close
                level_decision["close_type"] = f"PAPER_AI_LEVEL_{percent_to_close}PCT"
                level_decision["reasoning"] = f"PAPER AI decided partial at +{current_pnl:.1f}% - taking {percent_to_close}% (close #{partial_count+1})"
                
                # Increment counter
                trade['partial_close_count'] += 1
            return level_decision
        
        # 4. Time-based check
        time_decision = self.paper_check_time_based_exit(pair, trade)
        if time_decision.get("should_close", False):
            if time_decision["action"] == "TAKE_PARTIAL":
                percent_to_close = partial_percentages[partial_count]
                time_decision["partial_percent"] = percent_to_close
                time_decision["close_type"] = f"PAPER_TIME_CHECK_{percent_to_close}PCT"
                trade['partial_close_count'] += 1
            return time_decision
        
        # 5. Milestone check
        milestone_decision = self.paper_check_milestone_partial(pair, trade)
        if milestone_decision.get("should_close", False):
            if milestone_decision["action"] == "TAKE_PARTIAL":
                if partial_count < len(partial_percentages):
                    milestone_decision["partial_percent"] = partial_percentages[partial_count]
                else:
                    milestone_decision["partial_percent"] = partial_percentages[-1]
                trade['partial_close_count'] += 1
            return milestone_decision
        
        # No exit conditions met
        return {"should_close": False, "action": "HOLD_NEXT_LEVEL"}
    
    def paper_close_trade_immediately(self, pair, trade, close_reason="AI_DECISION", partial_percent=100):
        """
        Paper trading version - အတိအကျ real bot နဲ့ အတူတူ logic သုံးထားတယ်
        partial_percent = 100 → full close
        partial_percent < 100 → partial close (မှန်ကန်တဲ့ quantity/margin ratio)
        """
        try:
            current_price = self.real_bot.get_current_price(pair)

            # ====================== INITIAL VALUES ======================
            initial_quantity      = trade['quantity']
            initial_margin_usd    = trade['position_size_usd']
            initial_leverage      = trade['leverage']
            entry_price           = trade['entry_price']
            direction             = trade['direction']

            if initial_quantity <= 0:
                self.real_bot.print_color(f"Paper Warning: Zero quantity for {pair}, skipping close", self.real_bot.Fore.YELLOW)
                return False

            # ====================== CLOSE RATIO ======================
            close_ratio = partial_percent / 100.0
            close_ratio = min(1.0, max(0.0, close_ratio))  # 0-100% ကြားမှာပဲ ရှိရင်

            closed_quantity = initial_quantity * close_ratio
            closed_margin   = initial_margin_usd * close_ratio

            # Precision သတ်မှတ်ချက်
            closed_quantity = round(closed_quantity, 6)
            closed_margin   = round(closed_margin, 3)

            # ====================== P&L CALCULATION ======================
            if direction == 'LONG':
                pnl = (current_price - entry_price) * closed_quantity
            else:
                pnl = (entry_price - current_price) * closed_quantity
            pnl = round(pnl, 4)

            # ====================== PARTIAL CLOSE ======================
            if partial_percent < 100 and closed_quantity < initial_quantity:
                remaining_quantity = initial_quantity - closed_quantity
                remaining_margin   = initial_margin_usd - closed_margin

                # Dust position ရှိရင် အလိုအလျောက် full close လုပ်ပေး
                if remaining_quantity < 0.000001 or remaining_margin < 0.01:
                    self.real_bot.print_color(
                        f"Paper Dust detected ({remaining_quantity:.8f}), forcing full close", 
                        self.real_bot.Fore.YELLOW
                    )
                    return self.paper_close_trade_immediately(pair, trade, close_reason + " (DUST)", 100)

                # Update live paper position
                trade['quantity']        = remaining_quantity
                trade['position_size_usd'] = remaining_margin

                # Record partial close
                partial_trade = trade.copy()
                partial_trade.update({
                    'status'              : 'PARTIAL_CLOSE',
                    'exit_price'          : current_price,
                    'pnl'                 : pnl,
                    'close_reason'        : close_reason,
                    'close_time'          : self.real_bot.get_thailand_time(),
                    'partial_percent'     : partial_percent,
                    'closed_quantity'     : closed_quantity,
                    'closed_position_size': closed_margin,
                    'peak_pnl_pct'        : round(trade.get('peak_pnl', 0), 3),
                    'initial_position_size': initial_margin_usd,
                    'remaining_position_size': remaining_margin
                })

                self.available_budget += closed_margin + pnl
                self.add_paper_trade_to_history(partial_trade)

                color = self.real_bot.Fore.GREEN if pnl > 0 else self.real_bot.Fore.RED
                self.real_bot.print_color(
                    f"Paper Partial Close | {pair} | {partial_percent}% | "
                    f"Closed: {closed_quantity:.6f} (${closed_margin:.2f}) | "
                    f"P&L: ${pnl:+.2f} | {close_reason}", color
                )
                self.real_bot.print_color(
                    f"Paper Remaining: {remaining_quantity:.6f} (${remaining_margin:.2f} margin, {initial_leverage}x)", 
                    self.real_bot.Fore.CYAN
                )

                return True

            # ====================== FULL CLOSE ======================
            else:
                # Final P&L
                if direction == 'LONG':
                    final_pnl = (current_price - entry_price) * initial_quantity
                else:
                    final_pnl = (entry_price - current_price) * initial_quantity
                final_pnl = round(final_pnl, 4)

                # Update trade record
                trade.update({
                    'status'       : 'CLOSED',
                    'exit_price'   : current_price,
                    'pnl'          : final_pnl,
                    'close_reason' : close_reason,
                    'close_time'   : self.real_bot.get_thailand_time(),
                    'partial_percent': 100,
                    'peak_pnl_pct' : round(trade.get('peak_pnl', 0), 3)
                })

                self.available_budget += initial_margin_usd + final_pnl
                self.add_paper_trade_to_history(trade.copy())

                color = self.real_bot.Fore.GREEN if final_pnl > 0 else self.real_bot.Fore.RED
                self.real_bot.print_color(
                    f"PAPER FULL CLOSE | {pair} | Qty: {initial_quantity:.6f} | "
                    f"P&L: ${final_pnl:+.2f} | {close_reason}", color
                )

                # Clean up
                self.paper_positions.pop(pair, None)
                self.checked_3percent_levels.pop(pair, None)

                return True

        except Exception as e:
            self.real_bot.print_color(f"Paper Close Error ({pair}): {e}", self.real_bot.Fore.RED)
            import traceback
            traceback.print_exc()
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
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Max concurrent trades reached ({self.max_concurrent_trades})", self.Fore.RED)
                return False
                
            if position_size_usd > self.available_budget:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Insufficient budget", self.Fore.RED)
                return False
            
            # ✅ ENFORCE MINIMUM POSITION SIZE
            if position_size_usd < self.min_position_size:
                position_size_usd = self.min_position_size
                self.real_bot.print_color(f"⚠️ PAPER: Position size increased to minimum ${self.min_position_size}", self.Fore.YELLOW)
            
            notional_value = position_size_usd * leverage
            quantity = notional_value / entry_price
            quantity = round(quantity, 3)
            
            direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
            
            self.real_bot.print_color(f"\n🤖 PAPER TRADE EXECUTION ({self.percent_increment}% AI CHECK)", self.Fore.CYAN + self.Style.BRIGHT)
            self.real_bot.print_color("=" * 80, self.Fore.CYAN)
            self.real_bot.print_color(f"{direction_icon} {pair}", direction_color)
            self.real_bot.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.real_bot.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
            self.real_bot.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
            self.real_bot.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
            self.real_bot.print_color(f"🎯 EXIT STRATEGY: EVERY {self.percent_increment}% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
            self.real_bot.print_color(f"✅ MINIMUM POSITION: ${self.min_position_size}", self.Fore.GREEN)
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
                'peak_pnl': 0,
                'initial_position_size': position_size_usd
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
                
                self.real_bot.print_color(f"🔍 PAPER {self.percent_increment}% System Checking {pair}...", self.Fore.BLUE)
                
                # Get exit decision
                exit_decision = self.paper_get_3percent_exit_decision(pair, trade)
                
                if exit_decision.get("should_close", False):
                    close_type = exit_decision.get("close_type", "EXIT")
                    reasoning = exit_decision.get("reasoning", "No reason")
                    partial_percent = exit_decision.get("partial_percent", 100)
                    confidence = exit_decision.get("confidence", 0)
                    action = exit_decision.get("action", "TAKE_PARTIAL")
                    
                    self.real_bot.print_color(f"🎯 PAPER {self.percent_increment}% System Decision for {pair}:", self.Fore.CYAN + self.Style.BRIGHT)
                    self.real_bot.print_color(f"   Action: {action}", self.Fore.YELLOW)
                    self.real_bot.print_color(f"   Partial %: {partial_percent}%", self.Fore.MAGENTA)
                    self.real_bot.print_color(f"   Type: {close_type}", self.Fore.MAGENTA)
                    self.real_bot.print_color(f"   Confidence: {confidence}%", self.Fore.GREEN if confidence > 70 else self.Fore.YELLOW)
                    self.real_bot.print_color(f"   Reason: {reasoning}", self.Fore.WHITE)
                    
                    success = self.paper_close_trade_immediately(pair, trade, f"PAPER_{close_type}: {reasoning}", partial_percent)
                    if success and partial_percent == 100:
                        closed_positions.append(pair)
            
            return closed_positions
                    
        except Exception as e:
            self.real_bot.print_color(f"PAPER Monitoring error: {e}", self.Fore.RED)
            return []
    
    def display_paper_dashboard(self):
        """Display paper trading dashboard"""
        self.real_bot.print_color(f"\n🤖 PAPER TRADING DASHBOARD - {self.real_bot.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 90, self.Fore.CYAN)
        self.real_bot.print_color(f"🎯 EXIT STRATEGY: EVERY {self.percent_increment}% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"✅ MINIMUM POSITION: ${self.min_position_size}", self.Fore.GREEN)
        self.real_bot.print_color(f"📊 Check Levels: {self.min_check_level}%, {self.min_check_level+self.percent_increment}%, {self.min_check_level+(self.percent_increment*2)}%, etc.", self.Fore.MAGENTA)
        self.real_bot.print_color(f"⏰ Time Checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
        self.real_bot.print_color(f"💰 Milestone Partials: {', '.join(map(str, self.milestone_levels))}%", self.Fore.GREEN)
        
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
                    drawdown = max(0, peak - current_pnl)
                    drawdown_color = self.Fore.YELLOW if drawdown <= 2 else self.Fore.RED
                    self.real_bot.print_color(f"   🏔️ Peak: {peak:.1f}% | Drawdown: {drawdown:.1f}%", drawdown_color)
                
                self.real_bot.print_color("   " + "-" * 60, self.Fore.CYAN)
        
        if active_count == 0:
            self.real_bot.print_color("No active paper positions", self.Fore.YELLOW)
        else:
            total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
            self.real_bot.print_color(f"📊 Active Paper Positions: {active_count}/{self.max_concurrent_trades} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)
        
        self.real_bot.print_color(f"💰 Paper Balance: ${self.paper_balance:.2f} | Available: ${self.available_budget:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"✅ Minimum Position Size: ${self.min_position_size}", self.Fore.GREEN)
    
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
                initial_size = trade.get('initial_position_size', position_size)
                remaining_size = trade.get('remaining_position_size', 0)
                self.real_bot.print_color(f"     🔸 Partial: {trade['partial_percent']}% ({closed_qty:.4f}) closed | Remaining: ${remaining_size:.2f} of ${initial_size:.2f}", self.Fore.CYAN)
    
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
    
    def show_paper_analytics(self):
        """Show paper trading analytics"""
        if 'level_decisions' not in self.paper_analytics or not self.paper_analytics['level_decisions']:
            self.real_bot.print_color("No paper analytics data available yet.", self.Fore.YELLOW)
            return
        
        level_stats = {}
        for entry in self.paper_analytics['level_decisions']:
            level = entry['level']
            if level not in level_stats:
                level_stats[level] = {
                    "count": 0,
                    "decisions": {"HOLD_NEXT_LEVEL": 0, "TAKE_PARTIAL": 0, "CLOSE_FULL": 0}
                }
            
            level_stats[level]["count"] += 1
            decision = entry.get('ai_decision', 'UNKNOWN')
            
            if decision == "HOLD" or decision == "HOLD_NEXT_LEVEL":
                level_stats[level]["decisions"]["HOLD_NEXT_LEVEL"] += 1
            elif decision == "TAKE_PARTIAL":
                level_stats[level]["decisions"]["TAKE_PARTIAL"] += 1
            elif decision == "CLOSE_FULL":
                level_stats[level]["decisions"]["CLOSE_FULL"] += 1
            else:
                # Infer from partial_percent
                if entry.get('partial_percent', 0) > 0 and entry.get('partial_percent', 0) < 100:
                    level_stats[level]["decisions"]["TAKE_PARTIAL"] += 1
                elif entry.get('partial_percent', 0) == 100:
                    level_stats[level]["decisions"]["CLOSE_FULL"] += 1
                else:
                    level_stats[level]["decisions"]["HOLD_NEXT_LEVEL"] += 1
        
        self.real_bot.print_color("\n📊 PAPER 3% LEVEL ANALYTICS (FIXED)", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 80, self.Fore.CYAN)
        
        print(f"{'Level':<10} {'Count':<8} {'HOLD':<8} {'PARTIAL':<10} {'FULL':<8}")
        print("-" * 80)
        
        for level in sorted(level_stats.keys()):
            s = level_stats[level]
            if s["count"] >= 1:
                print(f"+{level}%:    {s['count']:<8} "
                      f"{s['decisions'].get('HOLD_NEXT_LEVEL', 0):<8} "
                      f"{s['decisions'].get('TAKE_PARTIAL', 0):<10} "
                      f"{s['decisions'].get('CLOSE_FULL', 0):<8}")
    
    def run_paper_trading_cycle(self):
        """Run paper trading cycle"""
        try:
            self.monitor_paper_positions()
            self.display_paper_dashboard()
            
            if hasattr(self, 'paper_cycle_count') and self.paper_cycle_count % 4 == 0:
                self.show_paper_history(8)
                self.show_paper_stats()
                
                # Show analytics every 8 cycles
                if self.paper_cycle_count % 8 == 0:
                    self.show_paper_analytics()
            
            self.real_bot.print_color(f"\nPAPER: DEEPSEEK SCANNING {len(self.available_pairs)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.available_pairs:
                if self.available_budget > self.min_position_size:  # ✅ Check against min position
                    market_data = self.real_bot.get_price_history(pair)
                    
                    ai_decision = self.real_bot.get_ai_trading_decision(pair, market_data)
                    
                    if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] >= self.min_position_size:
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
        self.real_bot.print_color("🚀 STARTING ENHANCED PAPER TRADING V6.0 WITH 3% AI CHECK SYSTEM!", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("💰 VIRTUAL $500 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"🎯 EXIT STRATEGY: EVERY {self.percent_increment}% AI CHECK", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"✅ MINIMUM POSITION SIZE: ${self.min_position_size}", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"📊 Check starts at: {self.min_check_level}%", self.Fore.MAGENTA)
        self.real_bot.print_color(f"⏰ Time checks: Every {self.time_based_check_minutes} minutes", self.Fore.BLUE)
        self.real_bot.print_color(f"⚙️ All features: Configuration, Analytics, Auto-calibration", self.Fore.BLUE)
        
        # Configuration for paper trading
        print("\n" + "="*70)
        print("ENHANCED PAPER TRADING CONFIGURATION:")
        print(f"1. Start checking at: {self.min_check_level}%")
        print(f"2. Check every: {self.percent_increment}%")
        print(f"3. Time checks every: {self.time_based_check_minutes} minutes")
        print(f"4. Emergency stop at: {self.emergency_stop}%")
        print(f"5. Minimum Position Size: ${self.min_position_size}")
        print(f"6. View Paper Analytics")
        print(f"7. Back to Main Menu")
        
        config_choice = input("\nSelect option (1-7): ").strip()
        
        if config_choice == "6":
            self.show_paper_analytics()
            input("\nPress Enter to continue...")
            return self.start_paper_trading()
        
        elif config_choice == "7":
            return
        
        elif config_choice != "":
            try:
                if config_choice == "1":
                    min_level = input(f"Start checking at % (default {self.min_check_level}): ").strip()
                    if min_level:
                        self.min_check_level = int(min_level)
                
                elif config_choice == "2":
                    increment = input(f"Check every % (default {self.percent_increment}): ").strip()
                    if increment:
                        self.percent_increment = int(increment)
                
                elif config_choice == "3":
                    time_check = input(f"Time checks every minutes (default {self.time_based_check_minutes}): ").strip()
                    if time_check:
                        self.time_based_check_minutes = int(time_check)
                
                elif config_choice == "4":
                    emergency = input(f"Emergency stop at % (default {self.emergency_stop}): ").strip()
                    if emergency:
                        self.emergency_stop = int(emergency)
                
                elif config_choice == "5":
                    min_pos = input(f"Minimum Position Size $ (default {self.min_position_size}): ").strip()
                    if min_pos:
                        self.min_position_size = int(min_pos)
                
                self.real_bot.print_color("✅ PAPER Configuration updated!", self.Fore.GREEN)
            except:
                self.real_bot.print_color("⚠️ Invalid configuration, using defaults", self.Fore.YELLOW)
        
        self.paper_cycle_count = 0
        while True:
            try:
                self.paper_cycle_count += 1
                self.real_bot.print_color(f"\n🔄 PAPER TRADING CYCLE {self.paper_cycle_count} ({self.percent_increment}% AI CHECK)", self.Fore.CYAN + self.Style.BRIGHT)
                self.real_bot.print_color("=" * 60, self.Fore.CYAN)
                self.run_paper_trading_cycle()
                self.real_bot.print_color(f"⏳ Next paper analysis in 3 minutes...", self.Fore.BLUE)
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                self.real_bot.print_color(f"\n🛑 PAPER TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_paper_history(15)
                self.show_paper_stats()
                self.show_paper_analytics()
                break
            except Exception as e:
                self.real_bot.print_color(f"PAPER: Main loop error: {e}", self.Fore.RED)
                time.sleep(self.monitoring_interval)


# ==================== TEST FUNCTION ====================
def test_partial_close_logic():
    """Test function to verify partial close logic"""
    print("\n🧪 TESTING PARTIAL CLOSE LOGIC")
    print("="*50)
    
    # Test case: 100$ position with 5x leverage
    initial_position = 100.0
    leverage = 5.0
    partial_percent = 50  # Close 50%
    current_price = 150.0  # Assume price increased
    entry_price = 100.0
    
    # Calculate
    initial_notional = initial_position * leverage  # 500$
    closed_notional = initial_notional * (partial_percent / 100.0)  # 250$
    closed_quantity = closed_notional / current_price  # 1.6667
    closed_margin = closed_notional / leverage  # 50$
    
    remaining_quantity = (initial_notional / entry_price) - closed_quantity  # Initial quantity - closed quantity
    remaining_margin = initial_position - closed_margin  # 50$
    
    print(f"Initial Position: ${initial_position}")
    print(f"Leverage: {leverage}x")
    print(f"Initial Notional: ${initial_notional}")
    print(f"\nClosing {partial_percent}%:")
    print(f"  Closed Notional: ${closed_notional}")
    print(f"  Closed Quantity: {closed_quantity:.4f}")
    print(f"  Closed Margin: ${closed_margin}")
    print(f"\nRemaining Position:")
    print(f"  Remaining Margin: ${remaining_margin}")
    print(f"  Remaining Quantity: {remaining_quantity:.4f}")
    print(f"  Effective Leverage: {(remaining_margin * leverage) / remaining_margin if remaining_margin > 0 else 0}x")
    
    # Test P&L calculation
    pnl = (current_price - entry_price) * closed_quantity
    print(f"\nP&L for closed portion: ${pnl:.2f}")
    
    return True


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    try:
        # First run the test
        test_partial_close_logic()
        
        print("\n" + "="*70)
        print("🤖 FIXED FULLY AUTONOMOUS AI TRADER V6.0")
        print("CORRECTED PARTIAL CLOSE SYSTEM")
        print("="*70)
        print("1. 🎯 REAL TRADING (Live Binance Account)")
        print("   - Fixed Partial Close Logic")
        print("   - Cross Mode Position")
        print("   - $25 Minimum Position Size")
        print("   - REAL Binance Execution")
        print("")
        print("2. 📝 PAPER TRADING (Virtual Simulation)")
        print("   - Same fixes as real trading")
        print("   - $25 Minimum Position")
        print("   - Risk-free testing")
        print("")
        print("3. 🔧 VIEW ANALYTICS & CONFIG")
        print("4. 🧪 RUN PARTIAL CLOSE TEST")
        print("5. ❌ EXIT")
        
        choice = input("\nSelect mode (1-5): ").strip()
        
        if choice == "1":
            bot = FullyAutonomous1HourAITrader()
            if bot.binance:
                bot.start_trading()
            else:
                print(f"\n❌ Binance connection failed. Switching to paper trading...")
                paper_bot = FullyAutonomous1HourPaperTrader(bot)
                paper_bot.start_paper_trading()
                
        elif choice == "2":
            bot = FullyAutonomous1HourAITrader()
            paper_bot = FullyAutonomous1HourPaperTrader(bot)
            paper_bot.start_paper_trading()
            
        elif choice == "3":
            bot = FullyAutonomous1HourAITrader()
            bot.show_analytics_menu()
            
        elif choice == "4":
            test_partial_close_logic()
            input("\nPress Enter to continue...")
            
        elif choice == "5":
            print(f"\n👋 Exiting...")
            
        else:
            print(f"\n❌ Invalid choice. Exiting...")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Program stopped by user")
    except Exception as e:
        print(f"\n❌ Main execution error: {e}")
        import traceback
        traceback.print_exc()
