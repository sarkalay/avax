# data_collector.py
# Fully Intelligent Self-Learning Data Collector (2025 Pro Version)
# အစ်ကို့အတွက် အထူးဖန်တီးပေးထားတာ

import csv
import os
import time
from datetime import datetime

DATA_FILE = "ml_training_data.csv"

def classify_trade_outcome(trade_data):
    """
    တကယ့် ဉာဏ်ရည်ထက်မြက်တဲ့ classification
    Winner-Turn-Loser တွေကို အဓိက ဖမ်းမယ်
    """
    try:
        pnl = trade_data.get("pnl", 0)
        peak_pnl_pct = trade_data.get("peak_pnl_pct", 0)
        close_reason = trade_data.get("close_reason", "").upper()

        # အနည်းဆုံး +9% ထိ တက်ဖူးပြီး နောက်ဆုံး ရှုံးသွားရင် → အဆိုးဆုံး အမှား
        if peak_pnl_pct >= 9.0 and pnl <= 0:
            return "WINNER_TURN_LOSER"
        
        # အမြတ်နဲ့ ပိတ်ပြီး peak က +8% အထက်ဆိုရင် → အရမ်းကောင်းတဲ့ အပြုအမူ
        elif pnl > 0 and peak_pnl_pct >= 8.0:
            return "GOOD_WINNER"
        
        # သာမန် အမြတ်ထွက်တဲ့ trade
        elif pnl > 0:
            return "PURE_WINNER"
        
        # SL ထိပြီး ရှုံးရင် → သီးသန့် အမျိုးအစား
        elif "STOP_LOSS" in close_reason or "STOP" in close_reason:
            return "STOP_LOSS_MISTAKE"
        
        # ကျန်တာ အကုန် သာမန် ရှုံးတာ
        else:
            return "PURE_LOSER"
    except Exception as e:
        print(f"❌ [CLASSIFICATION ERROR] {e}")
        return "UNKNOWN"

def log_trade_for_ml(trade_data, market_data=None):
    """
    ဘယ် trade ပဲဖြစ်ဖြစ် (Winner, Loser, Partial, Winner-Turn-Loser) အကုန် auto log
    တစ်ခါမှ run ပေးစရာ မလိုတော့ဘူး — သူ့ဘာသာသူ သိမ်းတယ်
    """
    try:
        if market_data is None:
            market_data = {}

        # === FIX: Handle missing fields gracefully ===
        # Peak PnL % 
        peak_pnl_pct = trade_data.get("peak_pnl_pct", 0.0)
        if peak_pnl_pct == 0 and "peak_pnl" in trade_data:
            peak_pnl_pct = trade_data["peak_pnl"]

        # PnL % တွက်ပေး (လိုအပ်ရင်)
        if "pnl_percent" not in trade_data:
            try:
                entry_price = trade_data.get('entry_price', 0)
                exit_price = trade_data.get('exit_price', 0)
                leverage = trade_data.get('leverage', 1)
                
                if entry_price > 0 and exit_price > 0:
                    if trade_data.get('direction') == 'LONG':
                        trade_data['pnl_percent'] = ((exit_price - entry_price) / entry_price) * 100 * leverage
                    else:
                        trade_data['pnl_percent'] = ((entry_price - exit_price) / entry_price) * 100 * leverage
                else:
                    trade_data['pnl_percent'] = 0
            except Exception as e:
                print(f"❌ [PNL CALC ERROR] {e}")
                trade_data['pnl_percent'] = 0

        # Intelligent classification
        outcome = classify_trade_outcome(trade_data)

        # === FIX: Ensure all required fields exist with safe defaults ===
        row = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "unix_time": trade_data.get("close_timestamp", time.time()),
            "pair": trade_data.get("pair", "UNKNOWN"),
            "direction": 1 if trade_data.get("direction") == "LONG" else 0,
            "entry_price": float(trade_data.get("entry_price", 0)),
            "exit_price": float(trade_data.get("exit_price", 0)),
            "pnl_usd": float(trade_data.get("pnl", 0)),
            "pnl_percent": round(trade_data.get('pnl_percent', 0), 3),
            "peak_pnl_pct": round(peak_pnl_pct, 3),
            "outcome_class": outcome,
            "leverage": trade_data.get("leverage", 5),
            "position_size_usd": float(trade_data.get("position_size_usd", 50.0)),
            "loss_percent": round(abs(trade_data.get("pnl", 0)) / trade_data.get("position_size_usd", 50.0) * 100, 2) if trade_data.get("pnl", 0) < 0 else 0,
            "atr_percent": market_data.get("atr_percent", 0.0),
            "volatility_spike": 1 if market_data.get("atr_percent", 0) > 3.0 else 0,
            "trend_strength": market_data.get("trend_strength", 0.0),
            "rsi": market_data.get("rsi", 50),
            "volume_change": market_data.get("volume_change", 0.0),
            "news_impact": 1 if market_data.get("news_impact", False) else 0,
            "sl_distance_pct": market_data.get("sl_distance_pct", 0.0),
            "close_reason": trade_data.get("close_reason", "MANUAL"),
            "is_partial_close": 1 if trade_data.get("partial_percent", 100) < 100 else 0,
            "partial_percent": trade_data.get("partial_percent", 100),
            "is_winner": 1 if trade_data.get("pnl", 0) > 0 else 0,
            "is_mistake": 1 if outcome in ["WINNER_TURN_LOSER", "STOP_LOSS_MISTAKE"] else 0
        }

        # CSV ထဲ ရေးထည့်
        file_exists = os.path.exists(DATA_FILE)
        
        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # === MISTAKE DETECTION & LEARNING FEEDBACK (2025 Ultra Pro Edition) ===
        if row["is_mistake"]:
            print(f"\033[91m{'═' * 70}\033[0m")
            print(f"‼️  MISTAKE DETECTED → AI IS LEARNING FROM THIS FAILURE!")
            print(f"   Pair         : {row['pair']:>12}")
            print(f"   Type         : {row['outcome_class']:>25}")
            print(f"   Peak Profit  : +{row['peak_pnl_pct']:>6.2f}%")
            print(f"   Final PnL    : ${row['pnl_usd']:>8.2f}")
            print(f"   → AI has memorized this losing pattern and will avoid it in the future.")
            print(f"\033[91m{'═' * 70}\033[0m\n")
        else:
            outcome = row['outcome_class']
            icon = {
                "GOOD_WINNER": "TROPHY",
                "PURE_WINNER": "CHECKMARK",
                "PURE_LOSER": "CROSS",
                "UNKNOWN": "QUESTION MARK"
            }.get(outcome, "DIAMOND")
            print(f"{icon} Trade logged → {row['pair']:>10} | PnL: ${row['pnl_usd']:>7.2f} | {outcome}\n")
        
        # Success message with better formatting
        icon_map = {
            "GOOD_WINNER": "🏆",
            "WINNER_TURN_LOSER": "💔", 
            "STOP_LOSS_MISTAKE": "⚠️",
            "PURE_WINNER": "✅",
            "PURE_LOSER": "❌",
            "UNKNOWN": "❓"
        }

        color_map = {
            "GOOD_WINNER": "\033[96m",      # Cyan
            "WINNER_TURN_LOSER": "\033[91m", # Red  
            "STOP_LOSS_MISTAKE": "\033[93m", # Yellow
            "PURE_WINNER": "\033[92m",       # Green
            "PURE_LOSER": "\033[91m",        # Red
            "UNKNOWN": "\033[97m"            # White
        }

        icon = icon_map.get(outcome, "❓")
        color = color_map.get(outcome, "\033[97m")
        
        pair = trade_data.get('pair', 'UNKNOWN')
        pnl_value = trade_data.get('pnl', 0)
        close_reason_display = trade_data.get('close_reason', 'N/A')
        
        print(f"{color}{icon} [AUTO ML LOG] {outcome:<20} | {pair:8} | "
              f"Peak: +{peak_pnl_pct:>5.1f}% → Final: ${pnl_value:>7.2f} | "
              f"{close_reason_display}\033[0m")
              
        return True

    except KeyError as e:
        print(f"❌ [ML KEY ERROR] Missing field: {e}")
        print(f"🔧 Available fields: {list(trade_data.keys())}")
        return False
    except Exception as e:
        print(f"❌ [ML CRITICAL ERROR] {e}")
        return False

def get_dataset_stats():
    """လက်ရှိ သင်ယူထားတဲ့ data ဘယ်လောက်ရှိပြီလဲ ကြည့်လို့ရတယ်"""
    if not os.path.exists(DATA_FILE):
        return "📊 No ML data yet - Waiting for first trade..."
    
    try:
        import pandas as pd
        df = pd.read_csv(DATA_FILE)
        stats = df['outcome_class'].value_counts()
        total = len(df)
        
        # Calculate win rate
        winning_trades = len(df[df['is_winner'] == 1])
        win_rate = (winning_trades / total * 100) if total > 0 else 0
        
        stats_text = f"📊 ML Dataset: {total} trades | Win Rate: {win_rate:.1f}%\n"
        for outcome, count in stats.items():
            stats_text += f"   {outcome}: {count}\n"
        
        return stats_text
    except Exception as e:
        return f"📊 Error reading ML data: {e}"

def backup_ml_data():
    """Backup ML data file"""
    try:
        if os.path.exists(DATA_FILE):
            backup_file = f"ml_training_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            import shutil
            shutil.copy2(DATA_FILE, backup_file)
            print(f"✅ ML data backed up to: {backup_file}")
            return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
    return False

# Test function to verify everything works
def test_data_collector():
    """Test the data collector with sample data"""
    print("🧪 Testing Data Collector...")
    
    sample_trade = {
        "pair": "TESTUSDT",
        "direction": "LONG", 
        "entry_price": 100.0,
        "exit_price": 105.0,
        "pnl": 25.0,
        "leverage": 5,
        "position_size_usd": 100.0,
        "close_reason": "TEST",
        "close_timestamp": time.time(),
        "peak_pnl_pct": 8.5,
        "partial_percent": 100
    }
    
    success = log_trade_for_ml(sample_trade)
    if success:
        print("✅ Data Collector Test: PASSED")
        print(get_dataset_stats())
    else:
        print("❌ Data Collector Test: FAILED")

# Test
if __name__ == "__main__":
    print("🚀 data_collector.py ready | Winner-Turn-Loser Detection: ENABLED")
    print("=" * 60)
    
    # Run test
    test_data_collector()
    
    print("=" * 60)
    print("📝 Usage: from data_collector import log_trade_for_ml")
    print("💡 Just call log_trade_for_ml(trade_data) after each trade!")
