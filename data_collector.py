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

def log_trade_for_ml(trade_data, market_data=None):
    """
    ဘယ် trade ပဲဖြစ်ဖြစ် (Winner, Loser, Partial, Winner-Turn-Loser) အကုန် auto log
    တစ်ခါမှ run ပေးစရာ မလိုတော့ဘူး — သူ့ဘာသာသူ သိမ်းတယ်
    """
    if market_data is None:
        market_data = {}

    # Peak PnL % တွက်ထည့်ပေး (bot.py က ပို့ပေးမယ်)
    peak_pnl_pct = trade_data.get("peak_pnl_pct", 0.0)

    # PnL % တွက်ပေး (လိုအပ်ရင်)
    if "pnl_percent" not in trade_data:
        if trade_data['direction'] == 'LONG':
            trade_data['pnl_percent'] = (trade_data['exit_price'] - trade_data['entry_price']) / trade_data['entry_price'] * 100 * trade_data.get('leverage', 1)
        else:
            trade_data['pnl_percent'] = (trade_data['entry_price'] - trade_data['exit_price']) / trade_data['entry_price'] * 100 * trade_data.get('leverage', 1)

    # Intelligent classification
    outcome = classify_trade_outcome(trade_data)

    row = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "unix_time": trade_data.get("close_timestamp", time.time()),
        "pair": trade_data["pair"],
        "direction": 1 if trade_data["direction"] == "LONG" else 0,
        "entry_price": float(trade_data["entry_price"]),
        "exit_price": float(trade_data["exit_price"]),
        "pnl_usd": float(trade_data["pnl"]),
        "pnl_percent": round(trade_data['pnl_percent'], 3),
        "peak_pnl_pct": round(peak_pnl_pct, 3),
        "outcome_class": outcome,  # အသစ်ထည့်ထားတာ — အဓိက!
        "leverage": trade_data.get("leverage", 5),
        "position_size_usd": float(trade_data.get("position_size_usd", 50.0)),
        "loss_percent": round(abs(trade_data["pnl"]) / trade_data.get("position_size_usd", 50.0) * 100, 2),
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
        "is_winner": 1 if trade_data["pnl"] > 0 else 0,
        "is_mistake": 1 if outcome in ["WINNER_TURN_LOSER", "STOP_LOSS_MISTAKE"] else 0  # ML က ဒါတွေကို အဓိက ရှောင်ရှားမယ်
    }

    # CSV ထဲ ရေးထည့်
    file_exists = os.path.exists(DATA_FILE)
    try:
        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        # Console မှာ ချွေတာတယ်
        icon = {
            "GOOD_WINNER": "TROPHY",
            "WINNER_TURN_LOSER": "BROKEN HEART",
            "STOP_LOSS_MISTAKE": "WARNING",
            "PURE_WINNER": "WIN",
            "PURE_LOSER": "LOSS"
        }.get(outcome, "QUESTION")

        color_meaning = {
            "GOOD_WINNER": "\033[96m",          # Cyan
            "WINNER_TURN_LOSER": "\033[91m",     # Red
            "STOP_LOSS_MISTAKE": "\033[93m",     # Yellow
            "PURE_WINNER": "\033[92m",           # Green
            "PURE_LOSER": "\033[91m"             # Red
        }.get(outcome, "\033[97m")

        print(f"{color_meaning}[AUTO LOG] {icon} {outcome:<18} | {trade_data['pair']:8} | "
              f"Peak: +{peak_pnl_pct:>5.1f}% → Final: {trade_data['pnl']:>6.2f}$ | "
              f"{trade_data.get('close_reason', 'N/A')}\033[0m")

    except Exception as e:
        print(f"[ERROR] Failed to log trade: {e}")

def get_dataset_stats():
    """လက်ရှိ သင်ယူထားတဲ့ data ဘယ်လောက်ရှိပြီလဲ ကြည့်လို့ရတယ်"""
    if not os.path.exists(DATA_FILE):
        return "No data yet"
    
    import pandas as pd
    try:
        df = pd.read_csv(DATA_FILE)
        stats = df['outcome_class'].value_counts()
        total = len(df)
        return f"Total: {total} | {dict(stats)}"
    except:
        return "Reading error"

# Test
if __name__ == "__main__":
    print("data_collector.py ready | Winner-Turn-Loser Detection: ENABLED")
    print(get_dataset_stats())
