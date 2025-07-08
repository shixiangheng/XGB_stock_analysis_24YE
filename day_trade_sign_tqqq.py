import pandas as pd
import matplotlib.pyplot as plt
import time
import winsound
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import pytz

eastern = pytz.timezone('US/Eastern')

# 用户填入自己的 API 认证
API_KEY = 'PK55SJLN5X7L81OB9D1P'
SECRET_KEY = '3iWrQIdPWUlbNFV1wNcBzGx2JReYbMEQOXtx9rH0'
BASE_URL = 'https://paper-api.alpaca.markets'
TICKER = 'TQQQ'
START_DATE = '2025-07-01'

# 初始化API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def play_alert(signal_type):
    if signal_type == "buy":
        winsound.Beep(1000, 500)
    elif signal_type == "sell":
        winsound.Beep(500, 500)

def get_latest_data():
    now = datetime.utcnow().replace(microsecond=0)
    start = START_DATE + 'T09:30:00Z'
    end = now.isoformat() + 'Z'

    bars = api.get_bars(
        symbol=TICKER,
        timeframe='5Min',
        start=start,
        end=end,
        feed = 'iex'
    ).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars[bars.index.get_level_values('symbol') == TICKER]

    bars.columns = [col.lower() for col in bars.columns]
    bars.index = pd.to_datetime(bars.index)
    bars["timestamp"] = bars.index
    return bars

def apply_strategy(df):
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["signal"] = 0
    df["sell_signal"] = 0
    df["up1"] = df["close"].diff(1).shift(1) > 0
    df["up2"] = df["close"].diff(2).shift(1) > 0
    df["volup"] = df["volume"].shift(1) > df["volume"].shift(2)

    df.loc[
        (df["ema9"] > df["ema21"]) &
        (df["ema9"].shift(1) <= df["ema21"].shift(1)) &
        (df["rsi"].between(50, 70)) &
        df["up1"] & df["up2"] & df["volup"],
        "signal"
    ] = 1

    # 出场逻辑（简化版）
    holding = False
    buy_price = 0
    buy_time = None

    for i in range(len(df)):
        row = df.iloc[i]
        time = row.name

        if row["signal"] == 1 and not holding:
            holding = True
            buy_price = row["open"]
            buy_time = time

        elif holding:
            current_price = row["close"]
            change_pct = (current_price - buy_price) / buy_price
            duration = (time - buy_time).total_seconds() / 60
            vw_delta = (current_price - row["vwap"]) / row["vwap"]
            rsi_prev = df["rsi"].iloc[i - 1] if i > 0 else row["rsi"]

            tp1 = change_pct >= 0.015
            tp2 = row["rsi"] > 70 and row["rsi"] < rsi_prev
            tp3 = vw_delta > 0.025

            if tp1 or tp2 or tp3:
                df.at[time, "sell_signal"] = -1
                holding = False
                continue

            sl_conditions = [
                change_pct <= -0.005,
                row["ema9"] <= row["ema21"],
                row["rsi"] <= 50,
                row["close"] < row["vwap"],
                duration > 15
            ]
            if sum(sl_conditions) >= 2:
                df.at[time, "sell_signal"] = -1
                holding = False

    # 🔔 只对最新K线触发信号播放声音
    latest = df.iloc[-1]
    # 转时区
    latest_time_eastern = latest.name.tz_convert(eastern) if latest.name.tzinfo else pytz.utc.localize(latest.name).astimezone(eastern)

    if latest["signal"] == 1:
        print(f"📈 BUY SIGNAL @ {latest_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}, close = {latest['close']:.2f}")
        play_alert("buy")
    elif latest["sell_signal"] == -1:
        print(f"💰 SELL SIGNAL @ {latest_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}, close = {latest['close']:.2f}")
        play_alert("sell")

    # 🧾 显示最近5个买卖信号
    recent_buys = df[df["signal"] == 1].tail(5)
    recent_sells = df[df["sell_signal"] == -1].tail(5)

    print("\n📋 Recent Buy Signals:")
    for idx, row in recent_buys.iterrows():
        # 转时区
        idx_eastern = idx.tz_convert(eastern) if idx.tzinfo else pytz.utc.localize(idx).astimezone(eastern)
        print(f"🟢 {idx_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} @ {row['close']:.2f}")

    print("\n📋 Recent Sell Signals:")
    for idx, row in recent_sells.iterrows():
        idx_eastern = idx.tz_convert(eastern) if idx.tzinfo else pytz.utc.localize(idx).astimezone(eastern)
        print(f"🔴 {idx_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} @ {row['close']:.2f}")

def main_loop():
    while True:
        print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Checking...")
        try:
            df = get_latest_data()
            apply_strategy(df)
        except Exception as e:
            print(f"❌ Error: {e}")

        print("⏳ Waiting for 5 minutes...\n")
        print(f"!!!!!!!!!!!!!!! For {TICKER}:")
        time.sleep(5 * 60)

if __name__ == "__main__":
    main_loop()