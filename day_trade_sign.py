import pandas as pd
import matplotlib.pyplot as plt
import time
import winsound
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import pytz
from utils_day_trade import generate_signals_and_backtest,generate_rsi3070_signals_and_backtest_delay
eastern = pytz.timezone('US/Eastern')
show_reason = 1
# 用户填入自己的 API 认证
API_KEY = 'PK55SJLN5X7L81OB9D1P'
SECRET_KEY = '3iWrQIdPWUlbNFV1wNcBzGx2JReYbMEQOXtx9rH0'
BASE_URL = 'https://paper-api.alpaca.markets'
#TICKER = 'TSLL'
START_DATE = '2025-07-07'

# 初始化API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def play_alert(signal_type):
    if signal_type == "buy":
        winsound.Beep(1000, 500)
    elif signal_type == "sell":
        winsound.Beep(500, 500)

def get_latest_data(TICKER):
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
    
    #df, trades = generate_signals_and_backtest(df)
    df,trades = generate_rsi3070_signals_and_backtest_delay(df,0,40,60)
    # 🔔 只对最新K线触发信号播放声音
    latest = df.iloc[-1]
    # 转时区
    latest_time_eastern = latest.name.tz_convert(eastern) if latest.name.tzinfo else pytz.utc.localize(latest.name).astimezone(eastern)

    if latest["signal"] == 1:
        print(f"📈 BUY SIGNAL @ {latest_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}, close = {latest['open']:.2f}")
        play_alert("buy")
    elif latest["sell_signal"] == -1:
        print(f"💰 SELL SIGNAL @ {latest_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}, close = {latest['close']:.2f}")
        play_alert("sell")
    if show_reason:
        print(trades)
    # 🧾 显示最近5个买卖信号
    recent_buys = df[df["signal"] == 1].tail(5)
    recent_sells = df[df["sell_signal"] == -1].tail(5)

    print("\n📋 Recent Buy Signals:")
    for idx, row in recent_buys.iterrows():
        # 转时区
        idx_eastern = idx.tz_convert(eastern) if idx.tzinfo else pytz.utc.localize(idx).astimezone(eastern)
        print(f"🟢 {idx_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} @ {row['open']:.2f}")

    print("\n📋 Recent Sell Signals:")
    for idx, row in recent_sells.iterrows():
        idx_eastern = idx.tz_convert(eastern) if idx.tzinfo else pytz.utc.localize(idx).astimezone(eastern)
        print(f"🔴 {idx_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} @ {row['close']:.2f}")

def main_loop():
    while True:
        print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Checking...")
        for TICKER in ['TSLL','NVDL']:
            try:
                df = get_latest_data(TICKER)
                print(f"------------------Below For {TICKER}: ---------------")
                apply_strategy(df)
            except Exception as e:
                print(f"❌ Error: {e}")
            #print(f"!!!!!!!!!!!!!!! Above For {TICKER}: !!!!!!!!!!!!!!")
            print("⏳ Waiting for 5 minutes...\n")
        print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Checking done...")
        time.sleep(200)

if __name__ == "__main__":
    main_loop()