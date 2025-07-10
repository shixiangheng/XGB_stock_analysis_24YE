import pandas as pd
import matplotlib.pyplot as plt
import time
import winsound
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import pytz
from utils_day_trade import (generate_signals_and_backtest,
                             generate_rsi3070_signals_and_backtest_delay,
                             check_recent_deals,
                             generate_rsi_dynamic_signals_and_backtest_delay,)
eastern = pytz.timezone('US/Eastern')
show_reason = 1
# 用户填入自己的 API 认证
API_KEY = 'PK55SJLN5X7L81OB9D1P'
SECRET_KEY = '3iWrQIdPWUlbNFV1wNcBzGx2JReYbMEQOXtx9rH0'
BASE_URL = 'https://paper-api.alpaca.markets'
#TICKER = 'TSLL'
START_DATE = '2024-07-07'

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
    #df,trades = generate_rsi3070_signals_and_backtest_delay(df,0,40,60)
    delay_bars = 3
    df,trades,base_rsi = generate_rsi_dynamic_signals_and_backtest_delay(df,delay_bars,40,60)
    print(f'now lower RSI is {base_rsi}')
    
    
    # 🔔 只对最新K线触发信号播放声音
    latest = df.iloc[-1]
    # 转时区
    latest_time_eastern = latest.name.tz_convert(eastern) if latest.name.tzinfo else pytz.utc.localize(latest.name).astimezone(eastern)
    print("latest bar info:")
    print(f"bar {latest_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}, close price = {latest['close']:.2f}")
    

    latest_index = df.index[-1]
    latest = df.loc[latest_index]
    latest_time_eastern = latest["timestamp"].tz_convert("America/New_York")

    # Define delay window
    lookback = df.iloc[-delay_bars:-1] if delay_bars > 1 else df.iloc[[-2]]
    if latest["signal_raw"] == 1:
        print('latest bar: show raw buy signal!!!')
    if latest["sell_signal_raw"] == -1:
        print('latest bar: raw sell signal identified!!!')

    # 1. Handle BUY signal
    if latest["signal"] == 1:
        future_window = df.iloc[-delay_bars:]
        play_alert("buy")
        if any(future_window["sell_signal_raw"] == -1):
            print(f"⚠️ WARNING: Ignored BUY signal @ {latest_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} – upcoming SELL signal within delay window")  
        else:
            print(f"📈 BUY SIGNAL @ {latest_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}, close = {latest['close']:.2f}")
            

    # 2. Handle SELL signal
    elif latest["sell_signal"] == -1:
        recent_window = lookback
        play_alert("sell")
        if any(recent_window["signal_raw"] == 1):
            print(f"⚠️ WARNING: Ignored SELL signal @ {latest_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} – recent BUY signal within delay window") 
        else:
            print(f"💰 SELL SIGNAL @ {latest_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}, close = {latest['close']:.2f}")      
    check_recent_deals(df,trades,show_reason)


    
    
def main_loop():
    while True:
        print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Checking...")
        for TICKER in ['NVDL']:
            try:
                df = get_latest_data(TICKER)
                print(f"------------------Below For {TICKER}: ---------------")
                apply_strategy(df)
            except Exception as e:
                print(f"❌ Error: {e}")
            #print(f"!!!!!!!!!!!!!!! Above For {TICKER}: !!!!!!!!!!!!!!")
        
        print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Checking done...")
        print("⏳ Waiting for 1 minute...\n")
        time.sleep(60)

if __name__ == "__main__":
    main_loop()