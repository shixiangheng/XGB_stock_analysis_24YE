import pandas as pd
import matplotlib.pyplot as plt
import time
import winsound
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import pytz
eastern = pytz.timezone('US/Eastern')
API_KEY = 'PK55SJLN5X7L81OB9D1P'
SECRET_KEY = '3iWrQIdPWUlbNFV1wNcBzGx2JReYbMEQOXtx9rH0'
BASE_URL = 'https://paper-api.alpaca.markets'
#TICKER = 'TSLL'
START_DATE = '2025-07-01'

# 初始化API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')


def generate_signals_and_backtest(df):
    """
    Add EMA, VWAP, RSI, and generate buy/sell signals with trade backtest records.
    Returns: updated df, trades list
    """
    # 1. Technical Indicators
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 2. Signal logic
    df["signal"] = 0
    df["sell_signal"] = 0
    df["up1"] = df["close"].diff(1).shift(1) > 0
    df["up2"] = df["close"].diff(1).shift(2) > 0
    df["volup"] = df["volume"].shift(1) > df["volume"].shift(2)

    signal_condition = (
     (df["ema9"] > df["ema21"]) &
        (df["ema9"].shift(1) <= df["ema21"].shift(1)) &
        (df["rsi"].between(50, 70)) &
        df["up1"] & df["up2"] & df["volup"]
    )

    df.loc[
    signal_condition,
    "signal"
    ] = 1
    # 3. Backtest logic
    holding = False
    buy_price = 0
    buy_datetime = None
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        time = row.name
        if row["signal"] == 1 and not holding:
            holding = True
            buy_price = row["open"]
            buy_datetime = row["timestamp"] if "timestamp" in row else time

        elif holding:
            current_price = row["close"]
            change_pct = (current_price - buy_price) / buy_price
            try:
                try:
                    duration = (pd.to_datetime(time) - pd.to_datetime(buy_datetime)).total_seconds() / 60
                except:
                    duration = ((time) - (buy_datetime)) / 60
            except Exception:
                duration = 0

            vw_delta = (current_price - row["vwap"]) / row["vwap"]
            rsi_prev = df["rsi"].iloc[i - 1] if i > 0 else row["rsi"]

            # Take-Profit
            tp1 = change_pct >= 0.015
            tp2 = row["rsi"] > 70 and row["rsi"] < rsi_prev
            tp3 = vw_delta > 0.025

            if tp1 or tp2 or tp3:
                df.at[time, "sell_signal"] = -1
                sell_datetime = row["timestamp"] if "timestamp" in row else time
                holding = False
                if buy_datetime and sell_datetime:
                    trades.append({
                        "Buy_Time": buy_datetime,
                        "Sell_Time": sell_datetime,
                        "Buy_Price": buy_price,
                        "Sell_Price": current_price,
                        "Return": change_pct
                    })
                    buy_datetime = None
                continue

            
            sl_conditions = {
                "loss > 0.5%": change_pct <= -0.005,
                "EMA9 < EMA21": row["ema9"] <= row["ema21"],
                "RSI < 50": row["rsi"] <= 50,
                "Close < VWAP": row["close"] < row["vwap"],
                "No rise in 15min": duration > 15
            }

            # 收集满足的止损原因
            triggered_reasons = [reason for reason, condition in sl_conditions.items() if condition]
            if len(triggered_reasons) >= 2:
                df.at[time, "sell_signal"] = -1
                sell_datetime = row["timestamp"] if "timestamp" in row else time
                holding = False
                if buy_datetime and sell_datetime:
                    trades.append({
                        "Buy_Time": buy_datetime,
                        "Sell_Time": sell_datetime,
                        "Buy_Price": buy_price,
                        "Sell_Price": current_price,
                        "Return": change_pct,
                        "Stop_Loss_Reasons": "; ".join(triggered_reasons),
                    })
                    buy_datetime = None

    return df, trades


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



def generate_signals_and_backtest_delay(df, delay=3):
    """
    Add EMA, VWAP, RSI, and generate buy/sell signals with delay and backtest records.
    Returns: updated df, trades list
    """
    # 1. Technical Indicators
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 2. Signal logic
    df["signal_raw"] = 0  # 原始信号
    df["signal"] = 0      # 延迟后的信号
    df["sell_signal_raw"] = 0
    df["sell_signal"] = 0

    df["up1"] = df["close"].diff(1).shift(1) > 0
    df["up2"] = df["close"].diff(1).shift(2) > 0
    df["volup"] = df["volume"].shift(1) > df["volume"].shift(2)

    signal_condition = (
        (df["ema9"] > df["ema21"]) &
        (df["ema9"].shift(1) <= df["ema21"].shift(1)) &
        (df["rsi"].between(50, 70)) &
        df["up1"] & df["up2"] & df["volup"]
    )

    df.loc[signal_condition, "signal_raw"] = 1

    # 延迟买入信号
    df["signal"] = df["signal_raw"].shift(delay).fillna(0)

    # 3. Backtest logic
    holding = False
    buy_price = 0
    buy_datetime = None
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        time = row.name

        if row["signal"] == 1 and not holding:
            holding = True
            buy_price = row["open"]
            buy_datetime = row["timestamp"] if "timestamp" in row else time

        elif holding:
            current_price = row["close"]
            change_pct = (current_price - buy_price) / buy_price
            try:
                duration = (pd.to_datetime(time) - pd.to_datetime(buy_datetime)).total_seconds() / 60
            except:
                duration = 0

            vw_delta = (current_price - row["vwap"]) / row["vwap"]
            rsi_prev = df["rsi"].iloc[i - 1] if i > 0 else row["rsi"]

            # Take-Profit
            tp1 = change_pct >= 0.015
            tp2 = row["rsi"] > 70 and row["rsi"] < rsi_prev
            tp3 = vw_delta > 0.025

            if tp1 or tp2 or tp3:
                df.at[time, "sell_signal_raw"] = -1
                # Apply sell delay
                if i + delay < len(df):
                    df.iat[i + delay, df.columns.get_loc("sell_signal")] = -1
                    sell_row = df.iloc[i + delay]
                    sell_time = sell_row.name
                    sell_price = sell_row["close"]
                    sell_datetime = sell_row["timestamp"] if "timestamp" in sell_row else sell_time
                else:
                    # If delayed bar doesn't exist yet
                    sell_datetime = row["timestamp"] if "timestamp" in row else time
                    sell_price = current_price
                    df.at[time, "sell_signal"] = -1

                holding = False
                trades.append({
                    "Buy_Time": buy_datetime,
                    "Sell_Time": sell_datetime,
                    "Buy_Price": buy_price,
                    "Sell_Price": sell_price,
                    "Return": (sell_price - buy_price) / buy_price
                })
                buy_datetime = None
                continue

            # Stop-Loss
            sl_conditions = {
                "loss > 0.5%": change_pct <= -0.005,
                "EMA9 < EMA21": row["ema9"] <= row["ema21"],
                "RSI < 50": row["rsi"] <= 50,
                "Close < VWAP": row["close"] < row["vwap"],
                "No rise in 15min": duration > 15
            }

            triggered_reasons = [k for k, v in sl_conditions.items() if v]
            if len(triggered_reasons) >= 2:
                df.at[time, "sell_signal_raw"] = -1
                if i + delay < len(df):
                    df.iat[i + delay, df.columns.get_loc("sell_signal")] = -1
                    sell_row = df.iloc[i + delay]
                    sell_time = sell_row.name
                    sell_price = sell_row["close"]
                    sell_datetime = sell_row["timestamp"] if "timestamp" in sell_row else sell_time
                else:
                    sell_datetime = row["timestamp"] if "timestamp" in row else time
                    sell_price = current_price
                    df.at[time, "sell_signal"] = -1

                holding = False
                trades.append({
                    "Buy_Time": buy_datetime,
                    "Sell_Time": sell_datetime,
                    "Buy_Price": buy_price,
                    "Sell_Price": sell_price,
                    "Return": (sell_price - buy_price) / buy_price,
                    "Stop_Loss_Reasons": "; ".join(triggered_reasons),
                })
                buy_datetime = None

    return df, trades

def backtest_plot(ticker,df,trades):

    results = pd.DataFrame(trades)
    if not results.empty:
        print("🔎 回测总结")
        print(results)
        recent_buys = df[df["signal"] == 1].tail(5)
        recent_sells = df[df["sell_signal"] == -1].tail(5)

        print("\n📋 Recent Buy Signals:")
        for _, row in recent_buys.iterrows():
            dt = row["timestamp"] if "timestamp" in row else row["Datetime"]
            dt = pd.to_datetime(dt)
            dt_eastern = dt.astimezone(eastern)
            print(f"🟢 {dt_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} @ {row['open']:.2f}")

        print("\n📋 Recent Sell Signals:")
        for _, row in recent_sells.iterrows():
            dt = row["timestamp"] if "timestamp" in row else row["Datetime"]
            dt = pd.to_datetime(dt)
            dt_eastern = dt.astimezone(eastern)
            print(f"🔴 {dt_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} @ {row['close']:.2f}")
        print(f"✅ 总交易次数: {len(results)}")
        print(f"🏆 胜率: {round((results['Return'] > 0).sum() / len(results) * 100, 2)}%")
        print(f"📈 总收益: {round(results['Return'].sum() * 100, 2)}%")
        print(f"💰 每笔平均收益: {round(results['Return'].mean() * 100, 2)}%")


    # 提取买卖点
    buy_signals = df[df["signal"] == 1]
    sell_signals = df[df["sell_signal"] == -1]

    # 绘图展示
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["close"], label="close", color="black")
    plt.plot(df.index, df["ema9"], label="EMA9", linestyle="--")
    plt.plot(df.index, df["ema21"], label="EMA21", linestyle="--")
    plt.plot(df.index, df["vwap"], label="VWAP", color="purple")
    plt.scatter(buy_signals.index, buy_signals["close"], marker="^", color="green", label="Buy", s=100)
    plt.scatter(sell_signals.index, sell_signals["close"], marker="v", color="red", label="Sell", s=100)
    plt.title(f"{ticker} - Strategy with Multi-Exit Conditions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()