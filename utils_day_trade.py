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
PROFIT_UPPER = 0.01
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
            buy_price = row["close"]
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
            tp_conditions = {
                "Profit > 1.0%": change_pct >= PROFIT_UPPER,
                "RSI peak (>70 then drop)": row["rsi"] > 70 and row["rsi"] < rsi_prev,
                "Above VWAP by >2.5%": vw_delta > 0.025
            }

            triggered_tp = [reason for reason, cond in tp_conditions.items() if cond]

            if triggered_tp:
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
                        "Stop_Reasons": "; ".join(triggered_tp),
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
                        "Stop_Reasons": "; ".join(triggered_reasons),
                    })
                    buy_datetime = None

    return df, trades


def get_latest_data(TICKER,start_date=START_DATE):
    now = datetime.utcnow().replace(microsecond=0)
    start = start_date + 'T09:30:00Z'
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
            buy_price = row["close"]
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

            # 止盈条件
            tp_conditions = {
                "Profit > 1.0%": change_pct >= PROFIT_UPPER,
                "RSI peak (>70 then drop)": row["rsi"] > 70 and row["rsi"] < rsi_prev,
                "Above VWAP by >2.5%": vw_delta > 0.025
            }

            triggered_tp = [reason for reason, cond in tp_conditions.items() if cond]

            if triggered_tp:
                df.at[time, "sell_signal_raw"] = -1
                if i + delay < len(df):
                    df.iat[i + delay, df.columns.get_loc("sell_signal")] = -1
                    sell_row = df.iloc[i + delay]
                    sell_time = sell_row.name
                    sell_price = sell_row["close"]
                    sell_datetime = sell_row["timestamp"] if "timestamp" in sell_row else sell_time
                else:
                    print(time)
                    print('top profit:')
                    print('warning: cannot delay as out of bound')
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
                    "Stop_Reasons": "; ".join(triggered_tp)
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
                    print(time)
                    print('top loss:')
                    print('warning: cannot delay as out of bound')
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
                    "Stop_Reasons": "; ".join(triggered_reasons),
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
            print(f"🟢 {dt_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} @ {row['close']:.2f}")

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
    if ('ema9' in df.columns) and  ('ema21' in df.columns) and  ('vwap' in df.columns):
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



def max_consecutive_losses(trades_df):
    """
    Calculate the maximum number of consecutive losing trades.
    
    Input:
        trades_df: pd.DataFrame with a 'Return' column (float)
    Returns:
        max_loss_streak: int - the longest consecutive loss streak
    """
    max_loss_streak = 0
    current_streak = 0

    for r in trades_df['Return']:
        if r < 0:
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0  # reset streak on a win

    return max_loss_streak


def generate_rsi3070_signals_and_backtest_delay(df, delay=3, low_rsi=30, high_rsi=70, profit_thresh=0.01, loss_thresh=-0.005):
    """
    Generate buy when RSI crosses 30 (with 2 up bars, volume increase, and increasing EMA divergence),
    sell when RSI crosses 70, or hits profit/loss threshold.
    All signals are delayed by `delay` bars.
    """

    # --- RSI calculation ---
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- EMA calculation ---
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema_diff"] = df["ema9"] - df["ema21"]
    df["ema_diff_prev"] = df["ema_diff"].shift(1)

    # --- Volume and price momentum ---
    df["up1"] = df["close"].diff(1).shift(1) > 0
    df["up2"] = df["close"].diff(1).shift(2) > 0
    df["vol_up"] = df["volume"].shift(1) > df["volume"].shift(2)

    # --- Signal initialization ---
    df["signal_raw"] = 0
    df["signal"] = 0
    df["sell_signal_raw"] = 0
    df["sell_signal"] = 0
    df["rsi_prev"] = df["rsi"].shift(1)

    # --- Buy condition: RSI crosses 30 + up bars + vol_up + EMA9-EMA21 gap widening ---
    # df.loc[
    #     (df["rsi_prev"] < low_rsi) &
    #     (df["rsi"] >= low_rsi) &
    #     df["up1"] & df["up2"] &
    #     df["vol_up"] ,
    #     #(df["ema_diff"] > 0),
    #     #(df["ema_diff"] > df["ema_diff_prev"]),
    #     "signal_raw"
    # ] = 1
    # --- Buy condition: RSI crosses 30 + up bars + vol_up + EMA9-EMA21 gap widening ---
    df.loc[
        (df["rsi_prev"] < low_rsi) &
        (df["rsi"] >= low_rsi) &
        df["vol_up"] &
        (df["ema_diff"] > df["ema_diff_prev"]),
        "signal_raw"
    ] = 1

    # --- Delayed Buy Signal ---
    df["signal"] = df["signal_raw"].shift(delay).fillna(0)

    # --- Backtest ---
    holding = False
    buy_price = 0
    buy_datetime = None
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        time = row.name

        # Execute delayed buy
        if row["signal"] == 1 and not holding:
            holding = True
            buy_price = row["close"]
            buy_datetime = row["timestamp"] if "timestamp" in row else time

        elif holding:
            current_price = row["close"]
            change_pct = (current_price - buy_price) / buy_price
            rsi = row["rsi"]

            # --- Sell logic ---
            sell_reason = None
            if rsi >= high_rsi:
                sell_reason = "RSI ≥ "+str(high_rsi)
            elif change_pct >= profit_thresh:
                sell_reason = f"Profit ≥ {profit_thresh:.1%}"
            elif change_pct <= loss_thresh:
                sell_reason = f"Loss ≤ {loss_thresh:.1%}"
            elif row["ema_diff"] < row["ema_diff_prev"]:
                sell_reason = "EMA diff shrinking"

            if sell_reason:
                df.at[time, "sell_signal_raw"] = -1
                if i + delay < len(df):
                    df.iat[i + delay, df.columns.get_loc("sell_signal")] = -1
                    sell_row = df.iloc[i + delay]
                    sell_time = sell_row.name
                    sell_price = sell_row["close"]
                    sell_datetime = sell_row["timestamp"] if "timestamp" in sell_row else sell_time
                else:
                    print(time)
                    print(f'❗ {sell_reason} triggered but cannot delay (out of bound)')
                    sell_datetime = row["timestamp"] if "timestamp" in row else time
                    sell_price = current_price
                    df.at[time, "sell_signal"] = -1

                trades.append({
                    "Buy_Time": buy_datetime,
                    "Sell_Time": sell_datetime,
                    "Buy_Price": buy_price,
                    "Sell_Price": sell_price,
                    "Return": (sell_price - buy_price) / buy_price,
                    "Sell_Reason": sell_reason
                })
                holding = False
                buy_datetime = None

    return df, trades


def check_recent_deals(df,trades,show_reason=1):
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
    if show_reason:
        df_trades = pd.DataFrame(trades)
        # Step 1: Localize to UTC if not already
        if df_trades["Buy_Time"].dt.tz is None or df_trades["Sell_Time"].dt.tz is None:
            df_trades["Buy_Time"] = df_trades["Buy_Time"].dt.tz_localize("UTC")
            df_trades["Sell_Time"] = df_trades["Sell_Time"].dt.tz_localize("UTC")
        # Step 2: Convert to US Eastern Time (handles daylight saving)
        df_trades["Buy_Time"] = df_trades["Buy_Time"].dt.tz_convert("US/Eastern")
        df_trades["Sell_Time"] = df_trades["Sell_Time"].dt.tz_convert("US/Eastern")
        print(df_trades.tail(5))

def generate_rsi_dynamic_signals_and_backtest_delay(df, delay=3, low_rsi_base=30, high_rsi=70, profit_thresh=0.01, loss_thresh=-0.005):
    """
    Generate buy when RSI crosses 30 (with volume increase and EMA divergence),
    dynamically adjust low_rsi using close prices around buy signal.
    Sell when RSI reaches 70, profit/loss threshold, or EMA diff shrinks.
    All executions use 'close' prices.
    """

    # --- RSI calculation ---
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- EMA calculation ---
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema_diff"] = df["ema9"] - df["ema21"]
    df["ema_diff_prev"] = df["ema_diff"].shift(1)

    # --- Volume condition ---
    df["vol_up"] = df["volume"].shift(1) > df["volume"].shift(2)

    # --- Signal initialization ---
    df["signal_raw"] = 0
    df["signal"] = 0
    df["sell_signal_raw"] = 0
    df["sell_signal"] = 0
    df["rsi_prev"] = df["rsi"].shift(1)
    base_rsi = low_rsi_base
    hard_lower = 35
    hard_upper = 50
    # --- Iterate to detect signal with dynamic RSI adjustment ---
    for i in range(3, len(df)):  # Ensure enough context before/after
        if base_rsi<hard_lower or base_rsi>hard_upper:
            base_rsi = 40
        rsi_prev = df.at[df.index[i - 1], "rsi"]
        rsi_now = df.at[df.index[i], "rsi"]

        if pd.notna(rsi_prev) and pd.notna(rsi_now) and rsi_prev < base_rsi <= rsi_now:
            close_now = df.at[df.index[i], "close"]
            closes_before = df["close"].iloc[i - 3:i]

            # Adjust low_rsi downward if prior 3 closes are all lower
            if all(cl < close_now for cl in closes_before):
                low_close_rsi = df["rsi"].iloc[i - 3:i].loc[closes_before.idxmin()]
                if low_close_rsi < base_rsi:
                    base_rsi -= 0.1 * abs(low_close_rsi - base_rsi)
            else:
                base_rsi += 0.1
            

            # Final buy condition check
            if (
                df.at[df.index[i - 1], "rsi"] < base_rsi and
                df.at[df.index[i], "rsi"] >= base_rsi and
                df.at[df.index[i - 1], "ema_diff"] < df.at[df.index[i], "ema_diff"] and
                df.at[df.index[i], "vol_up"]
            ):
                df.at[df.index[i], "signal_raw"] = 1
            
    # --- Delayed Buy Signal ---
    df["signal"] = df["signal_raw"].shift(delay).fillna(0)

    # --- Backtest ---
    holding = False
    buy_price = 0
    buy_datetime = None
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        time = row.name

        if row["signal"] == 1 and not holding:
            holding = True
            buy_price = row["close"]
            buy_datetime = row["timestamp"] if "timestamp" in row else time

        elif holding:
            current_price = row["close"]
            change_pct = (current_price - buy_price) / buy_price
            rsi = row["rsi"]

            # --- Sell logic ---
            sell_reason = None
            if rsi >= high_rsi:
                sell_reason = f"RSI ≥ {high_rsi}"
            elif change_pct >= profit_thresh:
                sell_reason = f"Profit ≥ {profit_thresh:.1%}"
            elif change_pct <= loss_thresh:
                sell_reason = f"Loss ≤ {loss_thresh:.1%}"
            elif row["ema_diff"] < row["ema_diff_prev"]:
                sell_reason = "EMA diff shrinking"

            if sell_reason:
                df.at[time, "sell_signal_raw"] = -1
                if i + delay < len(df):
                    df.iat[i + delay, df.columns.get_loc("sell_signal")] = -1
                    sell_row = df.iloc[i + delay]
                    sell_time = sell_row.name
                    sell_price = sell_row["close"]
                    sell_datetime = sell_row["timestamp"] if "timestamp" in sell_row else sell_time
                else:
                    print(time)
                    print(f'❗ {sell_reason} triggered but cannot delay (out of bound)')
                    sell_datetime = row["timestamp"] if "timestamp" in row else time
                    sell_price = current_price
                    df.at[time, "sell_signal"] = -1

                trades.append({
                    "Buy_Time": buy_datetime,
                    "Sell_Time": sell_datetime,
                    "Buy_Price": buy_price,
                    "Sell_Price": sell_price,
                    "Return": (sell_price - buy_price) / buy_price,
                    "Sell_Reason": sell_reason
                })
                holding = False
                buy_datetime = None

    return df, trades,base_rsi
