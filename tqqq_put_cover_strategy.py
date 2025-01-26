import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from RSI_threshold_train import calculate_rsi, calculate_rsi_ema
# 参数
ticker_symbol = "TQQQ"
start_date = "2022-01-31"
end_date = "2025-01-31"
shares_held = 27  # 持股数量
risk_free_rate = 0.04  # 无风险利率（年化）
days_to_expiration = 22  # 每次买入期权的到期时间（30天）
rsi_high_bond = 70

# Black-Scholes 期权定价函数和希腊值计算
def black_scholes_greeks(S, K, T, r, sigma, option_type="put"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)
    ) / 252
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    price = (
        S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        if option_type == "call"
        else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    )
    return price, delta, gamma, theta, vega

# 动态隐含波动率模拟函数
def dynamic_iv(close_prices, window=14, scaling_factor=1.5):
    log_returns = np.log(close_prices / close_prices.shift(1))
    rolling_volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
    implied_volatility = rolling_volatility * scaling_factor
    return implied_volatility.fillna(method="bfill")  # 处理缺失值

# 获取历史数据
ticker = yf.Ticker(ticker_symbol)
historical_data = ticker.history(start=start_date, end=end_date)


def remove_timezone(data):
    date_col = 'Date'
    # Check if the DataFrame has an index with datetime and a timezone
    if isinstance(data.index, pd.DatetimeIndex):
        # Check if the datetime index has a timezone
        if data.index.tz is not None:
            # Remove timezone awareness from the index
            data.index = data.index.tz_localize(None)
        # If index has timezone awareness, remove it, otherwise proceed as normal
    else:
        # Check if any datetime column has timezone information
        for column in data.select_dtypes(include=[object, 'datetime']):
            if isinstance(data[column].dtype, pd.Timestamp):
                if data[column].dt.tz is not None:
                    data[column] = data[column].dt.tz_localize(None)
    output_data = data.copy()
    # If index is timezone-aware, reset the index
    if isinstance(data.index, pd.DatetimeIndex):
        if data.index.tz is None:
            output_data=data.reset_index()  # Reset the index to a simple range
    output_data[date_col] = output_data[date_col].astype(str)
    return output_data

# RSI 计算函数
# RSI (14) Calculation
# def calculate_rsi(data, window=14):
#     # 计算每日变化（使用调整后的收盘价）
#     delta = data['Close'].diff()

#     # 分别计算正向和负向的变化
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

#     # 计算相对强弱RS
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
    
#     # 处理缺失值并返回RSI
#     return rsi.fillna(50)  # 用50填充缺失值，避免影响策略

# 使用修正后的 RSI 计算


# 计算 RSI 和动态隐含波动率
historical_data['RSI'] = calculate_rsi(historical_data)
historical_data['Implied Volatility'] = dynamic_iv(historical_data['Close'])

# 设置目标利润和最大亏损限制
target_profit_percent = 1.00  # 目标利润20%
max_loss_percent = 0.5  # 最大亏损30%

# 策略回测
cash = 0
portfolio_value = []
current_put = None  # 当前持有的看跌期权
buy_dates = []
put_info = []  # 记录期权买入时的信息

for i in range(1, len(historical_data)):
    today = historical_data.iloc[i]
    yesterday = historical_data.iloc[i - 1]

    # 股票价格变化
    stock_price = today['Close']
    stock_price_change = stock_price - yesterday['Close']

    # 当前的动态隐含波动率
    implied_volatility = today['Implied Volatility']

    # 检查当前是否有持有的期权
    if current_put:
        current_put["days_to_expiration"] -= 1
        if current_put["days_to_expiration"] <= 0:
            # 到期，期权价值归零
            cash += current_put["current_value"]
            current_put = None
        else:
            # 计算时间衰减后的期权价格
            T = current_put["days_to_expiration"] / 252
            price, delta, gamma, theta, vega = black_scholes_greeks(
                S=stock_price,
                K=current_put["strike_price"],
                T=T,
                r=risk_free_rate,
                sigma=implied_volatility,
                option_type="put",
            )
            current_put["current_value"] = price * 100

            # 检查是否达到目标利润或最大亏损
            if price >= current_put["initial_value"] * (1 + target_profit_percent):
                # 达到目标利润，卖出期权
                cash += current_put["current_value"]
                current_put = None
            elif price <= current_put["initial_value"] * (1 - max_loss_percent):
                # 达到最大亏损，卖出期权
                cash += current_put["current_value"]
                current_put = None
        # Predict RSI for today
    predicted_rsi = model.predict(historical_data)
    
    # Cap the RSI between 60 and 90
    predicted_rsi_clamped = np.clip(predicted_rsi, 60, 90)
    
    # Normalize the clamped values to the range [60, 90]
    predicted_rsi_normalized = 60 + ((predicted_rsi_clamped - 60) / (90 - 60)) * (90 - 60)
    
    # Dynamic threshold based on normalized RSI
    dynamic_threshold = predicted_rsi_normalized
    # 策略逻辑：RSI > 80 且没有持有期权时，买入看跌期权
    if today['RSI'] > dynamic_threshold and not current_put:
        strike_price = round(stock_price+50) #round(stock_price * 0.95 / 5) * 5
        T = days_to_expiration / 252
        price, delta, gamma, theta, vega = black_scholes_greeks(
            S=stock_price,
            K=strike_price,
            T=T,
            r=risk_free_rate,
            sigma=implied_volatility,
            option_type="put",
        )
        cash -= price * 100  # 买入期权
        current_put = {
            "strike_price": strike_price,
            "current_value": price * 100,
            "initial_value": price * 100,
            "days_to_expiration": days_to_expiration,
        }
        buy_dates.append(today.name)
        put_info.append({
            "Date": today.name,
            "Strike Price": strike_price,
            "Premium": price,
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega
        })

    # 计算总资产价值
    stock_value = shares_held * stock_price
    option_value = current_put["current_value"] if current_put else 0
    total_value = stock_value + option_value + cash
    portfolio_value.append(total_value)

# 输出每次买入期权的详细信息
put_info_df = pd.DataFrame(put_info)
print(put_info_df)



# 可视化策略表现
historical_data = historical_data.iloc[1:]  # 去掉首行
historical_data['Portfolio Value'] = portfolio_value

plt.figure(figsize=(12, 6))

# 绘制组合总价值
plt.plot(historical_data.index, historical_data['Portfolio Value'], label="Portfolio Value", color="blue")
# 绘制仅持股总价值
plt.plot(historical_data.index, historical_data['Close'] * shares_held, label="Stock Only Value", color="orange")

# 绘制买入期权位置
buy_dates_values = historical_data.loc[buy_dates]
plt.scatter(
    buy_dates_values.index, 
    buy_dates_values['Portfolio Value'], 
    color="red", 
    label="Put Option Bought", 
    zorder=5
)

# 图表标题和标签
plt.title("Protective Put Strategy with Greeks Info")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid()
plt.show()
remove_timezone(put_info_df).to_excel('put_price.xlsx',index=False)
remove_timezone(historical_data).to_excel('historical_data.xlsx',index=False)
