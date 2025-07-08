import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import joblib
from RSI_threshold_train import calculate_rsi,calculate_rsi_ema  # Import the RSI calculation function
from RSI_threshold_backtest import predict_today_vix_price
from tqqq_utils import *
# Parameters
ticker_symbol = "TQQQ"
vix_symbol = "^VIX"
start_date = "2021-01-31"
end_date = "2025-04-30"
shares_held = 100  # Number of shares held
risk_free_rate = 0.03  # Annualized risk-free rate
days_to_expiration = 22  # Option expiration in days
model_config = 'rsi_threshold_model.pkl'  # Pre-trained model path
pd_model = "tqqq_pd.pkl"
pd_data_file = 'tqqq_pd.xlsx'


SMA_1 = 'SMA_1'
SMA_2 = 'SMA_2'
initial_cash = 10000
date_col = 'Date'
pd_col = 'pd'
score_col = 'score'
avg_daily_return_col ='avg_daily_return'
score_avg_col = 'score_avg'
avg_days_for_score_threshold = 22 * 6
RSI_pred_threshold_col = 'predicted_rsi_thresholds'
diff_RSI_col = 'diff_rsi' 
RSI_col = 'RSI'
down_signal_col = 'down_signal'
daily_return_col = 'daily_return'
down_percentage_col = 'Down_Percentage'
price_col = 'Close'
previous_period_return_col = 'previous_period_return_col'
put_price_col = 'Put_Price'
target_profit_percent = 3  # 目标利润20%
max_loss_percent = 0.3  # 最大亏损30%
MULTIPLIER = 100
lra_val = 0#-0.00793635283388 #0.000897087
down_percentage_threshold = 0.8
next_day_down = -5

features = ['SMA_1', 'SMA_2', 'RSI', 'Volatility']
# 获取历史数据


# Fetch TQQQ historical data
ticker_symbol = "TQQQ"
vix_symbol = "^VIX"
# start_date = "2020-01-01"
# end_date = "2023-12-31"
date_col = "Date"

ticker = yf.Ticker(ticker_symbol)
historical_data = ticker.history(start=start_date, end=end_date).reset_index()
historical_data = normalize_date(historical_data, date_col)

# Fetch VIX historical data
vix = yf.Ticker(vix_symbol)
vix_data = vix.history(start=start_date, end=end_date).reset_index()
vix_data = normalize_date(vix_data, date_col)

# Merge TQQQ and VIX data on the date column
merged_data = historical_data.merge(vix_data[[date_col, 'Close']], on=date_col, how='inner')

# Rename columns for clarity
merged_data.rename(columns={'Close_x': 'Close', 'Close_y': 'VIX'}, inplace=True)

# Display the first few rows of the merged dataset
print(merged_data.head())
historical_data = merged_data.copy()
# RSI & Indicators
historical_data['RSI'] = calculate_rsi_ema(historical_data)
historical_data['Implied Volatility'] = dynamic_iv(historical_data['Close'])
historical_data[SMA_1] = historical_data['Close'].rolling(window=40).mean()
historical_data[SMA_2] = historical_data['Close'].rolling(window=100).mean()

# Portfolio variables
cash = 0
portfolio_value = []
current_put = None
buy_dates = []
put_info = []
down_label_dates = []

def manage_portfolio_with_options(historical_data, pd_model, model_config, initial_cash, shares_held, risk_free_rate, days_to_expiration, target_profit_percent, max_loss_percent):
    # Load RSI threshold model
    print(historical_data.columns)
    # Initialize portfolio values
    cash = initial_cash
    portfolio_value = []
    current_put = None
    buy_dates = []
    put_info = []
    put_price_l = [0]
    # Initial stock purchase with initial cash
    stock_price = historical_data.iloc[0]['Close']
    stock_value = shares_held * stock_price
    cash -= stock_value  # Deduct the amount used for stock purchase

    # Loop through historical data starting from the second day
    for i in range(1, len(historical_data)):
        today = historical_data.iloc[i]
        yesterday = historical_data.iloc[i - 1]
        stock_price = today['Close']

        # Current dynamic implied volatility
        implied_volatility = today['Implied Volatility']

        # Handle current options if any
        if current_put:
            # put_price_l.append(current_put["current_value"])
            
            current_put["days_to_expiration"] -= 1
            if current_put["days_to_expiration"] <= 0:
                # If the put has expired, exercise it at the strike price
                exercise_price = current_put["strike_price"]
                cash += max(0,  exercise_price - stock_price) * 100  # Exercise value if in-the-money
                current_put = None  # Clear the put option as it's exercised
            else:
                # Update the current value of the put using Black-Scholes
                T = current_put["days_to_expiration"] / 252
                price, _, _, _, _ = black_scholes_greeks(
                    S=stock_price,
                    K=current_put["strike_price"],
                    T=T,
                    r=risk_free_rate,
                    sigma=implied_volatility,
                    option_type="put",
                )
                current_put["current_value"] = price * 100
                print(price*100)
                # Check if target profit or max loss is reached
                if price * MULTIPLIER >= current_put["initial_value"] * (1 + target_profit_percent):
                    # Target profit reached, sell the put
                    cash += current_put["current_value"]
                    current_put = None
                elif price* MULTIPLIER <= current_put["initial_value"] * (1 - max_loss_percent):
                    # Max loss reached, sell the put
                    cash += current_put["current_value"]
                    current_put = None
        else:
            put_price_l.append(0)
            
        # Get dynamic RSI threshold for today
        dynamic_threshold = today['predicted_rsi_thresholds']
        # dynamic_threshold = predicted_rsi_thresholds[i]
        score_threshold = today[score_avg_col]
        
        # Buy logic: if RSI exceeds threshold and no current option held
        #if today[RSI_col] > dynamic_threshold and not current_put:
        if today[RSI_col] >= dynamic_threshold*0.95 and today['score'] > score_threshold and not current_put:
            strike_price = round(stock_price)
            T = days_to_expiration / 252
            price, delta, gamma, theta, vega = black_scholes_greeks(
                S=stock_price,
                K=strike_price,
                T=T,
                r=risk_free_rate,
                sigma=implied_volatility,
                option_type="put",
            )
            cash -= price * 100  # Deduct the amount used for buying the put option
            current_put = {
                "strike_price": strike_price,
                "current_value": price * 100,
                "initial_value": price * 100,
                "days_to_expiration": days_to_expiration,
            }
            print('new put_initial_price:',current_put["initial_value"])
            
            buy_dates.append(today.name)
            put_info.append({
                "Date": today.name,
                "Strike Price": strike_price,
                "Premium": price,
                "Delta": delta,
                "Gamma": gamma,
                "Theta": theta,
                "Vega": vega,
            })

        # Update portfolio value: Stock value + Option value + Cash
        stock_value = shares_held * stock_price
        option_value = current_put["current_value"] if current_put else 0
        total_value = stock_value + option_value + cash
        portfolio_value.append(total_value)
        # historical_data['put_price'] = put_price_l
    return portfolio_value, buy_dates, put_info, cash, historical_data

#historical_data = historical_data.reset_index()
historical_data[date_col] = pd.to_datetime(historical_data[date_col]).dt.tz_localize(None)  # Remove timezone
historical_data[date_col] = pd.to_datetime(historical_data[date_col]).dt.date  # Stripping time part

pd_data =  pd.read_excel(pd_data_file)
pd_data[date_col] = pd.to_datetime(pd_data[date_col]).dt.date  # Ensure date format

historical_data = historical_data.merge(pd_data[[date_col,pd_col]], on = date_col, how = 'left')
historical_data = historical_data.sort_values(date_col)
historical_data.set_index(date_col,inplace=True)

# get 22 day rolling avg daily return
historical_data = calculate_avg_daily_return(historical_data)
pd_model =  joblib.load(pd_model)
model = joblib.load(model_config)
predicted_rsi_thresholds = [70]  # Start with an initial RSI threshold

# Initialize a new column to store the condition result
historical_data[down_signal_col] = False  # Default to False
# define label for y cases
for i in range(1, len(historical_data)):
    current_price = historical_data['Close'].iloc[i]
    if i + 22 < len(historical_data):  # Check for the next 22 days
        future_prices = historical_data['Close'].iloc[i + 1:i + 22]
        next_day_daily_return = historical_data[daily_return_col].iloc[i + 1]
        # Mark the condition where current_price * 0.9 > future_prices.mean()
        if current_price * down_percentage_threshold > future_prices.mean() and next_day_daily_return < next_day_down:
            down_label_dates.append(historical_data.index[i])
            #print(current_price)
            historical_data.at[historical_data.index[i], down_signal_col] = True
    vix_input = historical_data['VIX'].iloc[i]  # Use the VIX value for the day
    predicted_rsi = predict_today_vix_price(SMA_1,SMA_2, historical_data, model, vix_input, current_price)  # Predict RSI for today
    predicted_rsi_thresholds.append(predicted_rsi)
# for i in range(1, len(historical_data)):
#     current_price = historical_data['Close'].iloc[i]
    

historical_data[RSI_pred_threshold_col] = predicted_rsi_thresholds

            


historical_data[diff_RSI_col] = historical_data[RSI_col] - historical_data[RSI_pred_threshold_col]
historical_data[diff_RSI_col] = historical_data[diff_RSI_col].clip(lower=0)

historical_data = calculate_down_percentage(historical_data)

historical_data[down_percentage_col] = historical_data[down_percentage_col].ffill().bfill()
# put price calc
historical_data = add_put_option_prices(
    historical_data,
    stock_price_col="Close",
    implied_volatility_col="Implied Volatility",
    risk_free_rate=risk_free_rate,
    days_to_expiration=days_to_expiration,
)
# lgd to be define as return from previous month. used to be  historical_data[diff_RSI_col]  
# 
'''
down percentage is negative,
 because if for recent 22 days, down percentage is gradually smaller and smaller, 
 '''
 
# historical_data[down_percentage_col] = 1 +  historical_data[down_percentage_col]
historical_data = add_return_percentage(historical_data, price_col, previous_period_return_col, days=22)
historical_data[score_col] = historical_data[pd_col] * historical_data[down_percentage_col] * historical_data['Close'] / historical_data[put_price_col]

# Filter scores based on the down_signal_col condition
filtered_scores = historical_data.loc[historical_data[down_signal_col], score_col]

# Compute the rolling mean for the filtered rows and handle empty values
historical_data[score_avg_col] = (
    filtered_scores.rolling(window=avg_days_for_score_threshold, min_periods=1)  # Allow partial windows
    .mean()
    .reindex(historical_data.index, fill_value=lra_val)  # Reindex and fill missing values with 0
)

historical_data[score_avg_col]=historical_data[score_avg_col].ffill()#fillna(lra_val) #

#historical_data[score_avg_col] = historical_data[score_col].rolling(avg_days_for_score_threshold).mean()
portfolio_value, buy_dates, put_info, cash, \
    historical_data = manage_portfolio_with_options(
                                                    historical_data, 
                                                    pd_model,
                                                    model_config, 
                                                    initial_cash, 
                                                    shares_held, 
                                                    risk_free_rate,
                                                                           days_to_expiration, target_profit_percent, max_loss_percent)

# Convert put_info to DataFrame
put_info_df = pd.DataFrame(put_info)
#print(put_info_df)

# Visualize results
historical_data = historical_data.iloc[1:]
historical_data['Portfolio Value'] = portfolio_value

plt.figure(figsize=(12, 6))
plt.plot(historical_data.index, historical_data['Portfolio Value'], label="Portfolio Value", color="blue")
plt.plot(historical_data.index, historical_data['Close'] * shares_held, label="Stock Only Value", color="orange")
# plt.plot(historical_data.index, historical_data['put_price'] * 100, label="Stock Only Value", color="orange")

buy_dates_values = historical_data.loc[buy_dates]
plt.scatter(
    buy_dates_values.index, 
    buy_dates_values['Portfolio Value'], 
    color="red", 
    label="Put Option Bought", 
    zorder=5
)

# down_dates_value = historical_data.loc[down_label_dates]
# plt.scatter(
#     down_dates_value.index, 
#     down_dates_value['Portfolio Value'], 
#     color="blue", 
#     label="Down Date", 
#     zorder=5
# )


# 图表标题和标签
plt.title("Protective Put Strategy with Greeks Info")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid()
plt.show()

put_info_df.to_excel('put_price.xlsx', index=False)
historical_data.reset_index().to_excel('historical_data.xlsx', index=False)

