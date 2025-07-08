# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 00:42:59 2025

@author: shixiangheng
"""

import joblib
import pandas as pd
import yfinance as yf
from RSI_threshold_train import calculate_rsi_ema
from tqqq_utils import dynamic_iv, black_scholes_greeks, add_return_percentage
from RSI_threshold_train import calculate_rsi,calculate_rsi_ema  # Import the RSI calculation function
from RSI_threshold_backtest import predict_today_vix_price
from tqqq_utils import *

SMA_1 = 'SMA_1'
SMA_2 = 'SMA_2'
iv_col = 'implied_volatility'
output_file = 'data_prepration_0130.xlsx'
# Load the pre-trained probability distribution model
pd_model_path = "tqqq_pd.pkl"
pd_model = joblib.load(pd_model_path)
risk_free_rate = 0.03
put_price_col = 'put_price'
# Load the main model for predicting RSI threshold
model_path = "rsi_threshold_model.pkl"
# Add return percentage for score calculation
previous_period_return_col = 'Previous_22d_Return'
score_col = 'score'
pd_col = 'pd'
down_percentage_col = 'Down_Percentage'
model = joblib.load(model_path)
# Load historical data for at least one year
historical_data = yf.download('TQQQ', period='1y', interval='1d')
historical_data = historical_data.reset_index().droplevel(1, axis=1)
historical_data.index = historical_data['Date']
# Load VIX historical data
vix_data = yf.download('^VIX', period='1y', interval='1d')
vix_data = vix_data.reset_index().droplevel(1, axis=1)
vix_data.index = vix_data['Date']

historical_data['VIX'] = vix_data['Close']

historical_data["RSI"] = calculate_rsi_ema(historical_data)

# Prepare data for score prediction
features = [SMA_1, SMA_2, 'RSI', 'Volatility']
historical_data[SMA_1] = historical_data['Close'].rolling(window=40).mean()
historical_data[SMA_2] = historical_data['Close'].rolling(window=100).mean()
historical_data['Volatility'] = historical_data['Close'].pct_change().rolling(window=10).std()
historical_data[iv_col] = dynamic_iv(historical_data['Close'])

# Predict RSI threshold
predicted_rsi_thresholds = [70]  # Start with an initial RSI threshold

down_label_dates = []
down_signal_col = 'Down_Signal'
historical_data[down_signal_col] = False  # Default to False

down_percentage_threshold = 0.9
next_day_down = -0.02
RSI_pred_threshold_col = 'Predicted_RSI_Threshold'


for i in range(1, len(historical_data)):
    current_price = historical_data['Close'].iloc[i]
    if i + 22 < len(historical_data):  # Check for the next 22 days
        future_prices = historical_data['Close'].iloc[i + 1:i + 22]
        next_day_daily_return = historical_data['Close'].pct_change().iloc[i + 1]
        if current_price * down_percentage_threshold > future_prices.mean() and next_day_daily_return < next_day_down:
            down_label_dates.append(historical_data.index[i])
            historical_data.at[historical_data.index[i], down_signal_col] = True
    vix_input = historical_data['VIX'].iloc[i]  # Use the VIX value for the day
    predicted_rsi = predict_today_vix_price(SMA_1, SMA_2, historical_data, model, vix_input, current_price)  # Predict RSI for today
    predicted_rsi_thresholds.append(predicted_rsi)

historical_data[RSI_pred_threshold_col] = predicted_rsi_thresholds
predicted_rsi_threshold = predicted_rsi_thresholds[-1]

historical_data = calculate_down_percentage(historical_data)
historical_data[down_percentage_col] = historical_data[down_percentage_col].ffill().bfill()

# Predict scores
historical_data['pd'] = pd_model.predict_proba(historical_data[features])[:, 1]


option_prices = []
for date, row in historical_data.iterrows():
    stock_price = row["Close"]
    strike_price = round(stock_price)  # Strike price is the rounded stock price
    implied_volatility = row[iv_col]  # Historical IV

    T = 22 / 252  # Set time to expiration to 22 days (22 trading days)

    if pd.notna(implied_volatility):  # Ensure IV data is available
        price, _, _, _, _ = black_scholes_greeks(
            S=stock_price,
            K=strike_price,
            T=T,
            r=risk_free_rate,
            sigma=implied_volatility,
            option_type="put",
        )
        option_prices.append(price)
    else:
        option_prices.append(None)  # Handle missing IV values



# Add calculated option prices to DataFrame
historical_data[put_price_col] = option_prices




historical_data = add_return_percentage(historical_data, 'Close', previous_period_return_col, days=22)
historical_data[score_col] = historical_data[pd_col] * historical_data[down_percentage_col] * historical_data['Close'] / historical_data[put_price_col]
historical_data.to_excel(output_file)
print('Output File:',output_file)

