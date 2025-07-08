import joblib
import pandas as pd
from RSI_threshold_train import calculate_rsi_ema
from tqqq_utils import *

SMA_1_col = 'SMA_1'
SMA_2_col = 'SMA_2'
# Load preprocessed data from Excel
data_path = "data_prepration_0130.xlsx"
historical_data = pd.read_excel(data_path)

# Load the pre-trained probability distribution model
pd_model_path = "tqqq_pd.pkl"
pd_model = joblib.load(pd_model_path)

# Load the main model for predicting RSI threshold
model_path = "rsi_threshold_model.pkl"
model = joblib.load(model_path)

# User input for today's data
today_close = float(input("Enter today's Close price: "))
today_vix = float(input("Enter today's VIX: "))
put_price_today = float(input("Enter today's Put price (100 contracts price): "))

# Calculate additional inputs
today_iv = dynamic_iv(pd.Series([today_close]))[0]
SMA_1 = historical_data['Close'].rolling(window=40).mean().iloc[-1]
SMA_2 = historical_data['Close'].rolling(window=100).mean().iloc[-1]
today_rsi = historical_data["RSI"].iloc[-1]
today_volatility = today_iv

# Predict RSI threshold
#predicted_rsi = model.predict([[today_vix, today_close, SMA_1, SMA_2, today_rsi, today_volatility]])[0]
current_price = today_close
predicted_rsi = predict_today_vix_price(SMA_1_col, SMA_2_col, historical_data, model, today_vix, current_price)  # Predict RSI for today

# Prepare data for score prediction
features = ['SMA_1', 'SMA_2', 'RSI', 'Volatility']
today_features = pd.DataFrame({
    "SMA_1": [SMA_1],
    "SMA_2": [SMA_2],
    "RSI": [today_rsi],
    "Volatility": [today_volatility]
})

today_score_prob = pd_model.predict_proba(today_features)[:, 1][0]  # Probability from model

down_percentage_col = 'Down_Percentage'
put_price_col = 'Put_Price'

# Calculate today's score
today_score = today_score_prob * historical_data[down_percentage_col].iloc[-1] * today_close / put_price_today
print(f"today_score_prob: {today_score_prob}")
print(f"historical_data[down_percentage_col].iloc[-1]: {historical_data[down_percentage_col].iloc[-1]}")


# Calculate the score threshold as the rolling mean of past scores
score_threshold = historical_data['score'].rolling(window=22*6, min_periods=1).mean().iloc[-1]

# Print calculated values
print(f"Today's RSI: {today_rsi:.2f}")
print(f"Predicted RSI Threshold: {predicted_rsi:.2f}")
print(f"Score: {today_score:.4f}")
print(f"Score Threshold: {score_threshold:.4f}")

# Decision: Should we buy a put option?
if today_rsi >= predicted_rsi * 0.95 and today_score > score_threshold:
    strike_price = round(today_close)
    days_to_expiration = 22
    risk_free_rate = 0.03
    
    # price, _, _, _, _ = black_scholes_greeks(
    #     S=today_close, K=strike_price, T=days_to_expiration / 252,
    #     r=risk_free_rate, sigma=today_iv, option_type="put"
    # )
    print(f"BUY PUT OPTION: Strike Price = {strike_price}, Premium = {put_price_today}")
else:
    print("No put option needed today.")


df = historical_data.reset_index()[:30]

import matplotlib.pyplot as plt

# Load your CSV or DataFrame
# Example: df = pd.read_csv("your_file.csv", parse_dates=['Date'])

# Ensure 'Date' is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set up figure and primary y-axis
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot Close and Portfolio Value on left y-axis
ax1.plot(df['Date'], df['Close'], label='TQQQ Close', color='blue')
#ax1.plot(df['Date'], df['Portfolio Value'], label='Portfolio Value', color='green')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price / Portfolio Value')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create secondary y-axis for score
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['Predicted_RSI_Threshold'], label='predicted_rsi_thresholds', color='red', linestyle='--')
ax2.set_ylabel('predicted_rsi_thresholds')
ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
ax2.legend(loc='upper right')

# Title and layout
plt.title('TQQQ Close, Portfolio Value, and Score')
plt.tight_layout()
plt.show()