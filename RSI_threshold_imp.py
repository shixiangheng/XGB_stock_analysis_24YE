import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib  # To save and load the model
from RSI_threshold_train import calculate_rsi
# Parameters
ticker_symbol = "TQQQ"
vix_symbol = "^VIX"  # VIX symbol
start_date = "2021-01-31"
end_date = "2025-01-31"
rsi_high_bond = 70  # RSI threshold


# User input for prediction
def predict_today_vix_price(vix, current_price):
    # Load the saved model
    model = joblib.load('rsi_threshold_model.pkl')
    # Prepare features for prediction
    sma40 = historical_data['SMA40'].iloc[-1]  # Last SMA40 value
    sma100 = historical_data['SMA100'].iloc[-1]  # Last SMA100 value
    daily_return = (current_price - historical_data['Close'].iloc[-1]) / historical_data['Close'].iloc[-1]  # Daily return
    X_today = pd.DataFrame([[current_price, daily_return, sma40, sma100, vix, historical_data['RSI'].iloc[-1]]],
                           columns=['Close', 'Daily Return', 'SMA40', 'SMA100', 'VIX', 'RSI'])
    # Predict RSI for today
    predicted_rsi = model.predict(X_today)
    return predicted_rsi[0]
# Download stock price and VIX data
# Main Execution
if __name__ == "__main__":
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(start=start_date, end=end_date)
    
    vix = yf.Ticker(vix_symbol)
    vix_data = vix.history(start=start_date, end=end_date)
    
    # Calculate RSI
    historical_data['RSI'] = calculate_rsi(historical_data)
    
    # Calculate SMA40 and SMA100
    historical_data['SMA40'] = historical_data['Close'].rolling(window=40).mean()
    historical_data['SMA100'] = historical_data['Close'].rolling(window=100).mean()
    
    # Ask user for input
    vix_input = float(input("Enter today's VIX value: "))
    current_price_input = float(input("Enter today's current stock price: "))
    
    # Predict RSI for today
    predicted_rsi_today = predict_today_vix_price(vix_input, current_price_input)
    print(f"Predicted RSI for today: {predicted_rsi_today}")