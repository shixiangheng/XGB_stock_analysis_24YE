import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib  # To save and load the model
from RSI_threshold_train import calculate_rsi,calculate_rsi_ema  # Import the RSI calculation function

# Parameters
ticker_symbol = "TQQQ"
vix_symbol = "^VIX"  # VIX symbol
start_date = "2024-01-31"
end_date = "2025-01-31"
model_config = 'rsi_threshold_model.pkl'

SMA_1 = 'SMA_1'
SMA_2 = 'SMA_2'
# User input for prediction (used to predict RSI on a given day)
def predict_today_vix_price(SMA_1, SMA_2, historical_data, model, vix, current_price):
    # Load the saved model
  
    
    # Prepare features for prediction
    sma40 = historical_data[SMA_1].iloc[-1]  # Last SMA40 value
    sma100 = historical_data[SMA_2].iloc[-1]  # Last SMA100 value
    daily_return = (current_price - historical_data['Close'].iloc[-1]) / historical_data['Close'].iloc[-1]  # Daily return
    X_today = pd.DataFrame([[current_price, daily_return, sma40, sma100, vix, historical_data['RSI'].iloc[-1]]],
                           columns=['Close', 'Daily Return', SMA_1, SMA_2, 'VIX', 'RSI'])
    
    # Predict RSI for today
    predicted_rsi = model.predict(X_today)
    
    # Cap the RSI between 60 and 90
    predicted_rsi_clamped = np.clip(predicted_rsi, 60, 90)
    
    # Normalize the clamped values to the range [60, 90]
    predicted_rsi_normalized = 60 + ((predicted_rsi_clamped - 60) / (90 - 60)) * (90 - 60)
    
    return predicted_rsi_normalized[0]

if __name__ == "__main__":
    model = joblib.load(model_config)
    # Download stock price and VIX data
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(start=start_date, end=end_date)
    
    vix = yf.Ticker(vix_symbol)
    vix_data = vix.history(start=start_date, end=end_date)
    
    # Calculate RSI
    historical_data['RSI'] = calculate_rsi_ema(historical_data)
    
    # Calculate SMA40 and SMA100
    historical_data[SMA_1] = historical_data['Close'].rolling(window=40).mean()
    historical_data[SMA_2] = historical_data['Close'].rolling(window=100).mean()
    
    # Merge VIX data with stock data on date index
    historical_data = historical_data.merge(vix_data[['Close']], left_index=True, right_index=True, how='left')
    historical_data.rename(columns={'Close_y': 'VIX'}, inplace=True)
    historical_data = historical_data.rename(columns={'Close_x': 'Close'})
    
    # Drop rows with missing values
    historical_data = historical_data.dropna()
    
    # Initialize a list to store predicted RSI values
    predicted_rsi_values = []
    
    # Backtest: Loop through the data and make predictions for each day
    for i in range(1, len(historical_data)):
        current_price = historical_data['Close'].iloc[i]
        vix_input = historical_data['VIX'].iloc[i]  # Use the VIX value for the day
        predicted_rsi = predict_today_vix_price(historical_data, model, vix_input, current_price)  # Predict RSI for today
        predicted_rsi_values.append(predicted_rsi)
    
    # Add the predicted RSI values to the dataframe
    historical_data['Predicted RSI'] = np.nan
    historical_data['Predicted RSI'].iloc[1:] = predicted_rsi_values  # Align predictions with dates
    
    # Cap the predicted RSI between 60 and 90
    historical_data['Predicted RSI'] = np.clip(historical_data['Predicted RSI'], 60, 90)
    
    # Calculate RSI breach (scatter points)
    historical_data['RSI_breach'] = np.where(historical_data['RSI'] - historical_data['Predicted RSI'] > 0, historical_data['Close'], np.nan)
    
    # Plotting the stock price and RSI breaches
    plt.figure(figsize=(14, 7))
    
    # Plot stock price as a line
    plt.plot(historical_data.index, historical_data['Close'], label='Stock Price', color='blue')
    plt.plot(historical_data.index, historical_data['RSI'], label='RSI', color='red')
    plt.plot(historical_data.index, historical_data['Predicted RSI'], label='RSI_threshold', color='purple')
    
    # Add RSI breaches as scatter points
    plt.scatter(historical_data.index, historical_data['RSI_breach'], label='RSI Breach', color='orange', s=50, marker='o', zorder=5)
    
    # Labels and title
    plt.title('Stock Price and RSI Breaches')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
