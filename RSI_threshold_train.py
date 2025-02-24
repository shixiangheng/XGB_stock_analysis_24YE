import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib  # To save and load the model
# Parameters
ticker_symbol = "TQQQ"
vix_symbol = "^VIX"  # VIX symbol
start_date = "2017-01-31"
end_date = "2025-01-31"
rsi_high_bond = 70  # RSI threshold
down_percentage_threshold = 0.8

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Fill missing values with 50 to avoid affecting strategy

def calculate_rsi_ema(data, window=14):
    delta = data['Close'].diff()

    # Separate positive and negative price changes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Initialize the first average gain and loss
    avg_gain = gain.rolling(window=window).mean().iloc[window - 1]
    avg_loss = loss.rolling(window=window).mean().iloc[window - 1]

    # Calculate subsequent averages using smoothing (like EMA)
    gain_ewm = gain.ewm(span=window, adjust=False).mean()
    loss_ewm = loss.ewm(span=window, adjust=False).mean()

    # Compute RSI
    rs = gain_ewm / loss_ewm
    rsi = 100 - (100 / (1 + rs))

    # Return RSI, filling missing values at the start
    rsi[:window] = 50  # Handle the initial window
    return rsi
SMA_1 = 'SMA_1'
SMA_2 = 'SMA_2'

if __name__ == "__main__":
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
    
    # Calculate daily returns
    historical_data['Daily Return'] = historical_data['Close'].pct_change()
    historical_data['RSI_Mod'] = historical_data['RSI']
    # Modify RSI according to price rule
    for i in range(len(historical_data)):
        current_price = historical_data['Close'].iloc[i]
        
        if i + 22 < len(historical_data):  # Check for the next 22 days
            future_prices = historical_data['Close'].iloc[i + 1:i + 22]
            if current_price * down_percentage_threshold < future_prices.mean():  # if not below 10% in next 22 BDs
                if historical_data['RSI_Mod'].iloc[i] <= 70:  
                    historical_data['RSI_Mod'].iloc[i] = 70# 
                    #historical_data['RSI_Mod'].iloc[i] + 20  # Set RSI to 70
            else:
                historical_data['RSI_Mod'].iloc[i] = historical_data['RSI_Mod'].iloc[i] - 10
    # Resample VIX data to match the stock data frequency (daily)
    #vix_data = vix_data[['Close']].resample('D').ffill()  # Forward-fill missing VIX data for daily frequency
    
    # Merge VIX data with stock data on date index
    historical_data = historical_data.merge(vix_data, left_index=True, right_index=True, how='left')
    historical_data.rename(columns={'Close_y': 'VIX'}, inplace=True)
    historical_data=historical_data.rename(columns={'Close_x':'Close'})
    # Drop rows with missing values
    
    # Prepare features and target
    X = historical_data[['Close', 'Daily Return', SMA_1, SMA_2, 'VIX','RSI']]
    y = historical_data['RSI_Mod']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=1000)
    print(y_test.index[0])
    # Train the model
    model.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(model, 'rsi_threshold_model.pkl')
    print("Model saved successfully!")
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Step 1: Clamp the predicted RSI values
    y_pred_clamped = np.clip(y_pred, 60, 90)
    
    # Step 2: Normalize the clamped values to the range [60, 90]
    y_pred_normalized = 60 + ((y_pred_clamped - y_pred_clamped.min()) / 
                              (y_pred_clamped.max() - y_pred_clamped.min())) * (90 - 60)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_normalized)
    print(f'Mean Squared Error: {mse}')
    
    # Plot actual vs predicted RSI
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Actual RSI Threshold", color='blue')
    plt.plot(y_test.index, y_pred_normalized, label="Predicted RSI threshold", color='red')
    plt.title("Actual vs Predicted RSI threshold")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.show()
