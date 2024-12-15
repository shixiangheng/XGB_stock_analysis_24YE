# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:54:20 2024

@author: shixiangheng
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime

# Step 1: Fetch Historical Data
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Step 2: Feature Engineering
def add_features(df):
    df['SMA_20'] = df['Close'].rolling(window=10).mean()  # 20-day Simple Moving Average
    df['SMA_50'] = df['Close'].rolling(window=200).mean()  # 50-day Simple Moving Average
    df['RSI'] = calculate_rsi(df['Close'], window=14)  # Relative Strength Index
    df['Daily_Return'] = df['Close'].pct_change()  # Daily returns
    df['Volatility'] = df['Close'].rolling(window=20).std()  # 20-day rolling volatility
    df = df.dropna(subset=['Close'])  # Drop rows with missing values
    return df

# RSI Calculation
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Step 3: Create Labels for Buy/Sell/Hold
def create_labels(df, threshold=0.05):
    df['Future_Return'] = df['Close'].pct_change(5).shift(-5)  # Future 5-day return
    df['Signal'] = 0  # Default to Hold
    df.loc[df['Future_Return'] > threshold, 'Signal'] = 1  # Buy Signal
    df.loc[df['Future_Return'] < -threshold, 'Signal'] = -1  # Sell Signal
    df = df.dropna()  # Drop rows with missing values
    return df

# Step 4: Train the XGBoost Model

def train_xgboost(X_train, y_train, X_test, y_test):
    # Map labels for y_train and y_test
    y_train_mapped = np.where(y_train == -1, 0, y_train + 1)
    y_test_mapped = np.where(y_test == -1, 0, y_test + 1)
    
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    
    # Train the model
    model.fit(
        X_train, y_train_mapped,
        eval_set=[(X_test, y_test_mapped)],
        verbose=True
    )
    return model


# Step 5: Backtest and Evaluate
def backtest(data, predictions, X_test_index):
    # test_data = X_test.copy()  # Copy the features from X_test
    # test_data['Signal'] = y_test
    # data=test_data
    data = data.loc[X_test.index]
    # Track buy and sell points
    data['Predicted_Signal'] = predictions
    data['Buy_Signal'] = np.where(data['Predicted_Signal'] == 1, data['Close'], np.nan)  # Buy signals
    data['Sell_Signal'] = np.where(data['Predicted_Signal'] == -1, data['Close'], np.nan)  # Sell signals

    # Calculate strategy returns
    data['Strategy_Return'] = data['Predicted_Signal'].shift(1) * data['Daily_Return']
    cumulative_strategy_return = (1 + data['Strategy_Return']).cumprod()
    cumulative_market_return = (1 + data['Daily_Return']).cumprod()

    # Plot the backtest results
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_strategy_return, label='Strategy Return', color='blue', alpha=0.7)
    plt.plot(cumulative_market_return, label='Market Return', color='green', alpha=0.7)

    # Plot Buy and Sell signals
    # Plot Buy Signals (assuming 'Buy_Signal' column exists in data)
    plt.scatter(data.index, data['Buy_Signal'], marker='^', color='g', label='Buy Signal', alpha=1)

    # Plot Sell Signals (assuming 'Sell_Signal' column exists in data)
    plt.scatter(data.index, data['Sell_Signal'], marker='v', color='r', label='Sell Signal', alpha=1)

    plt.legend()
    plt.title('Backtest Results with Buy/Sell Points')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.show()

def backtest(data, predictions, X_test_index):
    # Track buy and sell points
    data = data.loc[X_test_index]
    data['Predicted_Signal'] = predictions
    data['Buy_Signal'] = np.where(data['Predicted_Signal'] == 1, data['Close'], np.nan)  # Buy signals
    data['Sell_Signal'] = np.where(data['Predicted_Signal'] == -1, data['Close'], np.nan)  # Sell signals
    data = data.sort_index()
    print('max data date: ',data[-1:])
    # Calculate strategy returns
    data['Strategy_Return'] = data['Predicted_Signal'].shift(1) * data['Daily_Return']
    cumulative_strategy_return = (1 + data['Strategy_Return']).cumprod()
    cumulative_market_return = (1 + data['Daily_Return']).cumprod()

    # Create a new DataFrame for the plotting data that corresponds to the X_test_index
    filtered_data = data.loc[X_test_index]

    # Plot the backtest results
    plt.figure(figsize=(10, 6))

    # Plot full data points (close price and cumulative returns)
    plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.7)
    # plt.plot(cumulative_strategy_return, label='Strategy Return', color='orange', alpha=0.7)
    # plt.plot(cumulative_market_return, label='Market Return', color='green', alpha=0.7)

    # Plot Buy Signals
    plt.scatter(filtered_data.index, filtered_data['Buy_Signal'], marker='^', color='g', label='Buy Signal', alpha=1)

    # Plot Sell Signals
    plt.scatter(filtered_data.index, filtered_data['Sell_Signal'], marker='v', color='r', label='Sell Signal', alpha=1)

    plt.legend()
    plt.title('Backtest Results with Buy/Sell Points')
    plt.xlabel('Date')
    plt.ylabel('Price / Cumulative Return')
    plt.grid(True)
    plt.show()
    
def print_recent_buy_sell_dates(data,predictions,X_test_index):
    data=data.sort_index()
    data = data.loc[X_test_index]
    # Filter out non-null Buy and Sell signals
    data['Predicted_Signal'] = predictions
    data['Buy_Signal'] = np.where(data['Predicted_Signal'] == 1, data['Close'], np.nan)  # Buy signals
    data['Sell_Signal'] = np.where(data['Predicted_Signal'] == -1, data['Close'], np.nan)  # Sell signals
    buy_dates = data.loc[data['Buy_Signal'].notna(), 'Buy_Signal'].index
    sell_dates = data.loc[data['Sell_Signal'].notna(), 'Sell_Signal'].index
    buy_dates=sorted(list(buy_dates))
    sell_dates=sorted(list(sell_dates))
    # Get the most recent 5 Buy and Sell dates
    recent_buy_dates = buy_dates[-5:]  # Last 5 Buy Signal dates
    recent_sell_dates = sell_dates[-5:]  # Last 5 Sell Signal dates

    print("Recent 5 Buy Dates and Prices:")
    for date in recent_buy_dates:
        buy_price = data.loc[date, 'Close']
        print(f"- {date.strftime('%Y-%m-%d')}: {buy_price:.2f}")

    print("\nRecent 5 Sell Dates and Prices:")
    for date in recent_sell_dates:
        sell_price = data.loc[date, 'Close']
        print(f"- {date.strftime('%Y-%m-%d')}: {sell_price:.2f}")
        
        

# Forecast for a Specific Date
def forecast_for_date(features,symbol, target_date, target_start_date, target_end_date, models):
    # Fetch historical data up to the target date
    days_to_show=20
    data = fetch_data(symbol, start_date=target_start_date, end_date=target_end_date)

    # Check if the target date exists in the data
    if target_date not in data.index:
        print(f"Target date {target_date} not found in the data.")
        # Prompt user to input the close price for the missing date
        try:
            close_price = float(input(f"Enter the close price for {target_date}: "))
        except ValueError:
            print("Invalid input. Close price must be a number.")
            return

        # Add the missing target date and close price to the data
        new_row = pd.DataFrame({
            'Close': [close_price],
            'Open': [np.nan],  # Placeholder values, as they won't affect feature calculations
            'High': [np.nan],
            'Low': [np.nan],
            'Volume': [np.nan]
        }, index=[pd.Timestamp(target_date)])
        data = pd.concat([data, new_row]).sort_index()
    
    # Add features
    data = add_features(data)
    
    
    X=data[features]
    # Get final predictions using majority voting
    final_predictions = majority_vote(models, X)
    
    # Map back predictions to labels for evaluation
    final_predictions_mapped = final_predictions

    # Ensure the target date exists in the updated data
    if pd.Timestamp(target_date) not in data.index:
        print(data.index)
        print(f"Unable to process target date {target_date} after adding features.")
        return

    # Prepare the features for prediction
    
    recent_X = data[features].iloc[-days_to_show:].values  # The last 5 rows
    target_X = data[features].loc[[target_date]].values  # Only the target date row for prediction

    # Make prediction for the target date
    target_prediction = majority_vote(models,target_X)
    target_prediction_mapped = target_prediction #np.where(target_prediction == 0, -1, target_prediction - 1)

    print(f"Prediction for {target_date}: {'Buy' if target_prediction_mapped[0] == 1 else 'Sell' if target_prediction_mapped[0] == -1 else 'Hold'}")
    print(f"Price on {target_date}: {data.loc[target_date, 'Close']:.2f}")

    # Make predictions for the recent 5 days
    recent_predictions = majority_vote(models,recent_X)
    recent_predictions_mapped = recent_predictions #np.where(recent_predictions == 0, -1, recent_predictions - 1)
    
    print(f"\nRecent {days_to_show} days' predictions:")
    last_non_hold_date = None
    for i, (date, pred) in enumerate(zip(data.index[-days_to_show:], recent_predictions_mapped)):
        signal = 'Buy' if pred == 1 else 'Sell' if pred == -1 else 'Hold'
        print(f"- {date.strftime('%Y-%m-%d')}: {signal} | Price: {data.loc[date, 'Close']:.2f}")
        if signal != 'Hold':
            last_non_hold_date = date
            last_non_hold_signal = signal

    # Display the last date with a non-Hold signal
    if last_non_hold_date:
        #last_signal = 'Buy' if recent_predictions_mapped[-1] == 1 else 'Sell'
        print(f"\nLast non-Hold signal: {last_non_hold_signal} on {last_non_hold_date.strftime('%Y-%m-%d')}")
    else:
        print(f"\nNo non-Hold signals in the recent {days_to_show} days.")
    return data,final_predictions_mapped