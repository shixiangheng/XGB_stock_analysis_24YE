# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:40:38 2024

@author: shixiangheng

backtest
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Step 1: Fetch Historical Data
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data.reset_index().droplevel(1, axis=1)
    data.index = data['Date']
    return data

# Step 2: Feature Engineering
def add_features(df):
    sma1=40
    sma2=100
    rsi_window=20
    daily_return_window=5
    volatility_window=20
    df['SMA_1'] = df['Close'].rolling(window=sma1).mean()
    df['SMA_2'] = df['Close'].rolling(window=sma2).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=rsi_window)
    df['Daily_Return'] = df['Close'].pct_change(periods=daily_return_window)
    df['Volatility'] = df['Close'].rolling(window=volatility_window).std()
    #df = df.dropna()
    return df

# RSI Calculation
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Step 3: Create Labels for Buy/Sell/Hold
def create_labels(df, threshold=0.10):
    df['Future_Return'] = df['Close'].pct_change(5).shift(-5)  # Future 5-day return
    df['Signal'] = 0  # Default to Hold
    df.loc[df['Future_Return'] > threshold, 'Signal'] = 1  # Buy Signal
    df.loc[df['Future_Return'] < -threshold, 'Signal'] = -1  # Sell Signal
    df = df.dropna(subset=['Future_Return'])  # Drop rows with missing values
    #df = df.dropna()
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

def majority_vote(models, X):
    """
    Perform majority voting for a set of models.
    Args:
    - models: list of trained XGBoost models
    - X: Features for which predictions are made
    
    Returns:
    - final_predictions: Array of majority vote results for each instance in X
    """
    predictions = []
    for model in models:
        pred = model.predict(X)
        # Map predictions: 0 -> Sell (-1), 1 -> Hold (0), 2 -> Buy (1)
        mapped_pred = np.where(pred == 0, -1, pred - 1)
        predictions.append(mapped_pred)

    # Transpose to get predictions per sample
    predictions = np.array(predictions).T

    # Apply majority voting
    final_predictions = []
    for preds in predictions:
        counts = np.bincount(preds + 1, minlength=3)  # Shift by +1 for bincount
        final_decision = np.argmax(counts) - 1        # Map back (-1, 0, 1)
        final_predictions.append(final_decision)

    return np.array(final_predictions)

def cashflow_forecast(starting_cash, predictions, data, X_test_index):
    """
    Simulates a cashflow forecast based on trading signals and plots portfolio value over time.
    
    Args:
    - starting_cash (float): Initial capital available for investment.
    - predictions (array-like): Predicted signals (-1 for Sell, 1 for Buy, 0 for Hold).
    - data (DataFrame): Data containing stock prices.
    - X_test_index (Index): Index of the test set for alignment with predictions.
    
    Returns:
    - cashflow_df (DataFrame): DataFrame tracking cashflow over time.
    """
    # Align data to predictions
    data = data.loc[X_test_index]
    data['Predicted_Signal'] = predictions
    data = data.sort_index()

    # Initialize variables
    cash = starting_cash
    stocks_held = 0
    cashflow_records = []

    # Simulate cashflow based on predictions
    for i in range(len(data)):
        date = data.index[i]
        signal = data['Predicted_Signal'].iloc[i]
        close_price = data['Close'].iloc[i]

        if signal == 1:  # Buy
            if cash > 0:
                stocks_held = cash / close_price
                cash = 0
                cashflow_records.append({'Date': date, 'Action': 'Buy', 'Price': close_price, 'Cash': cash, 'Stocks': stocks_held})
        elif signal == -1:  # Sell
            if stocks_held > 0:
                cash = stocks_held * close_price
                stocks_held = 0
                cashflow_records.append({'Date': date, 'Action': 'Sell', 'Price': close_price, 'Cash': cash, 'Stocks': stocks_held})
        else:  # Hold
            cashflow_records.append({'Date': date, 'Action': 'Hold', 'Price': close_price, 'Cash': cash, 'Stocks': stocks_held})

    # Create DataFrame for cashflow tracking
    cashflow_df = pd.DataFrame(cashflow_records)
    cashflow_df.set_index('Date', inplace=True)

    # Add a column for total value (cash + stocks value)
    cashflow_df['Total_Value'] = cashflow_df['Cash'] + (cashflow_df['Stocks'] * data['Close'])

    # Plot the total portfolio value over time
    plt.figure(figsize=(10, 6))
    plt.plot(cashflow_df.index, cashflow_df['Total_Value'], label='Total Portfolio Value', color='blue', alpha=0.7)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return cashflow_df

# Main Execution
if __name__ == "__main__":
    # Fetch Data
    today_date = datetime.today().strftime('%Y-%m-%d')
    # Get yesterday's date
    yesterday_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    folder=r'C:\Users\shixiangheng\Desktop\Henz\stock\XGB_stock_analysis_24YE\model'
    backtest_type= ''#'insample' #''#
    symbol = "TQQQ"
    start_date = "2019-12-31"
    end_date =  "2024-12-31"
    #end_date = today_date
    data = fetch_data(symbol, start_date, end_date)
    threshold=0.1   # 25 precision
    # Feature Engineering and Labeling
    data = add_features(data)
    data = create_labels(data,threshold)
    
    # Prepare Data for Model
    features = ['SMA_1', 'SMA_2', 'RSI', 'Volatility', 'Daily_Return']
    target = 'Signal'
    X = data[features]
    y = data[target]
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    # # Train Model
    # model = train_xgboost(X_train, y_train, X_test, y_test)
    
    
    # save model
    #model.save_model('tqqq_xgboost_model_20241202.json') 
    
    # List to store the trained models
    models = []
    
    # Train 20 models
    for i in range(9):
        # Split the data differently for each model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        # Train the model
        model = train_xgboost(X_train, y_train, X_test, y_test)
        model.save_model(folder+'\\tqqq_xgboost_model_20250625_model'+str(i)+'.json') 
        models.append(model)
    
    # prepare oot X_test
    oot_start_date = end_date#"2023-07-01"
    #end_date = "2024-12-01"
    oot_end_date = today_date
    oot_data = fetch_data(symbol, oot_start_date, oot_end_date)

    # Feature Engineering and Labeling
    oot_data = add_features(oot_data)
    oot_data = create_labels(oot_data,threshold)
    oot_data = oot_data.dropna()
    # Prepare X_test for OOT:
    features = ['SMA_1', 'SMA_2', 'RSI', 'Volatility', 'Daily_Return']
    target = 'Signal'
    X = oot_data[features]
    y = oot_data[target]
    
    #_, X_test_oot, __, y_test_oot = train_test_split(X, y, test_size=0.8, random_state=20)
    X_test_oot=X
    y_test_oot=y
    # Get final predictions using majority voting
    #X_test_oot = X_test_oot.dropna()
    X_test_oot.to_excel('TQQQ_X_test_oot.xlsx',index=False)


    final_predictions = majority_vote(models, X_test_oot)
    
    # Map back predictions to labels for evaluation
    # final_predictions_mapped = final_predictions
    print(final_predictions)
    # Evaluate the majority-vote model
    try:
        print(classification_report(y_test_oot, final_predictions, target_names=['Sell', 'Hold', 'Buy']))
    except:
        pass
    # Backtest using the aggregated predictions
    backtest(oot_data, final_predictions, X_test_oot.index)
    # Initial cash to start with
    starting_cash = 10000
    
    # Generate cashflow forecast and plot
    cashflow = cashflow_forecast(starting_cash, final_predictions, oot_data, X_test_oot.index)

    # Print recent buy/sell dates
    print_recent_buy_sell_dates(oot_data, final_predictions, X_test_oot.index)

   