# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:44:49 2024

@author: shixiangheng

This is XGboost, trying for different feature combination, 
try to improve accuracy.
"""

import itertools
from xgboost_signal import *
import numpy as np

# Define parameter combinations
sma_windows = [(40,100)] #[(20, 50), (40, 100), (10, 200)]  # SMA combinations
rsi_windows = [14, 10, 20]  # RSI lookback periods
daily_return_windows = [1, 2, 5]  # Daily return lookback windows
volatility_windows = [20, 10, 30]  # Volatility rolling windows

# Accuracy results storage
results = []

# Modified Feature Engineering Function
def add_features_with_custom_parameters(df, sma1, sma2, rsi_window, daily_return_window, volatility_window):
    df[f'SMA_{sma1}'] = df['Close'].rolling(window=sma1).mean()
    df[f'SMA_{sma2}'] = df['Close'].rolling(window=sma2).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=rsi_window)
    df['Daily_Return'] = df['Close'].pct_change(periods=daily_return_window)
    df['Volatility'] = df['Close'].rolling(window=volatility_window).std()
    df = df.dropna()
    return df

# Modified Feature Engineering Function
def add_features_with_custom_sma(df, sma1, sma2):
    df[f'SMA_{sma1}'] = df['Close'].rolling(window=sma1).mean()
    df[f'SMA_{sma2}'] = df['Close'].rolling(window=sma2).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df = df.dropna()
    return df


# Main Execution
if __name__ == "__main__":
# Fetch Data
    today_date = datetime.today().strftime('%Y-%m-%d')
    
    backtest_type= '' #'insample' #''#
    symbol = "TQQQ"
    start_date = "201-12-01"
    end_date = "2023-12-01"
    #end_date = today_date
    data = fetch_data(symbol, start_date, end_date)
    threshold=0.05
    
    # Feature combinations testing loop
    for sma_windows, rsi_window, daily_return_window, volatility_window in itertools.product(
        sma_windows, rsi_windows, daily_return_windows, volatility_windows
    ):
        sma1=sma_windows[0]
        sma2=sma_windows[1]
        print(f"Testing SMA_{sma1}, SMA_{sma2}, RSI_{rsi_window}, Daily_Return_{daily_return_window}, Volatility_{volatility_window}...")
        
        # Prepare data with the custom parameters
        data_with_features = add_features_with_custom_parameters(
            data.copy(), sma1, sma2, rsi_window, daily_return_window, volatility_window
        )
        data_with_features = create_labels(data_with_features, threshold)
        
        # Prepare data for model training
        features = [
            f'SMA_{sma1}', f'SMA_{sma2}', 'RSI', 
            'Volatility', 'Daily_Return'
        ]
        X = data_with_features[features]
        y = data_with_features['Signal']
        sell_accuracy_l=[]
        for i in range(19):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
            
            # Train model
            model = train_xgboost(X_train, y_train, X_test, y_test)
            
            # Predict and evaluate
            predictions = model.predict(X_test)
            y_pred = np.where(predictions == 0, -1, predictions - 1)
            
            # Classification report
            report = classification_report(y_test, y_pred, target_names=['Sell', 'Hold', 'Buy'], output_dict=True)
            
            # Record results
            sell_accuracy = report['Sell']['precision']  # Precision for 'Sell' signals
            sell_accuracy_l.append(sell_accuracy)
        sell_accuracy=np.mean(sell_accuracy_l)
        results.append({
            'SMA_1': sma1,
            'SMA_2': sma2,
            'RSI_Window': rsi_window,
            'Daily_Return_Window': daily_return_window,
            'Volatility_Window': volatility_window,
            'Sell_Precision': sell_accuracy
        }) 
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to Excel
    results_df.to_excel(f"sell_signal_accuracy_{symbol}.xlsx", index=False)
    print(f"Results saved to 'sell_signal_accuracy_{symbol}.xlsx'")