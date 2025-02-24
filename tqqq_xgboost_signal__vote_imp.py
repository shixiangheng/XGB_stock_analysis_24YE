# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:24:33 2024

@author: shixiangheng
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from tqqq_xgboost_signal import fetch_data,add_features,calculate_rsi,backtest,majority_vote
#from utils import forecast_for_date

# Forecast for a Specific Date
def forecast_for_date(data, features, symbol, target_date, target_start_date, target_end_date, models, imp):
    if imp:
        # Fetch historical data up to the target date
        days_to_show=20
        
        data = data.dropna()
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
                'Open': [close_price],  # Placeholder values, as they won't affect feature calculations
                'High': [close_price],
                'Low': [close_price],
                'Adj Close': [close_price],
                'Volume': [10000]
            }, index=[pd.Timestamp(target_date)])
            data = pd.concat([data, new_row]).sort_index()
    else:
        data = data.dropna()
    
    # Add features
    data = add_features(data)
    data = data.dropna(subset=['SMA_2'])
    data.to_excel('X_implementation.xlsx')
    #data = data.dropna()
    
    
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

# Main Execution
if __name__ == "__main__":
    # Get today's date
    today_date = datetime.today().strftime('%Y-%m-%d')
    target_start_date='2023-12-01'
    target_end_date=today_date
    symbol = "TQQQ"
    target_date=today_date #"2024-11-25"
    features = ['SMA_1', 'SMA_2', 'RSI', 'Volatility', 'Daily_Return']
    folder=r'C:\Users\shixiangheng\Desktop\Henz\stock\XGB_stock_analysis_24YE\model'
    # Load the saved model
    #model = xgb.XGBClassifier()  # or use any model class that you used during training
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    #model.load_model('tqqq_xgboost_model_20241202.json')  # Load model from the saved file
    #model.load_model('tqqq_xgboost_model_20241202_model'+str(i)+'.json') 
    models=[]
    for i in range(9):
        model.load_model(folder+f'\{symbol}_xgboost_model_20241230_model'+str(i)+'.json') 
        models.append(model)  # 20241230  # 20250123
    
    # Use the model for predictions
    # Forecast for the given date
    #data=forecast_for_date(symbol, target_date,target_start_date, target_end_date, model)
    imp=1
    data = fetch_data(symbol, start_date=target_start_date, end_date=target_end_date)
    #y_pred_final = np.where(final_predictions == 0, -1, predictions - 1)
    X_,final_predictions_mapped=forecast_for_date(data, features, symbol, target_date,target_start_date, target_end_date, models,imp)

    backtest(X_, final_predictions_mapped, X_.index)
    print(f"Not accurate around the start date:{target_start_date}")
    
    