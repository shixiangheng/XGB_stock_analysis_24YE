# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 00:20:19 2024

@author: shixiangheng
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 23:31:42 2024

@author: shixiangheng
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import random
import xlsxwriter
from openpyxl import Workbook

def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def prepare_signals(data, short_window=40, long_window=100):
    data['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    data['mavg_diff'] = data['short_mavg'] - data['long_mavg']
    data['diff_change'] = data['mavg_diff'].diff()
    data['signal'] = np.where(data['diff_change'] > 0, 1.0, np.where(data['diff_change'] < 0, -1.0, 0.0))
    data['positions'] = data['signal'].diff()
    return data



def execute_trades(data, initial_capital=100.0, profit_to_buy=-3, profit_to_sell=40):
 
    position = 0
    cash = initial_capital
    portfolio_values = []
    last_trade_date = None
    buy_price = 0  # To track the price at which we last bought the stock

    for date, row in data.iterrows():
        notbuy = 0
        
        #if last_trade_date is not None and (date - last_trade_date).days < 1:
        #    notbuy = 1

        # Calculate the current profit or loss percentage
        if position > 0:
            current_price = row['Close']
            profit_loss_percent = ((current_price - buy_price) / buy_price) * 100
            
            # Check if the loss is more than 10% or the profit is more than 20%
            if profit_loss_percent <= profit_to_buy or profit_loss_percent >= profit_to_sell:
                cash = position * row['Close']
                position = 0.0
                buy_price = 0  # Reset buy_price since we sold
                last_trade_date = date
                notbuy = 1  # Prevent immediate rebuy

        if not notbuy:
            if row['positions'] == 2.0:  # From no position to buy
                if cash > 0:
                    position = cash / row['Close']
                    cash = 0.0
                    buy_price = row['Close']  # Update buy_price with the price at which we bought
                    last_trade_date = date
            elif row['positions'] == -2.0:  # From buy to sell
                if position > 0:
                    cash = position * row['Close']
                    position = 0.0
                    buy_price = 0  # Reset buy_price since we sold
                    last_trade_date = date
        
        portfolio_value = cash + position * row['Close']
        portfolio_values.append(portfolio_value)

    data['portfolio_value'] = portfolio_values
    return data

def plot_performance(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='TQQQ Price')
    plt.plot(data['portfolio_value'], label='Strategy Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('TQQQ Trading Strategy Performance')
    plt.legend()
    plt.show()

def sample_data(data, years, n_samples):
    sample_duration = 365 * years  # Duration of each sample in days
    samples = []
    # Ensure we calculate max_start_date correctly by not multiplying by 365 again
    max_start_date = data.index.max() - pd.Timedelta(days=sample_duration)
    for _ in range(n_samples):
        # Random start date selection within the allowable range
        start_date = random.choice(pd.date_range(data.index.min(), max_start_date))
        end_date = start_date + pd.Timedelta(days=sample_duration)
        sample = data.loc[start_date:end_date]
        if not sample.empty:
            samples.append(sample)
    return samples

def calculate_mse(data_with_trades, initial_capital, final_portfolio_value):
    # 计算直线的斜率（每日增加的价值）
    total_days = (data_with_trades.index[-1] - data_with_trades.index[0]).days
    daily_increase = (final_portfolio_value - initial_capital) / total_days

    # 计算每个时间点的预期投资组合价值
    expected_values = [initial_capital + daily_increase * (current_date - data_with_trades.index[0]).days for current_date in data_with_trades.index]

    # 计算实际值与预期值之间的差异的平方
    differences_squared = [(actual - expected)**2 for actual, expected in zip(data_with_trades['portfolio_value'], expected_values)]

    # 计算MSE
    mse = sum(differences_squared) / len(differences_squared)
    return mse

if __name__ == "__main__":
    symbol = 'TQQQ'
    years_in_sample = 3
    n_samples = 15
    profit_to_buy_options = [-0.5,-1,-1.5,-2,-2.5,-3, -8]
    profit_to_sell_options = [35,40, 50,70,150,250]
    full_data = fetch_data(symbol, '2014-01-01', '2024-01-01')

    results = []

    for profit_to_buy in profit_to_buy_options:
        for profit_to_sell in profit_to_sell_options:
            sample_profits = []
            sample_mse = []
            for i in range(n_samples):
                samples = sample_data(full_data, years_in_sample, 1)[0]
                data_with_signals = prepare_signals(samples)
                initial_capital = 10.0
                data_with_trades = execute_trades(data_with_signals, initial_capital, profit_to_buy=profit_to_buy, profit_to_sell=profit_to_sell)
                final_portfolio_value = data_with_trades['portfolio_value'].iloc[-1]
                profit = (final_portfolio_value - initial_capital) / initial_capital * 100
                mse = calculate_mse(data_with_trades, initial_capital, final_portfolio_value)
                sample_profits.append(profit)
                sample_mse.append(mse)
                print(f"Buy at {profit_to_buy}%, Sell at {profit_to_sell}% - Sample {i+1}: Final Portfolio Value: ${final_portfolio_value:.2f}, Profit: {profit:.2f}%, MSE: {mse:.2f}")

            avg_profit = np.mean(sample_profits)
            profit_volatility = np.std(sample_profits)
            avg_mse = np.mean(sample_mse)
            results.append([profit_to_buy, profit_to_sell, avg_profit, profit_volatility, avg_mse])

    results_df = pd.DataFrame(results, columns=['Profit to Buy', 'Profit to Sell', 'Average Profit rate', 'Profit Volatility', 'Average MSE'])

    results_df.to_excel('trading_strategy_results_with_mse.xlsx', index=False, engine='openpyxl')

    print(results_df)