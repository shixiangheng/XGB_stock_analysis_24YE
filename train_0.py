import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import random

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


if __name__ == "__main__":
    symbol = 'TQQQ'
    years_in_sample = 3  # Number of years in one sample
    n_samples = 10  # Number of samples
    profit_to_buy = -3  # Loss percentage to trigger a buy
    profit_to_sell = 40  # Profit percentage to trigger a sell

    full_data = fetch_data(symbol, '2019-01-01', '2024-01-01')

    # Store metrics for all runs to calculate average later
    all_profits = []

    for i in range(n_samples):
        samples = sample_data(full_data, years_in_sample, 1)[0]
        data_with_signals = prepare_signals(samples)
        initial_capital = 10.0
        data_with_trades = execute_trades(data_with_signals,initial_capital, profit_to_buy=profit_to_buy, profit_to_sell=profit_to_sell)
        final_portfolio_value = data_with_trades['portfolio_value'].iloc[-1]
        profit = (final_portfolio_value - initial_capital) / initial_capital * 100
        all_profits.append(profit)
        print(f"Sample {i+1}: Final Portfolio Value: ${final_portfolio_value:.2f}, Profit: {profit:.2f}%")
        plot_performance(data_with_trades)

    avg_profit = np.mean(all_profits)
    profit_volatility = np.std(all_profits)
    print(f"Average Profit across samples: {avg_profit:.2f}%")
    print(f"Profit Volatility across samples: {profit_volatility:.2f}%")
