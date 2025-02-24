# tqqq utils
import pandas as pd
import yfinance as yf
import pandas as pd
import numpy as np
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def calculate_avg_daily_return(df, window=22):
    """
    Calculate the average daily percentage return for the past 'window' days.

    Parameters:
        df (pd.DataFrame): A DataFrame with a 'Close' column.
        window (int): The number of days over which to calculate the average daily return (default is 22).

    Returns:
        pd.DataFrame: A DataFrame with the average daily percentage return by date.
    """
    if 'Close' not in df.columns:
        raise ValueError("The DataFrame must contain a 'Close' column.")

    # Calculate daily percentage returns
    df['daily_return'] = df['Close'].pct_change() * 100

    # Calculate the rolling average of daily returns
    df['avg_daily_return'] = df['daily_return'].rolling(window=window).mean()

    # Return the DataFrame with dates and the average daily return
    result_df = df#[['avg_daily_return']].dropna()
    return result_df

def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Black-Scholes Option Pricing & Greeks
def black_scholes_greeks(S, K, T, r, sigma, option_type="put"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)
    ) / 252
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    price = (
        S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        if option_type == "call"
        else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    )
    return price, delta, gamma, theta, vega

# Dynamic Implied Volatility (based on historical volatility)
def dynamic_iv(close_prices, window=14, scaling_factor=1.5):
    log_returns = np.log(close_prices / close_prices.shift(1))
    rolling_volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
    implied_volatility = rolling_volatility * scaling_factor
    return implied_volatility.fillna(method="bfill")  # 处理缺失值


# Remove time zone information from DataFrame
def remove_timezone(data):
    date_col = 'Date'
    # Check if the DataFrame has an index with datetime and a timezone
    if isinstance(data.index, pd.DatetimeIndex):
        # Check if the datetime index has a timezone
        if data.index.tz is not None:
            # Remove timezone awareness from the index
            data.index = data.index.tz_localize(None)
        # If index has timezone awareness, remove it, otherwise proceed as normal
    else:
        # Check if any datetime column has timezone information
        for column in data.select_dtypes(include=[object, 'datetime']):
            if isinstance(data[column].dtype, pd.Timestamp):
                if data[column].dt.tz is not None:
                    data[column] = data[column].dt.tz_localize(None)
    output_data = data.copy()
    # If index is timezone-aware, reset the index
    if isinstance(data.index, pd.DatetimeIndex):
        if data.index.tz is None:
            output_data=data.reset_index()  # Reset the index to a simple range
    output_data[date_col] = output_data[date_col].astype(str)
    return output_data
def add_technical_indicators(df):
    df['SMA_1'] = df['Close'].rolling(window=40).mean()
    df['SMA_2'] = df['Close'].rolling(window=100).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    return df


def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Define a function to normalize and format the date column
def normalize_date(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col]).dt.date  # Ensure date-only format
    return df
def add_return_percentage(df, price_col, new_col_name, days=22):
    """
    Add a column to the DataFrame that shows the return percentage
    comparing the value from 'price_col' today to the value 'days' ago.

    Args:
        df (pd.DataFrame): The input DataFrame with a date index or sorted by date.
        price_col (str): The column name for the price data.
        new_col_name (str): The name for the new column to store the return percentage.
        days (int): The number of days to look back for the comparison.

    Returns:
        pd.DataFrame: The DataFrame with the new column added.
    """
    # Calculate the return percentage and add as a new column
    df[new_col_name] = (df[price_col] - df[price_col].shift(days)) / df[price_col].shift(days) 
    return df


def add_put_option_prices(df, stock_price_col, implied_volatility_col, risk_free_rate, days_to_expiration):
    """
    Add a column for put option prices to a DataFrame.

    Args:
        df (pd.DataFrame): The historical data DataFrame.
        stock_price_col (str): The column name for the stock price.
        implied_volatility_col (str): The column name for implied volatility.
        risk_free_rate (float): The risk-free interest rate.
        days_to_expiration (int): Days to option expiration.

    Returns:
        pd.DataFrame: The DataFrame with the new column for put option prices.
    """
    put_prices = []

    for i, row in df.iterrows():
        stock_price = row[stock_price_col]
        implied_volatility = row[implied_volatility_col]

        # Ensure valid data for calculation
        if np.isnan(stock_price) or np.isnan(implied_volatility):
            put_prices.append(np.nan)
            continue

        # Black-Scholes parameters
        strike_price = round(stock_price)
        T = days_to_expiration / 252

        # Calculate the put price
        price, delta, gamma, theta, vega = black_scholes_greeks(
            S=stock_price,
            K=strike_price,
            T=T,
            r=risk_free_rate,
            sigma=implied_volatility,
            option_type="put",
        )

        # Append put price to the list
        put_prices.append(price * 100)  # Option price in dollar terms for 100 shares

    # Add the put prices as a new column to the DataFrame
    df["Put_Price"] = put_prices

    return df


def calculate_down_percentage(historical_data, window=22):
    """
    Calculate the down percentage (min/max) over a given window (default 22 days).
    Ensures:
    - The max price is not on the last day of the window.
    - The max price date comes before the min price date.
    
    Adds a new column 'Down_Percentage' aligned with the max price date.
    
    :param historical_data: DataFrame with a 'Close' column containing daily prices.
    :param window: The number of days to consider (default 22).
    :return: DataFrame with the new 'Down_Percentage' column.
    """
    down_percentage = [None] * len(historical_data)  # Placeholder for results

    for i in range(len(historical_data) - window):
        # Define the 22-day window
        window_data = historical_data.iloc[i:i+window]
        
        # Identify the max price but ensure it's not on the last day
        max_idx = window_data['Close'].idxmax()
        if max_idx == window_data.index[-1]:  # If max is on last day, skip
            continue

        # Find the min price after the max price
        min_idx = window_data.loc[max_idx:].idxmin()['Close']

        if min_idx > max_idx:  # Ensure max date is before min date
            max_price = historical_data.loc[max_idx, 'Close']
            min_price = historical_data.loc[min_idx, 'Close']
            down_pct = (min_price - max_price) / max_price  # Calculate drop percentage
            
            down_percentage[i] = down_pct  # Store result at max date

    historical_data['Down_Percentage'] = down_percentage  # Add to DataFrame
    return historical_data

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
    # Example Usage
    # Assume `data` is a DataFrame with a 'Close' column
    symbol = "TQQQ"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    data = fetch_data(symbol, start_date, end_date)
    
    avg_return_df = calculate_avg_daily_return(data)
    print(avg_return_df)
