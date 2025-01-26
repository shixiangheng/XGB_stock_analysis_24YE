

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
ticker_symbol = "TQQQ"
vix_symbol = "^VIX"  # VIX symbol
start_date = "2024-01-31"
end_date = "2025-01-31"
shares_held = 30  # Holding number of shares
rsi_high_bond = 70  # RSI threshold

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Fill missing values with 50 to avoid affecting strategy

# Download stock price and VIX data
ticker = yf.Ticker(ticker_symbol)
historical_data = ticker.history(start=start_date, end=end_date)

vix = yf.Ticker(vix_symbol)
vix_data = vix.history(start=start_date, end=end_date)

# Check column names for VIX data to ensure we use the correct one
print("VIX Data Columns:", vix_data.columns)

# Calculate RSI
historical_data['RSI'] = calculate_rsi(historical_data)

# Calculate SMA40 and SMA100
historical_data['SMA40'] = historical_data['Close'].rolling(window=40).mean()
historical_data['SMA100'] = historical_data['Close'].rolling(window=100).mean()

# Modify RSI according to price rule
for i in range(len(historical_data)):
    current_price = historical_data['Close'].iloc[i]
    
    if i + 22 < len(historical_data):  # Check for the next 22 days
        future_prices = historical_data['Close'].iloc[i + 1:i + 22]
        if current_price * 0.9 < future_prices.mean():
            historical_data['RSI'].iloc[i] = rsi_high_bond  # Set RSI to 70

# Resample VIX data to match the stock data frequency (daily)
vix_data = vix_data[['Close']].resample('D').ffill()  # Forward-fill missing VIX data for daily frequency

# Check column names in VIX data to ensure correct merging
print("VIX Data Resampled Columns:", vix_data.columns)

# Merge VIX data with stock data on date index
historical_data = historical_data.merge(vix_data, left_index=True, right_index=True, how='left')
historical_data.rename(columns={'Close_y': 'VIX'}, inplace=True)

# Plot the data
fig, ax1 = plt.subplots(figsize=(14, 8))
historical_data=historical_data.rename(columns={'Close_x':'Close'})

# Plot stock price, SMA40, and SMA100
ax1.plot(historical_data.index, historical_data['Close'], label="Stock Price", color="blue", alpha=0.7)
#ax1.plot(historical_data.index, historical_data['SMA40'], label="SMA40", color="green", linestyle="--")
#ax1.plot(historical_data.index, historical_data['SMA100'], label="SMA100", color="orange", linestyle="--")
ax1.set_xlabel("Date")
ax1.set_ylabel("Stock Price")
ax1.set_title(f"{ticker_symbol} Stock Price, SMA40, SMA100, RSI, and VIX")
ax1.legend(loc="upper left")
ax1.grid(True)

# Create a second y-axis to plot RSI and VIX
ax2 = ax1.twinx()
ax2.plot(historical_data.index, historical_data['RSI'], label="RSI", color="purple", alpha=0.7)
ax2.axhline(70, color="red", linestyle="--", label="RSI High Bond (70)")  # RSI threshold
ax2.set_ylabel("RSI")

# Create a third y-axis to plot VIX
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset VIX axis
ax3.plot(historical_data.index, historical_data['VIX'], label="VIX", color="red", alpha=0.7)
ax3.set_ylabel("VIX")

# Display the plot
fig.tight_layout()
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax3.legend(loc="lower right")
plt.show()
