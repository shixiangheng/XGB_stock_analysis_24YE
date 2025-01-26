import xgboost as xgb
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Fetch Data (for a specific symbol, like TQQQ)
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Calculate 5-day Return Percentage
def calculate_5_day_return(df):
    df['5_day_return'] = df['Close'].pct_change(periods=5) * 100
    return df

# Define Signal based on 5-day return percentage
def generate_signal(df):
    # Convert continuous returns into signals (buy, hold, sell)
    df['Signal'] = np.where(df['5_day_return'] > 10, 0,  # Buy Signal
                            np.where(df['5_day_return'] < -10, 1,  # Sell Signal
                                     0.5))  # Hold Signal (mapped to 0.5)
    return df

# Add technical features for model (e.g., moving averages, RSI)
def add_technical_indicators(df):
    df['SMA_1'] = df['Close'].rolling(window=40).mean()
    df['SMA_2'] = df['Close'].rolling(window=100).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    return df

# Calculate RSI (Relative Strength Index)
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Train XGBoost model using XGBClassifier (for classification)
def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    return model

# Prepare Data for model training
symbol = "TQQQ"
start_date = "2017-12-01"
end_date = "2023-12-01"

# Fetch Data
data = fetch_data(symbol, start_date, end_date)

# Calculate 5-day return and generate signals
data = calculate_5_day_return(data)
data = generate_signal(data)

# Add Technical Indicators
data = add_technical_indicators(data)

# Features for the model
features = ['SMA_1', 'SMA_2', 'RSI', 'Volatility']
X = data[features]
y = data['Signal'].astype('int')  # Ensure the signals are integers for classification

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = train_xgboost(X_train, y_train, X_test, y_test)

# Make predictions
y_pred_prob = model.predict_proba(X_test)  # Get predicted probabilities (not the discrete class)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels (0=Buy, 1=Hold, 2=Sell)

# Classification report
# Classification report with specified labels
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Buy', 'Hold', 'Sell'], labels=[0, 1, 2]))


# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Plot the actual stock price (Close)
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y_test):], data['Close'][-len(y_test):], label='Actual Stock Price', color='blue', alpha=0.7)

low = 0.10
high = 0.90
# Mark Buy and Sell points
buy_signals = data.index[-len(y_test):][y_pred == 0]
sell_signals = data.index[-len(y_test):][y_pred == 2]

# Plot Buy points (green triangle up)
plt.scatter(buy_signals, data['Close'][-len(y_test):][y_pred == 0], label='Buy Signal', marker='^', color='green', alpha=1, zorder=5)

# Plot Sell points (red triangle down)
plt.scatter(sell_signals, data['Close'][-len(y_test):][y_pred == 2], label='Sell Signal', marker='v', color='red', alpha=1, zorder=5)

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'Stock Price with Buy/Sell Signals for {symbol}')
plt.legend()
plt.grid(True)
plt.show()
