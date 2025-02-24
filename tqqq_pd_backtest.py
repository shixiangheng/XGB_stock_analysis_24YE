import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from tqqq_utils import fetch_data

from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
# Fetch Data (for a specific symbol, like TQQQ)
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Calculate Daily Returns
def calculate_daily_return(df):
    df['daily_return'] = df['Close'].pct_change() * 100
    return df

# Define Binary Signal based on Threshold
def generate_binary_signal(df, lower_threshold=-3, upper_threshold=5):
    df['Signal'] = np.where(df['daily_return'] < lower_threshold, 1,  # Signal: 1 if return < threshold
                            0)  # Signal: 0 otherwise
    return df

# Add Technical Indicators
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

# Load historical data and preprocess it
symbol = "TQQQ"
start_date = "2017-01-31"
end_date = "2025-01-31"
lower_threshold = -2.5
upper_threshold = 5

# Fetch data and calculate features
data = fetch_data(symbol, start_date, end_date)
data = calculate_daily_return(data)
data = generate_binary_signal(data, lower_threshold, upper_threshold)
data = add_technical_indicators(data)
data.dropna(inplace=True)  # Drop rows with NaN values from rolling calculations

# Features for the model
features = ['SMA_1', 'SMA_2', 'RSI', 'Volatility']
X = data[features]
y = data['Signal']

# Load the pre-trained model
model_path = "tqqq_pd.pkl"
model = joblib.load(model_path)

# Make predictions
y_pred_prob = model.predict_proba(X)[:, 1]  # Probability of return < threshold
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Save results to a DataFrame
results = data.copy()
results['Predicted Signal'] = y_pred
results['pd'] = y_pred_prob

# Save results to an Excel file
output_file = "tqqq_pd.xlsx"
results.to_excel(output_file, index=True, sheet_name="Predictions")
print(f"Results saved to {output_file}")

# Evaluate the model
print("Classification Report:")
print(classification_report(y, y_pred, target_names=["Return >= "+str(lower_threshold)+"%", "Return < "+str(lower_threshold)+"%"]))

# Calculate AUC-ROC
auc = roc_auc_score(y, y_pred_prob)
print(f"AUC-ROC: {auc:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

# Plot predictions vs actual returns
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['daily_return'], label="Actual Daily Return", color="blue", alpha=0.7)

# Mark predicted signals
signal_dates = data.index[y_pred == 1]
plt.scatter(signal_dates, data['daily_return'][y_pred == 1], 
            label="Predicted Return < "+str(lower_threshold)+"%", color="red", zorder=5, alpha=0.7)

plt.axhline(lower_threshold, color="gray", linestyle="--", label="Threshold: "+str(lower_threshold)+"%")
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.title(f'Daily Returns with Predicted Signals for {symbol}')
plt.legend()
plt.grid()
plt.show()
