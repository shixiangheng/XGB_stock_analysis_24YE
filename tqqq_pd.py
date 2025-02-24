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
from tqqq_utils import fetch_data


SMA_1 = 'SMA_1'
SMA_2 = 'SMA_2'
# Calculate Daily Returns
def calculate_daily_return(df):
    df['daily_return'] = df['Close'].pct_change() * 100
    return df

# Define Binary Signal based on Threshold
def generate_binary_signal(df, lower_threshold=-3, upper_threshold=5):
    df['Signal'] = np.where(df['daily_return'] < lower_threshold, 1,  # Signal: 1 if return < -5%
                            0)  # Signal: 0 otherwise
    return df

# Add Technical Indicators
def add_technical_indicators(df):
    df[SMA_1] = df['Close'].rolling(window=40).mean()
    df[SMA_2] = df['Close'].rolling(window=100).mean()
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

# Train XGBoost model with class weights
def train_xgboost(X_train, y_train, class_weights):
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", scale_pos_weight=class_weights)
    model.fit(X_train, y_train)
    return model

# Fetch data for the symbol
symbol = "TQQQ"
start_date = "2017-01-31"
end_date = "2025-01-31"
data = fetch_data(symbol, start_date, end_date)
lower_threshold = -2.5
upper_threshold = 5
# Calculate daily returns and generate binary signals
data = calculate_daily_return(data)
data = generate_binary_signal(data,lower_threshold,upper_threshold)

# Add technical indicators
data = add_technical_indicators(data)

# Drop NaN values caused by rolling calculations
data.dropna(inplace=True)

# Features for the model
features = [SMA_1, SMA_2, 'RSI', 'Volatility']
X = data[features]
y = data['Signal']  # Binary target: 1 if return < -5%, 0 otherwise

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalance using SMOTE (Oversampling the minority class)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Compute class weights to handle imbalance (as a fallback)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_ratio = class_weights[1] / class_weights[0]

# Train the XGBoost model with SMOTE-rebalanced data
model = train_xgboost(X_train_smote, y_train_smote, class_weights_ratio)
joblib.dump(model, "tqqq_pd.pkl")
# Make predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of return < -5%
y_pred = (y_pred_prob > 0.9).astype(int)  # Convert probabilities to binary predictions

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Return >= "+str(lower_threshold)+"%", "Return < "+str(upper_threshold)+"%"]))

# Calculate AUC-ROC
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC: {auc:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
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
plt.plot(data.index[-len(y_test):], data['daily_return'][-len(y_test):], label="Actual Daily Return", color="blue", alpha=0.7)

# Mark predicted signals
signal_dates = data.index[-len(y_test):][y_pred == 1]
plt.scatter(signal_dates, data['daily_return'][-len(y_test):][y_pred == 1], 
            label="Predicted Return < -5%", color="red", zorder=5, alpha=0.7)

plt.axhline(lower_threshold, color="gray", linestyle="--", label="Threshold: "+str(lower_threshold)+"%")
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.title(f'Daily Returns with Predicted Signals for {symbol}')
plt.legend()
plt.grid()
plt.show()
