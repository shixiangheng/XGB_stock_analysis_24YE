from pytrends.request import TrendReq
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqqq_xgboost_signal import fetch_data,calculate_rsi,backtest,majority_vote,train_xgboost,create_labels
from tqqq_xgboost_signal__vote_imp import forecast_for_date

# Step 1: Fetch Historical Data
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Step 2: Fetch VIX Data
def fetch_vix_data(start_date, end_date):
    vix = yf.download("^VIX", start=start_date, end=end_date)
    vix = vix[['Close']].rename(columns={'Close': 'VIX'})
    return vix

# Step 3: Fetch Google Trends Data
def fetch_google_trends_data(keywords, start_date, end_date):
    pytrends = TrendReq()
    pytrends.build_payload(keywords, timeframe=f"{start_date} {end_date}")
    trends_data = pytrends.interest_over_time()
    trends_data = trends_data.drop(columns='isPartial', errors='ignore')
    trends_data.to_excel('trends_data.xlsx',index=False)
    return trends_data

# Step 4: Build Sentiment Index
def build_sentiment_index(vix_data, trends_data):
    # Merge VIX and Trends data on date
    sentiment_data = vix_data.merge(trends_data, left_index=True, right_index=True, how='inner')

    # Normalize data for index calculation
    sentiment_data['VIX_Norm'] = (sentiment_data['VIX'] - sentiment_data['VIX'].min()) / (sentiment_data['VIX'].max() - sentiment_data['VIX'].min())
    for col in trends_data.columns:
        sentiment_data[f'{col}_Norm'] = (sentiment_data[col] - sentiment_data[col].min()) / (sentiment_data[col].max() - sentiment_data[col].min())

    # Calculate Sentiment Index (Weighted Average)
    sentiment_data['Sentiment_Index'] = 0.5 * sentiment_data['VIX_Norm'] + 0.5 * sentiment_data[[f'{col}_Norm' for col in trends_data.columns]].mean(axis=1)
    return sentiment_data[['Sentiment_Index']]

def add_features_vix(df, sentiment_index):
    sma1, sma2, rsi_window, volatility_window, daily_return_window = 40, 100, 20, 20, 5
    df['SMA_1'] = df['Close'].rolling(window=sma1).mean()
    df['SMA_2'] = df['Close'].rolling(window=sma2).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=rsi_window)
    df['Daily_Return'] = df['Close'].pct_change(periods=daily_return_window)
    df['Volatility'] = df['Close'].rolling(window=volatility_window).std()
    df = df.merge(sentiment_index, left_index=True, right_index=True, how='left')
    return df

# Step 6: Main Execution
if __name__ == "__main__":
    # Fetch TQQQ and VIX Data
    symbol = "TQQQ"
    today_date = datetime.today().strftime('%Y-%m-%d')
    target_date=today_date
    start_date = "2017-12-01"
    end_date = "2023-12-01"
    threshold = 0.10
    folder=r'C:\Users\shixiangheng\Desktop\Henz\stock\XGB_stock_analysis_24YE\model'
    data = fetch_data(symbol, start_date, end_date)
    vix_data = fetch_vix_data(start_date, end_date)

    # Fetch Google Trends Data
    keywords = ["stock market", "trading", "recession"]
    trends_data = fetch_google_trends_data(keywords, start_date, end_date)
    
    # Build Sentiment Index
    sentiment_index = build_sentiment_index(vix_data, trends_data)

    # Add Features
    data = create_labels(data,threshold)
    data = add_features_vix(data, sentiment_index)
    
    print(data.columns)
    # Prepare Data for Model
    features = ['SMA_1', 'SMA_2', 'RSI', 'Volatility', 'Daily_Return', 'Sentiment_Index']
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
    for i in range(5):
        # Split the data differently for each model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        # Train the model
        model = train_xgboost(X_train, y_train, X_test, y_test)
        model.save_model(folder+'\\tqqq_xgboost_model_vix_v0_model'+str(i)+'.json') 
        models.append(model)
    # Continue with existing model pipeline...
    
    models=[]
    for i in range(1):
        model.load_model(folder+f'\{symbol}_xgboost_model_vix_v0_model'+str(i)+'.json') 
        models.append(model)  # 20241230  # 20250123
    
    # Use the model for predictions
    # Forecast for the given date
    #data=forecast_for_date(symbol, target_date,oot_start_date, oot_end_date, model)
    oot_start_date = end_date
    oot_end_date = today_date
    imp=0
    oot_data_ori = fetch_data(symbol, start_date=oot_start_date, end_date=oot_end_date)
    vix_data = fetch_vix_data(oot_start_date, oot_end_date)
    # Fetch Google Trends Data
    keywords = ["stock market", "trading", "recession"]
    trends_data = fetch_google_trends_data(keywords, oot_start_date, oot_end_date)
    
    # Build Sentiment Index
    sentiment_index = build_sentiment_index(vix_data, trends_data)
    
    
    #y_pred_final = np.where(final_predictions == 0, -1, predictions - 1)
    #data = create_labels(oot_data_ori,threshold)
    oot_data_feature_vix = add_features_vix(oot_data_ori, sentiment_index)
    
    
    
    oot_data_feature_vix = create_labels(oot_data_feature_vix,threshold)
    #data_feature,final_predictions_mapped=forecast_for_date(oot_data_ori, features, symbol, target_date,oot_start_date, oot_end_date, models,imp)
    target = 'Signal'
    #X = oot_data[features]
    X_test_oot = oot_data_feature_vix[features]
    y_test_oot = oot_data_feature_vix[target]
    
    final_predictions_mapped = majority_vote(models, X_test_oot)
    
    print(classification_report(y_test_oot, final_predictions_mapped, target_names=['Sell', 'Hold', 'Buy']))
    backtest(oot_data_feature_vix, final_predictions_mapped, oot_data_feature_vix.index)
