import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import joblib
import os

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=FutureWarning)

def get_ticker_data(ticker, start_date, hold_threshold=0.0010):
    """
    단일 종목의 데이터를 다운로드하고 피처(Feature)와 타겟(Target)을 생성하여 반환합니다.
    (ml-test7.py와 동일한 전처리 과정을 거쳐야 모델이 정상적으로 인식합니다.)
    """
    INTERVALS_IN_TRADING_DAY = 39 
    
    df = yf.download(ticker, start=start_date, interval='5m', progress=False)

    if df.empty:
        print(f"[{ticker}] 데이터를 다운로드하지 못했습니다.")
        return None, None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.resample('10min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA5_Ratio'] = df['Close'] / df['MA5']
    df['MA20_Ratio'] = df['Close'] / df['MA20']

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['20STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['20STD'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['20STD'] * 2)
    df['Percent_B'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])

    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()

    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Period_Return'] = df['Close'].pct_change()

    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()
    df['ATR_Ratio'] = df['ATR'] / df['Close']

    features_to_lag = ['MA5_Ratio', 'RSI', 'MACD', 'Percent_B', 'Period_Return', 'Stoch_K', 'ATR_Ratio']
    lag_features_short = []
    lag_features_daily = []
    new_columns = {} 

    for feature in features_to_lag:
        for lag in range(1, 4):
            col_name = f'{feature}_lag_{lag}p'
            new_columns[col_name] = df[feature].shift(lag)
            lag_features_short.append(col_name)
        for lag_hour in [1, 2]:
            lag_period = lag_hour * 6
            col_name = f'{feature}_lag_{lag_hour}h'
            new_columns[col_name] = df[feature].shift(lag_period)
            lag_features_short.append(col_name)
        for lag in range(1, 8):
            col_name = f'{feature}_lag_{lag}d'
            new_columns[col_name] = df[feature].shift(lag * INTERVALS_IN_TRADING_DAY)
            lag_features_daily.append(col_name)

    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    df = df.dropna()

    base_features = ['MA5_Ratio', 'MA20_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'Percent_B', 'Volume_Ratio', 'Period_Return', 'Stoch_K', 'Stoch_D', 'ATR_Ratio']
    X = df[base_features + lag_features_short + lag_features_daily]

    future_return = df['Close'].pct_change().shift(-1)
    conditions = [future_return > hold_threshold, future_return < -hold_threshold]
    choices = [2, 0]
    y = np.select(conditions, choices, default=1)
    y = pd.Series(y, index=df.index)

    X, y, df = X[:-1], y[:-1], df[:-1]

    if len(X) < 100:
        print(f"[{ticker}] 검증할 수 있는 데이터가 부족합니다.")
        return None, None, None
        
    return X, y, df

def test_unseen_tickers(tickers, model_filename='global_stock_model.pkl'):
    if not os.path.exists(model_filename):
        print(f"\n[오류] 저장된 모델 파일('{model_filename}')을 찾을 수 없습니다.")
        print("먼저 ml-test7.py를 실행하여 모델을 학습하고 저장해주세요.")
        return

    print(f"\n저장된 글로벌 모델('{model_filename}')을 불러옵니다...")
    global_model = joblib.load(model_filename)
    print("=> 모델 로드 완료!\n")

    START_DATE = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')

    for ticker in tickers:
        print(f"{'-'*60}")
        print(f"[{ticker}] 미학습 신규 종목 예측 및 백테스팅 (전체 기간 적용)")
        
        X, y, df = get_ticker_data(ticker, START_DATE)
        if X is None: continue

        # 모델이 한 번도 본 적 없는 데이터이므로 전체(X)를 테스트 데이터로 사용
        predictions = global_model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"=> 모델 정확도: {accuracy:.4f}")
        
        actual_returns = df.loc[X.index, 'Close'].pct_change().shift(-1).fillna(0)
        strategy_returns = actual_returns * (predictions == 2)
        
        cumulative_strategy_returns = (1 + strategy_returns).cumprod()
        cumulative_buy_and_hold_returns = (1 + actual_returns).cumprod()

        print(f"=> 모델 전략 누적 수익률: {(cumulative_strategy_returns.iloc[-1] - 1) * 100:.2f}%")
        print(f"=> 단순 보유(Buy & Hold) 수익률: {(cumulative_buy_and_hold_returns.iloc[-1] - 1) * 100:.2f}%")

if __name__ == "__main__":
    # 학습에 사용하지 않았던 완전히 새로운 종목들
    unseen_tickers = ['035720.KS', '005380.KS', '068270.KS'] # 카카오, 현대차, 셀트리온
    
    # 이미 학습되어 저장된 global_stock_model.pkl 을 불러와서 성능을 평가합니다.
    test_unseen_tickers(unseen_tickers)