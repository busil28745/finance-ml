import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import joblib
import os
import pytz

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=FutureWarning)

def get_latest_features(ticker):
    """
    단일 종목의 가장 최근 데이터를 다운로드하고 전처리하여,
    '현재 시점'의 피처(X) 1개 행만을 반환합니다.
    """
    INTERVALS_IN_TRADING_DAY = 39 
    
    kst = pytz.timezone('Asia/Seoul')
    START_DATE = (datetime.now(kst) - timedelta(days=59)).strftime('%Y-%m-%d')
    
    df = yf.download(ticker, start=START_DATE, interval='5m', progress=False)

    if df.empty:
        print(f"[{ticker}] 데이터를 다운로드하지 못했습니다.")
        return None, None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.resample('10min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    # --- 데이터 전처리 및 피처 엔지니어링 (학습 때와 100% 동일해야 함) ---
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

    # 가장 최근 캔들 데이터(마지막 행)만 추출
    latest_X = X.iloc[[-1]]
    latest_price = df['Close'].iloc[-1]
    candle_time = df.index[-1]

    # yfinance에서 가져온 시간이 UTC 등 다른 시간대일 경우 한국 시간(KST)으로 변환
    if candle_time.tzinfo is not None:
        candle_time = candle_time.tz_convert('Asia/Seoul')

    return latest_X, latest_price, candle_time

def predict_realtime(tickers, model_filename='global_stock_model.pkl'):
    if not os.path.exists(model_filename):
        print(f"\n[오류] 저장된 모델 파일('{model_filename}')을 찾을 수 없습니다.")
        return

    kst = pytz.timezone('Asia/Seoul')
    print(f"\n[{datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')}] 실시간 주가 예측을 시작합니다...")
    global_model = joblib.load(model_filename)

    for ticker in tickers:
        latest_X, current_price, candle_time = get_latest_features(ticker)
        if latest_X is None: continue

        # yfinance를 통해 종목명(Company Name) 가져오기
        try:
            stock_name = yf.Ticker(ticker).info.get('longName', ticker)
        except Exception:
            stock_name = ticker # 이름 가져오기 실패 시 종목코드 출력

        # 모델 예측 (0: 하락, 1: 보합, 2: 상승)
        pred = global_model.predict(latest_X)[0]
        
        # 각 클래스별 확률값(Confidence) 계산
        probs = global_model.predict_proba(latest_X)[0]
        prob_down, prob_hold, prob_up = probs[0], probs[1], probs[2]

        print(f"\n{'-'*50}")
        print(f"📈 종목: {stock_name} ({ticker})")
        print(f"🕒 기준시간: {candle_time.strftime('%Y-%m-%d %H:%M:%S')} (가장 최근 완성된 10분 캔들)")
        print(f"💰 현재가격: {current_price:,.0f} KRW")
        
        if pred == 2:
            print(f"🚀 AI 예측: 상승 (매수 🟢)  [확률: {prob_up*100:.1f}%]")
        elif pred == 1:
            print(f"➖ AI 예측: 보합 (관망 🟡)  [확률: {prob_hold*100:.1f}%]")
        else:
            print(f"📉 AI 예측: 하락 (매도 🔴)  [확률: {prob_down*100:.1f}%]")
        
        print(f"   (상세 확률 -> 하락: {prob_down*100:.1f}% | 보합: {prob_hold*100:.1f}% | 상승: {prob_up*100:.1f}%)")

if __name__ == "__main__":
    # 지금 당장 다음 10분의 방향이 궁금한 종목들을 입력하세요
    target_tickers = ['005930.KS', '000660.KS', '035720.KS', '005380.KS'] # 삼성전자, SK하이닉스, 카카오, 현대차
    predict_realtime(target_tickers)