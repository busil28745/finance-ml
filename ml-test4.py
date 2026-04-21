import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
from datetime import datetime, timedelta

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=FutureWarning)

def run_stock_prediction_model(ticker, hold_threshold=0.0010):
    """
    특정 주식 종목의 10분 단위 추세를 분석하고 예측하는 머신러닝 파이프라인
    :param ticker: 야후 파이낸스 종목 코드 (예: '005930.KS')
    :param hold_threshold: '보합(Hold)'으로 간주할 수익률 임계값 (기본 0.10%)
    """
    print(f"\n{'='*60}")
    print(f"[{ticker}] 주가 방향 예측 및 백테스팅 파이프라인 시작")
    print(f"{'='*60}")

    START_DATE = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
    INTERVALS_IN_TRADING_DAY = 39 # 한국 증시 기준 10분 캔들 약 39개

    # 1. 데이터 수집
    print(f"1. 데이터 수집 중... (5분 데이터를 10분 단위로 변환)")
    # progress=False로 설정하여 콘솔 출력을 깔끔하게 만듭니다.
    df = yf.download(ticker, start=START_DATE, interval='5m', progress=False)

    if df.empty:
        print(f"[{ticker}] 데이터를 다운로드하지 못했습니다. 종목 코드나 기간을 확인해주세요.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 10분 단위 리샘플링
    df = df.resample('10min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # 2. 데이터 전처리 및 피처 엔지니어링
    print("2. 기술적 지표 및 과거 추세(Lagged Features) 생성 중...")
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

    # 과거 추세 반영
    features_to_lag = ['MA5_Ratio', 'RSI', 'MACD', 'Percent_B', 'Period_Return', 'Stoch_K', 'ATR_Ratio']
    lag_features_short = []
    lag_features_daily = []

    for feature in features_to_lag:
        for lag in range(1, 4):
            col_name = f'{feature}_lag_{lag}p'
            df[col_name] = df[feature].shift(lag)
            lag_features_short.append(col_name)
        
        for lag_hour in [1, 2]:
            lag_period = lag_hour * 6
            col_name = f'{feature}_lag_{lag_hour}h'
            df[col_name] = df[feature].shift(lag_period)
            lag_features_short.append(col_name)

        for lag in range(1, 8):
            col_name = f'{feature}_lag_{lag}d'
            df[col_name] = df[feature].shift(lag * INTERVALS_IN_TRADING_DAY)
            lag_features_daily.append(col_name)

    df = df.dropna()

    base_features = ['MA5_Ratio', 'MA20_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'Percent_B', 'Volume_Ratio', 'Period_Return', 'Stoch_K', 'Stoch_D', 'ATR_Ratio']
    X = df[base_features + lag_features_short + lag_features_daily]

    # Target 생성 (3-Class)
    future_return = df['Close'].pct_change().shift(-1)
    conditions = [future_return > hold_threshold, future_return < -hold_threshold]
    choices = [2, 0]
    y = np.select(conditions, choices, default=1)
    y = pd.Series(y, index=df.index)

    X = X[:-1]
    y = y[:-1]

    if len(X) < 100:
        print(f"[{ticker}] 학습할 수 있는 데이터가 부족합니다. (현재 길이: {len(X)})")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # 3. 모델 학습
    print("3. 모델 학습 및 하이퍼파라미터 최적화 중... (시간이 소요될 수 있습니다)")
    param_grid = {'max_depth': [3, 5], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 300], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    tscv = TimeSeriesSplit(n_splits=3)
    # verbose=0으로 설정하여 콘솔이 지저분해지는 것을 방지합니다.
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=0, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    print(f"   => 최적 파라미터 적용 완료: {grid_search.best_params_}")

    # 4. 모델 평가 및 백테스팅 계산
    predictions = model.predict(X_test)
    print(f"\n4. 예측 결과 분석 (다음 '10분' 방향성)")
    print(f"   => 전체 정확도: {accuracy_score(y_test, predictions):.4f}")
    print("\n[상세 분류 보고서]")
    print(classification_report(y_test, predictions, target_names=['하락(0)', '보합(1)', '상승(2)']))

    actual_returns = df.loc[X_test.index, 'Close'].pct_change().shift(-1).fillna(0)
    strategy_returns = actual_returns * (predictions == 2)
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    cumulative_buy_and_hold_returns = (1 + actual_returns).cumprod()

    print(f"\n[백테스팅 최종 결과 (Test 기간 기준)]")
    print(f"   => 모델 전략 누적 수익률: {(cumulative_strategy_returns.iloc[-1] - 1) * 100:.2f}%")
    print(f"   => 단순 보유(Buy & Hold) 수익률: {(cumulative_buy_and_hold_returns.iloc[-1] - 1) * 100:.2f}%")

    # 5. 시각화 (하나의 창에 3개의 그래프를 모두 표시)
    print("\n5. 시각화 결과를 출력합니다...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 20))

    pd.Series(model.feature_importances_, index=X.columns).nlargest(15).sort_values().plot(kind='barh', color='darkcyan', ax=axes[0])
    axes[0].set_title(f'[{ticker}] Top 15 Feature Importances for 10m Trend Prediction')
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)

    test_prices = df.loc[X_test.index, 'Close']
    axes[1].plot(test_prices.index, test_prices, label='Close Price', color='black', alpha=0.5, linewidth=1.5)
    axes[1].scatter(test_prices[predictions == 2].index, test_prices[predictions == 2], marker='^', color='green', label='Predict: Up', s=80, zorder=5)
    axes[1].scatter(test_prices[predictions == 1].index, test_prices[predictions == 1], marker='_', color='orange', label='Predict: Hold', s=80, zorder=5)
    axes[1].scatter(test_prices[predictions == 0].index, test_prices[predictions == 0], marker='v', color='red', label='Predict: Down', s=80, zorder=5)
    axes[1].set_title(f"[{ticker}] Price with Model Predictions")
    axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(cumulative_strategy_returns.index, cumulative_strategy_returns, label='Model Strategy Cumulative Return', color='royalblue', linewidth=2)
    axes[2].plot(cumulative_buy_and_hold_returns.index, cumulative_buy_and_hold_returns, label='Buy & Hold Benchmark', color='grey', linestyle='--', linewidth=2)
    axes[2].set_title(f"[{ticker}] Backtesting Results")
    axes[2].legend(); axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# 파일 직접 실행 시 아래 종목들에 대해 차례대로 파이프라인을 실행합니다.
if __name__ == "__main__":
    # 원하는 주식 코드를 리스트처럼 자유롭게 추가해서 테스트해보세요.
    run_stock_prediction_model('005930.KS') # 삼성전자
    
    # 주석을 풀면 SK하이닉스도 이어서 검사합니다.
    # run_stock_prediction_model('000660.KS') # SK하이닉스