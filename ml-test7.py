import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
from datetime import datetime, timedelta
import joblib
import os

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=FutureWarning)

def get_ticker_data(ticker, start_date, hold_threshold=0.0010):
    """
    단일 종목의 데이터를 다운로드하고 피처(Feature)와 타겟(Target)을 생성하여 반환합니다.
    """
    INTERVALS_IN_TRADING_DAY = 39 # 한국 증시 기준 10분 캔들 약 39개
    
    # progress=False로 설정하여 콘솔 출력을 깔끔하게 만듭니다.
    df = yf.download(ticker, start=start_date, interval='5m', progress=False)

    if df.empty:
        print(f"[{ticker}] 데이터를 다운로드하지 못했습니다.")
        return None, None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 10분 단위 리샘플링
    df = df.resample('10min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    # --- 데이터 전처리 및 피처 엔지니어링 ---
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

    # 과거 추세 반영 (Lagged Features)
    features_to_lag = ['MA5_Ratio', 'RSI', 'MACD', 'Percent_B', 'Period_Return', 'Stoch_K', 'ATR_Ratio']
    lag_features_short = []
    lag_features_daily = []
    new_columns = {} # 새로 추가할 컬럼들을 한 번에 모아둘 딕셔너리

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

    # 한 번에 병합(concat)하여 DataFrame 파편화(Fragmentation) 경고를 해결합니다.
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

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
    df = df[:-1]

    if len(X) < 100:
        print(f"[{ticker}] 학습할 수 있는 데이터가 부족합니다.")
        return None, None, None
        
    return X, y, df

def run_multi_stock_model(tickers, use_saved_model=False, model_filename='global_stock_model.pkl'):
    """
    여러 종목을 동시에 학습하는 단일 머신러닝 파이프라인
    """
    print(f"\n{'='*70}")
    print(f"다중 종목 통합 주가 추세 학습 모델 시작 (종목수: {len(tickers)}개)")
    print(f"{'='*70}")

    START_DATE = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
    
    train_data_list = []
    test_data_dict = {}

    # 1. 모든 종목의 데이터 수집 및 분할
    for ticker in tickers:
        print(f"[{ticker}] 데이터 수집 및 피처 생성 중...")
        X, y, df = get_ticker_data(ticker, START_DATE)
        if X is None: continue
        
        # 시계열 순서가 깨지지 않게 각 종목별로 8:2로 나눕니다.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        
        # 학습 데이터는 하나로 합치기 위해 리스트에 저장 (target 포함)
        train_df = X_train.copy()
        train_df['TARGET'] = y_train
        train_data_list.append(train_df)
        
        # 테스트 데이터는 나중에 개별 평가를 위해 딕셔너리에 저장
        test_data_dict[ticker] = {'X_test': X_test, 'y_test': y_test, 'df': df}

    # 2. 전 종목 학습 데이터 통합 및 시계열 정렬
    # 여러 종목의 데이터가 섞이더라도, 시간이 빠른 데이터가 앞에 오도록 인덱스(시간) 기준으로 정렬합니다.
    combined_train_df = pd.concat(train_data_list).sort_index()
    X_train_all = combined_train_df.drop(columns=['TARGET'])
    y_train_all = combined_train_df['TARGET']

    print(f"\n=> [통합 학습 데이터 생성 완료] 총 데이터 개수: {len(X_train_all)} 행")

    # 3. 글로벌 모델 학습 또는 불러오기
    if use_saved_model and os.path.exists(model_filename):
        print(f"\n저장된 모델('{model_filename}')을 불러옵니다...")
        global_model = joblib.load(model_filename)
        print("=> 모델 로드 완료!")
    else:
        print("\n통합 모델 학습 및 하이퍼파라미터 최적화 중... (데이터가 많아 시간이 소요될 수 있습니다)")
        param_grid = {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=0, scoring='accuracy')
        grid_search.fit(X_train_all, y_train_all)

        global_model = grid_search.best_estimator_
        print(f"=> 최적 파라미터 적용 완료: {grid_search.best_params_}")
        
        # 학습된 모델을 파일로 저장
        joblib.dump(global_model, model_filename)
        print(f"=> 학습된 모델을 '{model_filename}' 파일로 저장했습니다.")

    # 4. 글로벌 피처 중요도 시각화
    plt.figure(figsize=(10, 6))
    pd.Series(global_model.feature_importances_, index=X_train_all.columns).nlargest(15).sort_values().plot(kind='barh', color='teal')
    plt.title('Global Model: Top 15 Feature Importances (All Tickers Combined)')
    plt.xlabel('Importance Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() # 창을 닫아야 다음 개별 종목 평가가 진행됩니다.

    # 5. 개별 종목별 모델 평가 및 백테스팅
    for ticker, data in test_data_dict.items():
        print(f"\n{'-'*50}")
        print(f"[{ticker}] 단일 모델을 활용한 개별 종목 예측 및 백테스팅")
        
        X_test, y_test, df = data['X_test'], data['y_test'], data['df']
        predictions = global_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        print(f"=> 모델 정확도: {accuracy:.4f}")
        
        actual_returns = df.loc[X_test.index, 'Close'].pct_change().shift(-1).fillna(0)
        strategy_returns = actual_returns * (predictions == 2)
        cumulative_strategy_returns = (1 + strategy_returns).cumprod()
        cumulative_buy_and_hold_returns = (1 + actual_returns).cumprod()

        print(f"=> 모델 전략 누적 수익률: {(cumulative_strategy_returns.iloc[-1] - 1) * 100:.2f}%")
        print(f"=> 단순 보유(Buy & Hold) 수익률: {(cumulative_buy_and_hold_returns.iloc[-1] - 1) * 100:.2f}%")

        # 개별 종목 차트 그리기
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        test_prices = df.loc[X_test.index, 'Close']
        
        axes[0].plot(test_prices.index, test_prices, label='Close Price', color='black', alpha=0.5, linewidth=1.5)
        axes[0].scatter(test_prices[predictions == 2].index, test_prices[predictions == 2], marker='^', color='green', label='Predict: Up', s=80, zorder=5)
        axes[0].scatter(test_prices[predictions == 1].index, test_prices[predictions == 1], marker='_', color='orange', label='Predict: Hold', s=80, zorder=5)
        axes[0].scatter(test_prices[predictions == 0].index, test_prices[predictions == 0], marker='v', color='red', label='Predict: Down', s=80, zorder=5)
        axes[0].set_title(f"[{ticker}] Price with Global Model Predictions")
        axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)

        axes[1].plot(cumulative_strategy_returns.index, cumulative_strategy_returns, label='Model Strategy', color='royalblue', linewidth=2)
        axes[1].plot(cumulative_buy_and_hold_returns.index, cumulative_buy_and_hold_returns, label='Buy & Hold Benchmark', color='grey', linestyle='--', linewidth=2)
        axes[1].set_title(f"[{ticker}] Backtesting Results")
        axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show() # 창을 닫으면 다음 종목 차트가 열립니다.

if __name__ == "__main__":
    # 여러 종목을 리스트에 넣고 한 번에 학습시킵니다.
    target_tickers = ['005930.KS', '000660.KS', '035420.KS'] # 삼성전자, SK하이닉스, 네이버
    
    # 처음 실행할 때는 use_saved_model=False 로 두고 학습 및 저장합니다.
    # 한 번 저장(global_stock_model.pkl 파일 생성)된 이후에는 
    # use_saved_model=True 로 변경하면 1초 만에 바로 평가/백테스트로 넘어갑니다!
    run_multi_stock_model(target_tickers, use_saved_model=False)