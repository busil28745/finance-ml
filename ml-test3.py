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

# --- 설정값 ---
TICKER = '005930.KS'
# 10분 단위 데이터는 yfinance에서 직접 지원하지 않으므로 5분(5m) 데이터를 받아 변환합니다.
START_DATE = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
INTERVALS_IN_TRADING_DAY = 39 # 한국 증시는 6.5시간이므로, 10분 간격 캔들은 하루에 약 39개 생성됩니다.
HOLD_THRESHOLD = 0.0010 # 0.10% 이내 변동은 '보합'으로 간주 (기준 완화)

# 1. 데이터 수집 (삼성전자, 5분별 데이터 다운로드 후 10분으로 변환)
print(f"'{TICKER}'의 5분 단위 데이터를 다운로드하여 10분 단위로 변환합니다... (시작일: {START_DATE})")
samsung_df = yf.download(TICKER, start=START_DATE, interval='5m')

if samsung_df.empty:
    print("데이터를 다운로드하지 못했습니다. 종목 코드나 기간을 확인해주세요.")
    exit()

# yfinance MultiIndex 컬럼 처리
if isinstance(samsung_df.columns, pd.MultiIndex):
    samsung_df.columns = samsung_df.columns.get_level_values(0)

# 10분 단위로 리샘플링 (Resampling)
# 5분 데이터를 10분 단위로 묶어서 시가, 고가, 저가, 종가, 거래량 캔들을 새로 생성합니다.
samsung_df = samsung_df.resample('10min').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# 2. 데이터 전처리 및 피처 엔지니어링 (시간 단위 기준)
# 이제 모든 window는 '일'이 아닌 '시간'을 의미합니다. (예: 5시간 이동평균)
samsung_df['MA5'] = samsung_df['Close'].rolling(window=5).mean()
samsung_df['MA20'] = samsung_df['Close'].rolling(window=20).mean()
samsung_df['MA5_Ratio'] = samsung_df['Close'] / samsung_df['MA5']
samsung_df['MA20_Ratio'] = samsung_df['Close'] / samsung_df['MA20']

exp1 = samsung_df['Close'].ewm(span=12, adjust=False).mean()
exp2 = samsung_df['Close'].ewm(span=26, adjust=False).mean()
samsung_df['MACD'] = exp1 - exp2
samsung_df['MACD_Signal'] = samsung_df['MACD'].ewm(span=9, adjust=False).mean()

samsung_df['20STD'] = samsung_df['Close'].rolling(window=20).std()
samsung_df['Upper_Band'] = samsung_df['MA20'] + (samsung_df['20STD'] * 2)
samsung_df['Lower_Band'] = samsung_df['MA20'] - (samsung_df['20STD'] * 2)
samsung_df['Percent_B'] = (samsung_df['Close'] - samsung_df['Lower_Band']) / (samsung_df['Upper_Band'] - samsung_df['Lower_Band'])

samsung_df['Volume_Ratio'] = samsung_df['Volume'] / samsung_df['Volume'].rolling(window=5).mean()

delta = samsung_df['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
samsung_df['RSI'] = 100 - (100 / (1 + rs))

samsung_df['Period_Return'] = samsung_df['Close'].pct_change()

# [추가 지표] 변동성과 모멘텀을 나타내는 강력한 지표 추가
# 1. Stochastic Oscillator (현재 가격이 최근 변동폭 중 어디쯤 있는지)
low_min = samsung_df['Low'].rolling(window=14).min()
high_max = samsung_df['High'].rolling(window=14).max()
samsung_df['Stoch_K'] = 100 * ((samsung_df['Close'] - low_min) / (high_max - low_min))
samsung_df['Stoch_D'] = samsung_df['Stoch_K'].rolling(window=3).mean()

# 2. ATR (Average True Range - 변동성 지표)
high_low = samsung_df['High'] - samsung_df['Low']
high_close = np.abs(samsung_df['High'] - samsung_df['Close'].shift())
low_close = np.abs(samsung_df['Low'] - samsung_df['Close'].shift())
samsung_df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()
samsung_df['ATR_Ratio'] = samsung_df['ATR'] / samsung_df['Close']

# [핵심] 과거 추세 반영 (Lagged Features)
# 1) 단기 단위 Lag: 10, 20, 30분, 1시간, 2시간 전 데이터
# 2) 일 단위 Lag: 1~7일 전 데이터 (1일 = 39 캔들로 근사)
features_to_lag = ['MA5_Ratio', 'RSI', 'MACD', 'Percent_B', 'Period_Return', 'Stoch_K', 'ATR_Ratio']
lag_features_short = []
lag_features_daily = []

for feature in features_to_lag:
    # 1~3구간(10~30분) 전 데이터 추가
    for lag in range(1, 4):
        col_name = f'{feature}_lag_{lag}p' # p for period
        samsung_df[col_name] = samsung_df[feature].shift(lag)
        lag_features_short.append(col_name)
    
    # 1시간(60분), 2시간(120분) 전 데이터 추가
    for lag_hour in [1, 2]:
        lag_period = lag_hour * 6 # 1h = 6*10m, 2h = 12*10m
        col_name = f'{feature}_lag_{lag_hour}h' # h for hour
        samsung_df[col_name] = samsung_df[feature].shift(lag_period)
        lag_features_short.append(col_name)

    # 1~7일 전 데이터 추가
    for lag in range(1, 8):
        col_name = f'{feature}_lag_{lag}d'
        # 1일 전을 39구간 전으로 근사하여 계산
        samsung_df[col_name] = samsung_df[feature].shift(lag * INTERVALS_IN_TRADING_DAY)
        lag_features_daily.append(col_name)

# 초반의 결측치(NaN) 제거
samsung_df = samsung_df.dropna()

# 3. 학습 데이터 준비
base_features = ['MA5_Ratio', 'MA20_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'Percent_B', 'Volume_Ratio', 'Period_Return', 'Stoch_K', 'Stoch_D', 'ATR_Ratio']
X = samsung_df[base_features + lag_features_short + lag_features_daily]

# Target: 다음 '10분'의 종가 등락을 3-class로 분류 (하락:0, 보합:1, 상승:2)
future_return = samsung_df['Close'].pct_change().shift(-1)
conditions = [
    future_return > HOLD_THRESHOLD,
    future_return < -HOLD_THRESHOLD
]
choices = [2, 0] # 상승: 2, 하락: 0
y = np.select(conditions, choices, default=1) # 그 외는 보합: 1
y = pd.Series(y, index=samsung_df.index)

# 마지막 데이터는 다음 10분 데이터가 없으므로 제거
X = X[:-1]
y = y[:-1]

if len(X) < 100:
    print("데이터가 너무 적어 학습을 진행할 수 없습니다. 시작 날짜를 더 과거로 설정해보세요.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# 4. 머신러닝 모델 학습 및 평가 (하이퍼파라미터 튜닝 추가)
print("\nGridSearchCV를 사용하여 최적의 하이퍼파라미터를 탐색합니다...")
print("시간이 다소 소요될 수 있습니다...")

# 탐색할 하이퍼파라미터 그리드 설정 (후보 값들을 지정)
# 너무 많은 값을 넣으면 시간이 매우 오래 걸릴 수 있으므로, 대표적인 값들로 구성합니다.
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300],
    'subsample': [0.8, 1.0],         # 데이터 샘플링 비율 (과적합 방지)
    'colsample_bytree': [0.8, 1.0]   # 피처 샘플링 비율 (과적합 방지)
}

# 기본 XGBoost 모델
xgb = XGBClassifier(random_state=42, eval_metric='logloss')

# GridSearchCV 객체 생성
# 시계열 데이터이므로 미래 데이터가 과거 학습에 사용되는 Data Leakage를 막기 위해 TimeSeriesSplit 사용
tscv = TimeSeriesSplit(n_splits=3)

# n_jobs=-1: 사용 가능한 모든 CPU 코어를 사용하여 병렬로 학습
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=1, scoring='accuracy')

# 그리드 서치 실행
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 그 때의 모델을 가져옵니다.
print(f"\n최적의 하이퍼파라미터: {grid_search.best_params_}")
model = grid_search.best_estimator_

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n모델 예측 정확도 (다음 '10분' 예측): {accuracy:.4f}")
print("\n[상세 예측 결과]")
print(classification_report(y_test, predictions, target_names=['하락(0)', '보합(1)', '상승(2)']))

# 5. 시각화: 어떤 추세 지표가 예측에 가장 큰 영향을 미쳤을까? (Feature Importance)
plt.figure(figsize=(12, 8))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(15).sort_values().plot(kind='barh', color='darkcyan')
plt.title('Top 15 Feature Importances for 10m Trend Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6. 시각화: 실제 주가 차트 위에 예측 결과(매수/매도 신호) 표시
# 모델이 테스트한 기간(X_test)에 해당하는 실제 주가(Close)를 가져옵니다.
test_prices = samsung_df.loc[X_test.index, 'Close']

plt.figure(figsize=(14, 7))
# 실제 주가를 검은색 선으로 표시
plt.plot(test_prices.index, test_prices, label='Close Price', color='black', alpha=0.5, linewidth=1.5)

# 상승(2), 보합(1), 하락(0) 예측 신호 분리
buy_signals = test_prices[predictions == 2]
hold_signals = test_prices[predictions == 1]
sell_signals = test_prices[predictions == 0]

plt.scatter(buy_signals.index, buy_signals, marker='^', color='green', label='Predict: Up (Buy Signal)', s=80, zorder=5)
plt.scatter(hold_signals.index, hold_signals, marker='_', color='orange', label='Predict: Hold Signal', s=80, zorder=5)
plt.scatter(sell_signals.index, sell_signals, marker='v', color='red', label='Predict: Down (Sell Signal)', s=80, zorder=5)

plt.title(f"{TICKER} Price with Model Predictions (Test Period)")
plt.xlabel('Date / Time')
plt.ylabel('Price (KRW)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. 시각화: 가상 매매 수익률(백테스팅) 결과
# 테스트 기간 동안의 실제 수익률을 계산합니다.
# 예측 시점(t)을 기준으로 다음 기간(t+1)의 수익률을 가져와야 하므로 shift(-1)을 사용합니다.
actual_returns = samsung_df.loc[X_test.index, 'Close'].pct_change().shift(-1).fillna(0)

# 모델 전략 수익률: 상승(2)으로 예측했을 때만 해당 기간의 실제 수익률을 얻습니다.
strategy_returns = actual_returns * (predictions == 2)

# 누적 수익률 계산 (수익률을 복리로 계속 더해나감)
cumulative_strategy_returns = (1 + strategy_returns).cumprod()
# 비교 기준(Benchmark): 테스트 기간 동안 단순히 주식을 보유했을 때의 누적 수익률
cumulative_buy_and_hold_returns = (1 + actual_returns).cumprod()

# 최종 수익률 터미널 출력
final_strategy_return = (cumulative_strategy_returns.iloc[-1] - 1) * 100
final_benchmark_return = (cumulative_buy_and_hold_returns.iloc[-1] - 1) * 100
print(f"\n[백테스팅 최종 결과 (Test 기간 기준)]")
print(f"모델 전략 누적 수익률: {final_strategy_return:.2f}%")
print(f"단순 보유(Buy & Hold) 수익률: {final_benchmark_return:.2f}%")

plt.figure(figsize=(14, 7))
plt.plot(cumulative_strategy_returns.index, cumulative_strategy_returns, label='Model Strategy Cumulative Return', color='royalblue', linewidth=2)
plt.plot(cumulative_buy_and_hold_returns.index, cumulative_buy_and_hold_returns, label='Buy and Hold Cumulative Return (Benchmark)', color='grey', linestyle='--', linewidth=2)

plt.title(f"{TICKER} Backtesting Results")
plt.xlabel('Date / Time')
plt.ylabel('Cumulative Returns (1 = 100%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()