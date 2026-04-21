import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 1. 데이터 수집 (삼성전자, 2020년 ~ 현재)
# 삼성전자 종목 코드는 '005930.KS' 입니다.
samsung_df = yf.download('005930.KS', start='2020-01-01')

# yfinance 라이브러리 버전에 따라 단일 종목을 다운로드할 때도 컬럼이 멀티인덱스(MultiIndex)로 생성될 수 있습니다.
# 이 경우 ['Close'] 등으로 컬럼을 선택하면 시리즈(Series)가 아닌 데이터프레임(DataFrame)이 반환되어 이후 연산에서 오류가 발생합니다.
# 컬럼이 멀티인덱스인 경우, 상위 레벨의 인덱스만 사용하도록 하여 단일 인덱스로 변경해줍니다.
if isinstance(samsung_df.columns, pd.MultiIndex):
    samsung_df.columns = samsung_df.columns.get_level_values(0)

# 2. 데이터 전처리 및 피처 엔지니어링 (차트 분석 지표 추가)

# 2-1. 이동평균선(Moving Average) 계산
# 5일, 20일 이동평균선은 단기, 중기 추세를 파악하는데 많이 사용됩니다.
samsung_df['MA5'] = samsung_df['Close'].rolling(window=5).mean()
samsung_df['MA20'] = samsung_df['Close'].rolling(window=20).mean()

# 머신러닝 모델은 주가의 절대값보다 비율(추세)이나 오실레이터 값을 학습하는 것이 더 유리합니다.
samsung_df['MA5_Ratio'] = samsung_df['Close'] / samsung_df['MA5']
samsung_df['MA20_Ratio'] = samsung_df['Close'] / samsung_df['MA20']

# 2-1-2. MACD (Moving Average Convergence Divergence) 계산
# 단기 지수이동평균(12일)과 장기 지수이동평균(26일)의 차이를 이용하여 추세 전환점을 찾습니다.
exp1 = samsung_df['Close'].ewm(span=12, adjust=False).mean()
exp2 = samsung_df['Close'].ewm(span=26, adjust=False).mean()
samsung_df['MACD'] = exp1 - exp2
samsung_df['MACD_Signal'] = samsung_df['MACD'].ewm(span=9, adjust=False).mean()

# 2-1-3. 볼린저 밴드 (Bollinger Bands) 및 %B 계산
# 주가의 변동성과 현재 주가가 밴드 내 어느 위치에 있는지 파악합니다.
samsung_df['20STD'] = samsung_df['Close'].rolling(window=20).std()
samsung_df['Upper_Band'] = samsung_df['MA20'] + (samsung_df['20STD'] * 2)
samsung_df['Lower_Band'] = samsung_df['MA20'] - (samsung_df['20STD'] * 2)
samsung_df['Percent_B'] = (samsung_df['Close'] - samsung_df['Lower_Band']) / (samsung_df['Upper_Band'] - samsung_df['Lower_Band'])

# 2-1-4. 거래량(Volume) 비율
# 차트 분석에서 중요한 거래량의 단기 변화 추세를 확인합니다.
samsung_df['Volume_Ratio'] = samsung_df['Volume'] / samsung_df['Volume'].rolling(window=5).mean()

# 2-2. RSI(Relative Strength Index) 계산
# RSI는 주가의 상승압력과 하락압력 간의 상대적인 강도를 나타냅니다. (과매수/과매도 판단)
delta = samsung_df['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
samsung_df['RSI'] = 100 - (100 / (1 + rs))

# 2-3. 결측치(NaN) 제거
# 이동평균, RSI 계산 초반에는 값이 없으므로 해당 행들을 제거합니다.
samsung_df = samsung_df.dropna()

# 3. 학습 데이터 준비

# 3-1. Feature(X)와 Target(y) 정의
# Feature: 모델이 학습할 데이터 (이동평균선, RSI 등)
# Target: 모델이 맞춰야 할 정답 (다음 날 주가 상승 or 하락)
features = ['MA5_Ratio', 'MA20_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'Percent_B', 'Volume_Ratio']
X = samsung_df[features]

# Target: 다음 날 종가가 오늘 종가보다 높으면 1(상승), 아니면 0(하락)
y = np.where(samsung_df['Close'].shift(-1) > samsung_df['Close'], 1, 0)

# 마지막 날의 y값은 다음 날 데이터가 없으므로 X, y에서 모두 제거
X = X[:-1]
y = y[:-1]


# 3-2. 데이터를 학습용과 테스트용으로 분리
# 주식 데이터는 시간 순서가 중요하므로, shuffle=False 옵션을 주어 순서를 유지합니다.
# 보통 과거 데이터(80%)로 학습하고, 최신 데이터(20%)로 예측 성능을 테스트합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)


# 4. 머신러닝 모델 학습 및 평가

# 4-1. 모델 생성 및 학습
# RandomForest는 여러 개의 결정 트리를 합쳐 예측 성능을 높이는 앙상블 모델입니다.
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# XGBoost는 Gradient Boosting 기반의 고성능 앙상블 모델입니다.
model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 4-2. 예측 수행
predictions = model.predict(X_test)

# 4-3. 모델 성능 평가
# accuracy_score는 전체 예측 중 얼마나 맞췄는지를 나타냅니다.
accuracy = accuracy_score(y_test, predictions)
print(f"모델 예측 정확도: {accuracy:.4f}")
print("\n[상세 예측 결과]")
print(classification_report(y_test, predictions, target_names=['하락(0)', '상승(1)']))


# 5. 시각화: 주가 및 이동평균선 차트
plt.figure(figsize=(14, 7))
plt.plot(samsung_df.index, samsung_df['Close'], label='Close Price', color='black', alpha=0.6)
plt.plot(samsung_df.index, samsung_df['Upper_Band'], label='Upper Band', color='red', linestyle='dashed', alpha=0.4)
plt.plot(samsung_df.index, samsung_df['Lower_Band'], label='Lower Band', color='blue', linestyle='dashed', alpha=0.4)
plt.title('Samsung Electronics Stock Price with Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.legend()
plt.grid(True)
plt.show()
