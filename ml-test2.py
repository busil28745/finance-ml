import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 1. 데이터 수집 (삼성전자, 2020년 ~ 현재)
samsung_df = yf.download('005930.KS', start='2020-01-01')

if isinstance(samsung_df.columns, pd.MultiIndex):
    samsung_df.columns = samsung_df.columns.get_level_values(0)

# 2. 데이터 전처리 및 피처 엔지니어링
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

# [추가] 일일 수익률 (현재의 가격 모멘텀 파악)
samsung_df['Daily_Return'] = samsung_df['Close'].pct_change()

# [추가] 과거 추세 반영 (Lagged Features)
# 과거 1~3일 동안의 지표 값을 추가하여 단순 현재 시점뿐만 아니라 '추세 흐름'을 학습하게 합니다.
features_to_lag = ['MA5_Ratio', 'RSI', 'MACD', 'Percent_B', 'Daily_Return']
for feature in features_to_lag:
    for lag in range(1, 4):
        samsung_df[f'{feature}_lag_{lag}'] = samsung_df[feature].shift(lag)

# 초반의 결측치(NaN) 제거
samsung_df = samsung_df.dropna()

# 3. 학습 데이터 준비
base_features = ['MA5_Ratio', 'MA20_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'Percent_B', 'Volume_Ratio', 'Daily_Return']
lag_features = [f'{feat}_lag_{lag}' for feat in features_to_lag for lag in range(1, 4)]
X = samsung_df[base_features + lag_features]

# Target: 다음 날 종가가 오늘 종가보다 높으면 1(상승), 아니면 0(하락)
y = np.where(samsung_df['Close'].shift(-1) > samsung_df['Close'], 1, 0)

X = X[:-1]
y = y[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# 4. 머신러닝 모델 학습 및 평가
# 주식 데이터 특성상 과적합(Overfitting)을 방지하기 위해 max_depth와 learning_rate를 조정했습니다.
model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"모델 예측 정확도: {accuracy:.4f}")
print("\n[상세 예측 결과]")
print(classification_report(y_test, predictions, target_names=['하락(0)', '상승(1)']))

# 5. 시각화: 어떤 추세 지표가 예측에 가장 큰 영향을 미쳤을까? (Feature Importance)
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title('Top 10 Feature Importances for Trend Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

