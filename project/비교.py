import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "C:\\Windows\\Fonts\\malgun.ttf"  # 한글 폰트 파일 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


# 그래프 표시
plt.show()

# 데이터 불러오기
population_data = pd.read_csv("인구수.csv")
economic_activity_data = pd.read_csv("경제활동.csv")

# 두 데이터 합치기
merged_data = pd.merge(population_data, economic_activity_data, on='yearmonth')

# Feature와 Target 설정
X = merged_data[['man', 'woman']]
y = merged_data['total']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 모델 평가 (평균 제곱 오차 계산)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 예측 결과 시각화
plt.scatter(y_test, y_pred)
plt.xlabel("실제 인구수")
plt.ylabel("예측 인구수")
plt.title("실제 인구수 vs 예측 인구수")
plt.show()
