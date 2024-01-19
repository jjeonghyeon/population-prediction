import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# Matplotlib에서 한국어 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한국어 폰트의 경로로 수정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 스케일러 및 다항 특성 전역 선언
scaler = StandardScaler()
poly = PolynomialFeatures(degree=1)

def prediction_model(df, target_column='total'):
    x = df.index.astype(int).values.reshape(-1, 1)
    y = df[target_column].values.reshape(-1, 1)

    # 데이터 스케일링 (표준화)
    x_scaled = scaler.fit_transform(x)

    # 다항 특성 추가
    x_poly = poly.fit_transform(x_scaled)

    model = LinearRegression().fit(x_poly, y)
    return model

def predict_future_population(model, df, future_years):
    last_index = df.index.astype(int).max()
    future_index = np.array(range(last_index + 1, last_index + 1 + future_years)).reshape(-1, 1)

    # 데이터 스케일링 적용
    future_index_scaled = scaler.transform(future_index)

    # 다항 특성 추가
    future_index_poly = poly.transform(future_index_scaled)

    future_population = model.predict(future_index_poly)
    return future_population

def main():
    # 데이터 불러오기
    df = pd.read_csv('인구수.csv', index_col='yearmonth')

    # yearmonth를 정수로 변환
    df.index = df.index.astype(int)

    # 특정 년도의 인구수 예측
    target_year = int(input("예측하고자 하는 날짜를 입력하세요: "))

    # 각각의 카테고리에 대한 모델 학습 및 예측
    for category in df.columns[1:]:
        print(f"\n{category} 예측 결과:")

        # 해당 카테고리를 예측하는 모델 학습
        model = prediction_model(df, target_column=category)

        # 미래 년도 예측
        future_population = predict_future_population(model, df, target_year - df.index.min() + 1)

        # 결과 출력
        print(f"예상 {target_year} 서울 {category} 평균 인구수: {int(future_population[-1]):,d}")

        # 산점도 그리기
        future_index = range(df.index.max() + 1, df.index.max() + 1 + len(future_population))
        plt.scatter(df.index, df[category], label='실제 데이터', color='blue')
        plt.scatter(future_index, future_population, label='예측 데이터', color='red')
        plt.xlabel('Year-Month')
        plt.ylabel('Population')
        plt.title(f'Population Prediction for {category}')
        plt.legend()
        plt.show()

    # 전체 총 인구수 예측
    total_model = prediction_model(df)
    total_future_population = predict_future_population(total_model, df, target_year - df.index.min() + 1)
    print(f"\n예상 {target_year} 서울 평균 인구수: {int(total_future_population[-1]):,d}")

if __name__ == "__main__":
    main()
