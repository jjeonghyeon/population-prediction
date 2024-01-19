import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
    return future_population, future_index.flatten()

def plot_scatter(ax, df, target_year, category, future_population, future_index):
    ax.scatter(df.index, df[category], label='Actual Population', color='blue')
    ax.scatter(future_index, future_population, label=f'Predicted Population in {target_year}', color='red')
    ax.set_title(f'{category} 인구의 시간에 따른 산점도')
    ax.set_xlabel('날짜')
    ax.set_ylabel(f'{category} 평균 인구수')
    ax.legend()

def main():
    # 데이터 불러오기
    df = pd.read_csv('인구수.csv', index_col='yearmonth')

    # yearmonth를 정수로 변환
    df.index = df.index.astype(int)

    # 데이터 확인
    print(df.head())

    # 특정 년도의 인구수 예측
    target_year = int(input("예측하고자 하는 날짜를 입력하세요: "))

    # 각각의 카테고리에 대한 모델 학습 및 예측
    fig, axs = plt.subplots(4, 3, figsize=(15, 12))

    for i, category in enumerate(df.columns[1:]):
        print(f"\n{category} 예측 결과:")

        # 해당 카테고리를 예측하는 모델 학습
        model = prediction_model(df, target_column=category)

        # 미래 년도 예측
        future_population, future_index = predict_future_population(model, df, target_year - df.index.min() + 1)

        # 산점도 시각화
        plot_scatter(axs[i // 3, i % 3], df, target_year, category, future_population, future_index)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
