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


def plot_population(df, results, target_year):
    plt.figure(figsize=(10, 6))

    # 실제 데이터
    plt.plot(df.index, df.iloc[:, 1:], marker='o', linestyle='-', label='실제')

    # 예측 데이터
    for category, future_population in results.items():
        future_index = range(df.index.max() + 1, df.index.max() + 1 + len(future_population))
        plt.plot(future_index, future_population, marker='o', linestyle='-', label=f'예측 {category}')

    plt.title(f"경제활동 예측 결과 ({target_year}년)")
    plt.xlabel("년도")
    plt.ylabel("경제활동 인구수")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # 데이터 불러오기
    df = pd.read_csv('경제활동.csv', index_col='yearmonth')

    # yearmonth를 정수로 변환
    df.index = df.index.astype(int)

    # 특정 년도의 인구수 예측
    target_year = int(input("예측하고자 하는 날짜를 입력하세요: "))

    # 각각의 카테고리에 대한 모델 학습 및 예측
    results = {}
    for category in df.columns[1:]:
        print(f"\n{category} 예측 결과:")

        # 해당 카테고리를 예측하는 모델 학습
        model = prediction_model(df, target_column=category)

        # 미래 년도 예측
        future_population = predict_future_population(model, df, target_year - df.index.min() + 1)

        # 결과 출력
        print(f"예상 {target_year} 서울 {category} 평균 경제활동 인구수: {int(future_population[-1]):,d}")

        # 결과 저장
        results[category] = future_population

    # 전체 총 인구수 예측
    total_model = prediction_model(df)
    total_future_population = predict_future_population(total_model, df, target_year - df.index.min() + 1)
    print(f"\n예상 {target_year} 서울 평균 경제활동 인구수: {int(total_future_population[-1]):,d}")

    # 그래프 그리기
    plot_population(df, results, target_year)


if __name__ == "__main__":
    main()
