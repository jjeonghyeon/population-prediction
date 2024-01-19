import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# Matplotlib에서 한국어 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
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


def plot_pie_chart(total_actual_population, economic_population, category):
    labels = ['총 인구수', '경제활동 인구수']
    sizes = [total_actual_population, economic_population]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    # 아래 두 줄을 추가하여 x와 y축을 제거합니다.
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_axis_off()

    plt.title(f'{category} 인구 비율')
    plt.show()



def main():
    # 데이터 불러오기
    df_total = pd.read_csv('인구수.csv', index_col='yearmonth')
    df_economic = pd.read_csv('경제활동.csv', index_col='yearmonth')

    # yearmonth를 정수로 변환
    df_total.index = df_total.index.astype(int)
    df_economic.index = df_economic.index.astype(int)

    # 특정 년도의 인구수 예측
    target_year = int(input("예측하고자 하는 날짜를 입력하세요: "))

    # 전체 총 인구수 예측
    total_model = prediction_model(df_total)
    total_predicted_population = predict_future_population(total_model, df_total, target_year - df_total.index.min() + 1)

    # 각각의 카테고리에 대한 모델 학습 및 예측
    results_economic = {}
    for category in df_economic.columns[1:]:
        print(f"\n{category} 예측 결과:")

        # 해당 카테고리를 예측하는 모델 학습
        model = prediction_model(df_economic, target_column=category)

        # 미래 년도 예측
        population_economic = predict_future_population(model, df_economic, target_year - df_economic.index.min() + 1)

        # 결과 출력
        print(f"예상 {target_year} 서울 {category} 평균 경제활동 인구수: {int(population_economic[-1]):,d}")

        # 결과 저장
        results_economic[category] = population_economic

        # 비율 파이 그래프 시각화
        plot_pie_chart(total_predicted_population[-1], population_economic[-1], category)


if __name__ == "__main__":
    main()
