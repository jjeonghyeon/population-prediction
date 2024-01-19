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

def plot_pie_chart(labels, sizes, title):
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()

def main():
    # 데이터 불러오기
    df = pd.read_csv('인구수.csv', index_col='yearmonth')

    # yearmonth를 정수로 변환
    df.index = df.index.astype(int)

    # 'man'과 'woman' 카테고리 합계 계산
    df['man_average_total'] = df[['man1-19', 'man20-29', 'man30-39', 'man40-49', 'man50-59', 'man60']].sum(axis=1)
    df['woman_average_total'] = df[['woman1-19', 'woman20-29', 'woman30-39', 'woman40-49', 'woman50-59', 'woman60']].sum(axis=1)

    # 데이터 확인
    print(df.head())

    # 특정 년도의 인구수 예측
    target_year = int(input("예측하고자 하는 날짜를 입력하세요: "))

    # 결과를 저장할 리스트 초기화
    labels = ['man', 'woman']
    sizes = []

    # 각각의 카테고리에 대한 모델 학습 및 예측
    for category in ['man_average_total', 'woman_average_total']:
        print(f"\n{category} 예측 결과:")

        # 해당 카테고리를 예측하는 모델 학습
        model = prediction_model(df, target_column=category)

        # 미래 년도 예측
        future_population = predict_future_population(model, df, target_year - df.index.min() + 1)

        # 결과 출력
        print(f"예상 {target_year} 서울 {category} 평균 인구수: {int(future_population[-1]):,d}")

        # 결과를 리스트에 추가
        sizes.append(int(future_population[-1]))

    # 파이 그래프로 시각화
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f"예상 {target_year} 서울 평균 인구 구성")
    plt.show()

if __name__ == "__main__":
    main()
