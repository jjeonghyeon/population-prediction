import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


def prediction_model(df, target_column='total'):
    x = df.index.astype(int).values.reshape(-1, 1)
    y = df[target_column].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model


def predict_future_population(model, df, future_years):
    last_index = df.index.astype(int).max()
    future_index = np.array(range(last_index + 1, last_index + 1 + future_years)).reshape(-1, 1)
    future_population = model.predict(future_index)
    return future_population


def main():
    # 데이터 불러오기
    df = pd.read_csv('인구수.csv', index_col='yearmonth')

    # yearmonth를 정수로 변환
    df.index = df.index.astype(int)

    # 데이터 확인
    print(df.head())

    # 특정 년도의 인구수 예측
    target_year = int(input("예측하고자 하는 날짜를 입력하세요: "))

    # 사용자가 원하는 나이대 입력
    target_age_group = input("원하는 나이대를 입력하세요 (예: man30-39): ")

    # 원하는 나이대에 해당하는 결과만을 출력
    if target_age_group in df.columns:
        print(f"\n{target_age_group} 예측 결과:")

        # 해당 나이대를 예측하는 모델 학습
        model = prediction_model(df, target_column=target_age_group)

        # 미래 년도 예측
        future_population = predict_future_population(model, df, target_year - df.index.min() + 1)

        # 결과 출력
        print(f"예상 {target_year}년 서울 {target_age_group} 평균 인구수: {int(future_population[-1]):,d}")
    else:
        print(f"\n{target_age_group}에 해당하는 나이대가 데이터에 없습니다.")


if __name__ == "__main__":
    main()
