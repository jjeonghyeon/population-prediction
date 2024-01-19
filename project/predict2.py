import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# Matplotlib에서 한국어 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한국어 폰트의 경로로 수정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

def prediction_model(df, target_column='total'):
    x = df.index.astype(int).values.reshape(-1, 1)
    y = df[target_column].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model

def predict_future_population(model, df, future_years, target_column):
    last_index = df.index.astype(int).max()
    future_index = np.array(range(last_index + 1, last_index + 1 + future_years)).reshape(-1, 1)
    future_population = model.predict(future_index)

    # 결과를 현재 데이터프레임과 일치하는 길이로 조정
    index_to_add = pd.RangeIndex(start=last_index + 1, stop=last_index + 1 + future_years, step=1)
    future_df = pd.DataFrame(index=index_to_add, columns=df.columns)
    future_df[target_column] = future_population.flatten()

    return future_df

def plot_population(df, future_df, target_year, target_age_group):
    plt.figure(figsize=(10, 7))
    plt.title(f"{target_age_group}의 인구 예측")
    plt.xlabel("년도")
    plt.ylabel("인구수")

    # 실제 인구를 나타내는 선 그리기
    plt.plot(df.index, df[target_age_group], label='실제 인구', color='blue', marker='o')

    # 예측된 인구를 나타내는 선 그리기
    plt.plot(future_df.index, future_df[target_age_group], label='예측된 인구', color='red', marker='o')

    plt.legend()
    plt.show()

def main():
    # 데이터 불러오기
    df = pd.read_csv('인구수.csv', index_col='yearmonth')
    df.index = df.index.astype(int)

    # 데이터 확인
    print(df.head())

    # 특정 년도와 나이대의 인구수 예측
    target_year = int(input("예측하고자 하는 날짜를 입력하세요 (예: 205001): "))
    target_age_group = input("원하는 나이대를 입력하세요 (예: man1-19): ")

    # 해당 나이대를 예측하는 모델 학습
    model = prediction_model(df, target_column=target_age_group)

    # 미래 년도 예측
    future_df = predict_future_population(model, df, target_year - df.index.min() + 1, target_column=target_age_group)

    # 결과 출력
    print(f"\n예상 {target_year}년 서울 {target_age_group} 평균 인구수: {int(future_df[target_age_group].iloc[-1]):,d}")

    # 시각화
    plot_population(df, future_df, target_year, target_age_group)

if __name__ == "__main__":
    main()
