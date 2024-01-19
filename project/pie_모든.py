import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# Matplotlib에서 한국어 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한국어 폰트의 경로로 수정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 인구수 데이터 불러오기
population_data = pd.read_csv('인구수.csv', encoding='utf-8')

# 'yearmonth'를 기준으로 데이터를 남성과 여성으로 나누기
man_data = population_data[['yearmonth', 'man1-19', 'man20-29', 'man30-39', 'man40-49', 'man50-59', 'man60']]
woman_data = population_data[['yearmonth', 'woman1-19', 'woman20-29', 'woman30-39', 'woman40-49', 'woman50-59', 'woman60']]

# 각각의 데이터에서 나이대별 합 구하기
man_sum = man_data.sum(axis=1)
woman_sum = woman_data.sum(axis=1)

# 전체 합으로 나누어 비율 계산
man_ratio = man_sum / (man_sum + woman_sum)
woman_ratio = woman_sum / (man_sum + woman_sum)

# 파이 차트 그리기
labels = ['남성', '여성']
sizes = [man_ratio.mean(), woman_ratio.mean()]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('나이대별 남성과 여성 비율')
plt.show()


