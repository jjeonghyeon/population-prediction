import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# Matplotlib에서 한국어 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한국어 폰트의 경로로 수정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 인구수 데이터 불러오기
population_data = pd.read_csv('인구수.csv', encoding='utf-8')

# 경제활동 데이터 불러오기
economic_activity_data = pd.read_csv('경제활동.csv', encoding='utf-8')

# 두 데이터셋을 'yearmonth'를 기준으로 병합
merged_data = pd.merge(population_data, economic_activity_data, on='yearmonth')

# 남성과 여성 비율을 파이 차트로 시각화
labels = ['남성', '여성']
sizes = [merged_data['man'].mean(), merged_data['woman'].mean()]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('남성과 여성 경제활동 비율')
plt.show()
