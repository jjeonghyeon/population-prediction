import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('인구수.csv', index_col='yearmonth')

# 'man'과 'woman' 카테고리 합계 계산
df['man_average_total'] = df[['man1-19', 'man20-29', 'man30-39', 'man40-49', 'man50-59', 'man60']].sum(axis=1)
df['woman_average_total'] = df[['woman1-19', 'woman20-29', 'woman30-39', 'woman40-49', 'woman50-59', 'woman60']].sum(axis=1)

# 파이 차트 생성
labels = ['Man', 'Woman']
sizes = [df['man_average_total'].sum(), df['woman_average_total'].sum()]
colors = ['lightblue', 'lightcoral']
explode = (0.1, 0)  # 조각을 떼어내기 위한 설정

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # 원을 원형으로 유지

plt.title('seoul average population (Man vs Woman)')
plt.show()
