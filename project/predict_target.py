import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
df = pd.read_csv('경제활동.csv')

# 그래프 생성
plt.figure(figsize=(10, 6))

# 남성 그래프
plt.plot(df['yearmonth'], df['man'], marker='o', linestyle='-', label='남성')

# 여성 그래프
plt.plot(df['yearmonth'], df['woman'], marker='o', linestyle='-', label='여성')

# 그래프 속성 설정
plt.title('average economic activity')
plt.xlabel('yearmonth')
plt.ylabel('population')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# 그래프 표시
plt.show()
