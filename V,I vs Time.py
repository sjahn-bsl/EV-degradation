import pandas as pd
import matplotlib.pyplot as plt

# 파일 경로
file_path = r"D:\SamsungSTF\Processed_Data\TripByTrip_soc\EV6\bms_01241225206-2023-10-trip-2.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path)

# time 열을 datetime 형식으로 변환
df['time'] = pd.to_datetime(df['time'])

# 그래프 설정
plt.figure(figsize=(12, 10))

# (1) Pack Current vs Time 그래프
plt.subplot(2, 1, 1)  # 2행 1열 중 첫 번째 그래프
plt.plot(df['time'], df['pack_current'], label='Pack Current (A)', color='b')
plt.xlabel('Time')
plt.ylabel('Pack Current (A)')
plt.title('Pack Current vs. Time')
plt.xticks(rotation=45)  # x축 눈금 회전
plt.grid(True)
plt.legend()

# (2) Pack Voltage vs Time 그래프
plt.subplot(2, 1, 2)  # 2행 1열 중 두 번째 그래프
plt.plot(df['time'], df['pack_volt'], label='Pack Voltage (V)', color='r')
plt.xlabel('Time')
plt.ylabel('Pack Voltage (V)')
plt.title('Pack Voltage vs. Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# 그래프 출력
plt.tight_layout()  # 레이아웃 조정
plt.show()
