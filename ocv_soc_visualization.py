import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. 경로 설정
excel_path = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\NE_Cell_Characterization_performance.xlsx"
csv_path = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 2hr notemp 40%.csv"

# 2. OCV-SOC 곡선 데이터 로드
ocv_soc_df = pd.read_excel(excel_path, sheet_name="SOC-OCV")
ocv_soc_df = ocv_soc_df.iloc[7:108]  # B8:B108, D8:D108

soc_values = ocv_soc_df.iloc[:, 1].astype(float, errors='ignore')  # B열: SOC
ocv_values = ocv_soc_df.iloc[:, 3].astype(float, errors='ignore')  # D열: OCV

# 3. 결과 CSV 파일 로드
results_df = pd.read_csv(csv_path)

# 4. 시각화
plt.figure(figsize=(10, 6))
plt.plot(ocv_values, soc_values, label='OCV-SOC Curve', color='black', linewidth=2)

# 색상 할당을 위한 컬러맵
from matplotlib.cm import get_cmap
cmap = get_cmap("tab20")
device_colors = {}

# 5. Trip 점들 표시
for idx, row in results_df.iterrows():
    device_id = row['device_id']
    color = device_colors.setdefault(device_id, cmap(len(device_colors) % 20))
    plt.scatter([row['OCV0'], row['OCV1']], [row['SOC0']*100, row['SOC1']*100],
                color=color, s=40, label=device_id if device_id not in device_colors else "")

# 중복 없는 범례 생성
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title="Device ID", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel("OCV (V)")
plt.ylabel("SOC (%)")
plt.title("OCV-SOC Curve with Trip Points")
plt.grid(True)
plt.tight_layout()

# 6. 표시
plt.show()
