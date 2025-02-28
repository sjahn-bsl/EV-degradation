import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 파일 경로 설정
file_paths = {
    "20% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224\BMS_SOC SOH 20%_results.csv",
    "30% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224\BMS_SOC SOH 30%_results.csv",
    "40% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224\BMS_SOC SOH 40%_results.csv"
}

# 저장할 폴더 경로
save_folder = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224\BMS_SOC SOH estimate histogram"
os.makedirs(save_folder, exist_ok=True)  # 폴더가 없으면 생성

# 개별 히스토그램 생성 및 저장
for label, path in file_paths.items():
    df = pd.read_csv(path)

    if "SOH (%)" in df.columns:
        trip_count = len(df)  # 전체 trip 개수 계산

        # SOH 값이 98 이상 100 이하인 Trip 개수 계산
        soh_98_100_count = ((df["SOH (%)"] >= 98) & (df["SOH (%)"] <= 100)).sum()

        plt.figure(figsize=(10, 5))
        plt.hist(df["SOH (%)"], bins=30, alpha=0.7, color='blue', edgecolor='black')

        # x축 눈금 간격 설정 (2 단위)
        x_min, x_max = int(df["SOH (%)"].min()), int(df["SOH (%)"].max())
        plt.xticks(np.arange(x_min, x_max + 1, 2), rotation=45)

        plt.xlabel("SOH (%)")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of SOH - {label}")
        plt.grid(True)

        # Trip 개수를 오른쪽 위에 표시
        text_x = x_max - (x_max - x_min) * 0.2  # x 위치 (오른쪽)
        text_y = plt.ylim()[1] * 0.9  # y 위치 (상단)

        plt.text(text_x, text_y, f"Trip Files: {trip_count}", fontsize=12, color="red", weight="bold")
        plt.text(text_x, text_y - plt.ylim()[1] * 0.05, f"SOH 98-100%: {soh_98_100_count}", fontsize=12, color="red",
                 weight="bold")

        # 저장할 파일 경로 설정
        save_path = os.path.join(save_folder, f"BMS_SOC_SOH_histogram_{label.split('%')[0]}.png")

        # 그래프 저장
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # 그래프 출력
        plt.show()

print(f"\n✅ 히스토그램 저장 완료! '{save_folder}' 폴더에 저장되었습니다.")
