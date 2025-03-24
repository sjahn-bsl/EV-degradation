import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 파일 경로 설정
file_paths = {
    "40% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 2hr notemp 40%.csv",
    "30% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 2hr notemp 30%.csv",
    "20% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 2hr notemp 20%.csv"
}

# 저장할 폴더 경로
save_folder = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC 2hr notemp Q_current histogram"
os.makedirs(save_folder, exist_ok=True)  # 폴더가 없으면 생성

# 개별 히스토그램 생성 및 저장
for label, path in file_paths.items():
    df = pd.read_csv(path)

    if "Q_current (Ah)" in df.columns:
        trip_count = len(df)  # 전체 trip 개수 계산

        # Q_current (Ah) 값이 50 이상 58 이하인 Trip 개수 및 비율 계산
        Q_current_50_58_count = ((df["Q_current (Ah)"] >= 50) & (df["Q_current (Ah)"] <= 58)).sum()
        Q_current_50_58_ratio = Q_current_50_58_count / trip_count * 100  # 백분율 계산

        plt.figure(figsize=(20, 10))
        plt.hist(df["Q_current (Ah)"], bins='auto', alpha=0.7, color='blue', edgecolor='black')

        # x축 눈금 간격 설정 (10 단위)
        x_min, x_max = int(df["Q_current (Ah)"].min()), int(df["Q_current (Ah)"].max())
        plt.xticks(np.arange(x_min, x_max + 1, 10), rotation=90, ha='right')

        plt.xlabel("Q_current (Ah)")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Q_current (Ah) - {label}")
        plt.grid(axis='y', linestyle="--", alpha=0.7)  # y축만 grid 추가하여 가독성 향상

        # Trip 개수를 오른쪽 위에 표시
        text_x = x_max - (x_max - x_min) * 0.2  # x 위치 (오른쪽)
        text_y = plt.ylim()[1] * 0.9  # y 위치 (상단)

        plt.text(text_x, text_y, f"Trip Files: {trip_count}", fontsize=12, color="red", weight="bold")
        plt.text(text_x, text_y - plt.ylim()[1] * 0.05, f"Q_current (Ah) 50-58%: {Q_current_50_58_count}", fontsize=12, color="red", weight="bold")
        plt.text(text_x, text_y - plt.ylim()[1] * 0.1, f"Ratio: {Q_current_50_58_ratio:.2f}%", fontsize=12, color="red", weight="bold")

        # 저장할 파일 경로 설정
        save_path = os.path.join(save_folder, f"Q_current (Ah)_histogram_{label.split('%')[0]}.png")

        # 그래프 저장
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # 그래프 출력
        plt.show()

print(f"\n 히스토그램 저장 완료! '{save_folder}' 폴더에 저장되었습니다.")
