import pandas as pd

# 파일 경로 설정
file_paths = {
    "20% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 2hr OCV0 20%.csv",
    "30% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 2hr OCV0 30%.csv",
    "40% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 2hr OCV0 40%.csv"
}

# 평균과 표준편차 저장용 딕셔너리
stats = {}

# 각 SOC 변화량에 대해 Q_current 평균 및 표준편차 계산
for label, path in file_paths.items():
    df = pd.read_csv(path)

    if "Q_current (Ah)" in df.columns:
        mean_value = df["Q_current (Ah)"].mean()
        std_dev_value = df["Q_current (Ah)"].std()
        stats[label] = {"Mean": mean_value, "Std Dev": std_dev_value}

# 평균과 표준편차 출력
for soc_change, values in stats.items():
    print(f"{soc_change}: Q_current 평균 = {values['Mean']:.2f}, 표준편차 = {values['Std Dev']:.2f}")
