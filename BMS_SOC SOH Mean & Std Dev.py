import pandas as pd

# 파일 경로 설정
file_paths = {
    "20% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224\BMS_SOC SOH 20%_results.csv",
    "30% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224\BMS_SOC SOH 30%_results.csv",
    "40% SOC Change": r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224\BMS_SOC SOH 40%_results.csv"
}

# 평균과 표준편차 저장용 딕셔너리
stats = {}

# 각 SOC 변화량에 대해 SOH 평균 및 표준편차 계산
for label, path in file_paths.items():
    df = pd.read_csv(path)

    if "SOH (%)" in df.columns:
        mean_value = df["SOH (%)"].mean()
        std_dev_value = df["SOH (%)"].std()
        stats[label] = {"Mean": mean_value, "Std Dev": std_dev_value}

# 평균과 표준편차 출력
for soc_change, values in stats.items():
    print(f"{soc_change}: SOH 평균 = {values['Mean']:.2f}, 표준편차 = {values['Std Dev']:.2f}")
