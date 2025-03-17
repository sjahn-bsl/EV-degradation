import os
import pandas as pd

# 파일 경로 설정
folder_path = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306"
file_names = [
    "BMS_SOC SOH results2 40%.csv",
    "BMS_SOC SOH results2 30%.csv",
    "BMS_SOC SOH results2 20%.csv"
]

# 음수 값 확인
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)

    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    # Q_current 열이 존재하는지 확인
    if "Q_current (Ah)" in df.columns:
        negative_values = df[df["Q_current (Ah)"] < 0]

        if not negative_values.empty:
            print(f"파일 '{file_name}'에 음수 Q_current 값이 {len(negative_values)}개 존재합니다.")
        else:
            print(f"파일 '{file_name}'에는 음수 Q_current 값이 없습니다.")
    else:
        print(f"파일 '{file_name}'에서 'Q_current' 열을 찾을 수 없습니다.")