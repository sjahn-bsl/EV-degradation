import pandas as pd

# 분석할 CSV 파일 목록
file_paths = [
    r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 40%.csv",
    r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 30%.csv",
    r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250306\OCV-SOC SOH results 20%.csv"
]

# 각 파일별로 분석 수행
for file_path in file_paths:
    print(f"\n 파일: {file_path}")

    try:
        # CSV 파일 불러오기
        df = pd.read_csv(file_path, encoding="utf-8-sig")

        # 필요한 열이 있는지 확인
        if 'delta_SOC' in df.columns and 'SOH_OCV (%)' in df.columns:
            # 각 케이스별 조건 정의
            case1 = ((df['delta_SOC'] < 0) & (df['SOH_OCV (%)'] > 0)).sum()
            case2 = ((df['delta_SOC'] < 0) & (df['SOH_OCV (%)'] < 0)).sum()
            case3 = ((df['delta_SOC'] > 0) & (df['SOH_OCV (%)'] < 0)).sum()
            case4 = ((df['delta_SOC'] > 0) & (df['SOH_OCV (%)'] > 0)).sum()

            # 결과 출력
            print(f"1. delta_SOC < 0이고 SOH_OCV (%) > 0인 경우: {case1}개")
            print(f"2. delta_SOC < 0이고 SOH_OCV (%) < 0인 경우: {case2}개")
            print(f"3. delta_SOC > 0이고 SOH_OCV (%) < 0인 경우: {case3}개")
            print(f"4. delta_SOC > 0이고 SOH_OCV (%) > 0인 경우: {case4}개")
        else:
            print("파일에 'delta_SOC' 또는 'SOH_OCV (%)' 열이 없습니다. 열 이름을 확인해주세요.")

    except Exception as e:
        print(f"파일을 처리하는 중 오류 발생: {e}")
