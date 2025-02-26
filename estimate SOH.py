import pandas as pd
import numpy as np

# CSV 파일 경로 (사용자가 지정한 파일명)
csv_file_path = r"C:\Users\6211s\SynologyDrive\SamsungSTF\Processed_Data\TripByTrip\bms_01241228076-2023-02-trip-18.csv"

# CSV 파일 불러오기
df = pd.read_csv(csv_file_path)

# 'time' 컬럼을 datetime 형태로 변환
df['time'] = pd.to_datetime(df['time'])

# SOC 값을 0~1 범위로 변환 (CSV 데이터가 0~100 범위를 가지므로 0.01을 곱함)
df['soc'] = df['soc'] * 0.01

# 'pack_current'를 이용하여 'cell_current' 계산 (2P 구조이므로 pack_current를 2로 나눔)
df['cell_current'] = df['pack_current'] / 2

# SOC 변화량 계산
delta_soc = df['soc'].iloc[-1] - df['soc'].iloc[0]

# SOC 변화량이 30% (0.3) 이상인 경우만 처리
if abs(delta_soc) >= 0.3:
    # 시간 간격 계산 (초 단위)
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['time_diff'] = df['time_diff'].fillna(0)  # inplace 제거 (FutureWarning 해결)

    # Q_current 계산 (사다리꼴 적분 방식 사용)
    df['integrand'] = df['cell_current'] * df['time_diff'] / 3600  # Ah 단위로 변환
    Q_current = abs(df['integrand'].sum()) / abs(delta_soc)  # SOC 변화량의 절댓값 적용

    # 초기 배터리 용량 (EV6의 pack Q_initial = 120.6 Ah 사용)
    Q_initial = 120.6 / 2  # 2P 구조 고려

    # SOH 계산
    SOH = (Q_current / Q_initial) * 100

    result = {
        "delta_SOC": delta_soc,
        "Q_current (Ah)": Q_current,
        "Q_initial (Ah)": Q_initial,
        "SOH (%)": SOH
    }

    # 결과 출력 (PyCharm 콘솔)
    print("\n==== SOH Calculation Result ====")
    for key, value in result.items():
        print(f"{key}: {value:.4f}")

else:
    print("\n[SYSTEM] SOC 변화량이 30% 미만이므로 SOH 계산을 수행하지 않음.")
