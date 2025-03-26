import os
import pandas as pd
from tqdm import tqdm

# 폴더 경로 설정
folder_path = r"D:\SamsungSTF\Processed_Data\TripByTrip_soc_2hr"

# 카운터 초기화
count_20 = 0
count_30 = 0
count_40 = 0
total_files = 0

# 파일 수집
csv_files = []
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

# 진행률 표시
for file_path in tqdm(csv_files, desc="Processing Files", unit="file"):
    try:
        df = pd.read_csv(file_path)

        soc_initial = df['soc'].iloc[30]
        soc_final = df['soc'].iloc[-31]
        delta_soc = soc_initial - soc_final

        # 변화량 조건별 카운트
        if delta_soc >= 20:
            count_20 += 1
        if delta_soc >= 30:
            count_30 += 1
        if delta_soc >= 40:
            count_40 += 1

        total_files += 1

    except Exception as e:
        print(f"[오류] {file_path}: {e}")

# 결과 출력
print("\n=== SOC 변화량 분석 결과 ===")
print(f"총 파일 수: {total_files}")
print(f"SOC 변화량 ≥ 20: {count_20}개")
print(f"SOC 변화량 ≥ 30: {count_30}개")
print(f"SOC 변화량 ≥ 40: {count_40}개")
