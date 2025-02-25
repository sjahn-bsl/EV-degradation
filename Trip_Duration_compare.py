import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # 진행 상태 표시

# 폴더 경로 설정
folder1 = r"D:\SamsungSTF\Processed_Data\TripByTrip_soc"
folder2 = r"D:\SamsungSTF\Processed_Data\TripByTrip_soc_2hr"
save_path = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224"

# 결과 저장할 폴더가 없으면 생성
if not os.path.exists(save_path):
    os.makedirs(save_path)


# 단일 파일을 처리하는 함수
def process_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            duration = (df['time'].max() - df['time'].min()).total_seconds()
            return duration
    except Exception as e:
        print(f" Error processing {file_path}: {e}")
    return None


# 모든 하위 폴더에서 CSV 파일 검색 및 병렬 처리
def get_trip_durations(folder):
    csv_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    # CSV 파일이 없는 경우
    if not csv_files:
        print(f" No CSV files found in {folder}")
        return []

    trip_durations = []

    with ProcessPoolExecutor() as executor, tqdm(total=len(csv_files), desc=f"Processing {folder}",
                                                 unit="file") as pbar:
        futures = {executor.submit(process_file, file): file for file in csv_files}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                trip_durations.append(result)
            pbar.update(1)  # tqdm 업데이트

    return trip_durations


# 메인 실행부
if __name__ == "__main__":
    # 병렬 실행
    durations1 = get_trip_durations(folder1)
    durations2 = get_trip_durations(folder2)

    # 평균 지속 시간 계산
    avg_duration1 = sum(durations1) / len(durations1) if durations1 else 0
    avg_duration2 = sum(durations2) / len(durations2) if durations2 else 0

    # 결과 출력
    print(f"Original Avg Duration: {avg_duration1} sec")
    print(f"2-hour gap Avg Duration: {avg_duration2} sec")

    # 결과를 txt 파일로 저장
    result_text = f"""Trip Duration Comparison
----------------------------------
Original Avg Duration: {avg_duration1} sec
2-hour gap Avg Duration: {avg_duration2} sec
----------------------------------
Interpretation:
- If Avg Duration increases -> Trip expansion is effective
- If Avg Duration decreases or is the same -> New criteria may be too strict
"""

    result_file = os.path.join(save_path, "trip_duration_comparison.txt")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f" Results saved to: {result_file}")
