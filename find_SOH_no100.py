import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 데이터 폴더 경로
data_folder = r"D:\SamsungSTF\Processed_Data\Merged"

# 폴더가 존재하는지 확인
if not os.path.exists(data_folder):
    print(f"경로를 찾을 수 없습니다: {data_folder}")
    exit()

# 모든 하위 폴더에서 CSV 파일 검색
csv_files = []
for root, _, files in os.walk(data_folder):  # 하위 폴더까지 탐색
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))  # 전체 경로 저장

# CSV 파일이 없는 경우 처리
if not csv_files:
    print(f"폴더에 CSV 파일이 없습니다: {data_folder}")
    exit()

# 개별 파일 처리 함수
def process_file(file_path):
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path, memory_map=True)

        # 🔹 열 이름 대문자로 변환 ('soh' -> 'SOH')
        df.columns = df.columns.str.strip().str.upper()

        # 🔹 "SOH" 열이 존재하는지 확인
        if 'SOH' in df.columns:
            df['SOH'] = pd.to_numeric(df['SOH'], errors='coerce')  # 문자열 -> 숫자로 변환
            df = df.dropna(subset=['SOH'])  # NaN 값 제거

            # 🔹 소수점 5자리까지 반올림 후 비교
            if (df['SOH'].round(5) != 100).any():
                return file_path
    except Exception as e:
        print(f"파일 {file_path}을 읽는 중 오류 발생: {e}")
    return None


# 병렬 처리로 실행
if __name__ == "__main__":
    files_with_non_100_soh = []
    with ProcessPoolExecutor() as executor:
        with tqdm(total=len(csv_files), desc="Processing CSV files", unit="file") as pbar:
            future_to_file = {executor.submit(process_file, file): file for file in csv_files}

            for future in as_completed(future_to_file):
                result = future.result()
                if result is not None:
                    files_with_non_100_soh.append(result)
                    print(f"soh 값이 100이 아닌 데이터가 포함된 파일: {result}")

                pbar.update(1)

    # 결과 출력
    if files_with_non_100_soh:
        print("\nsoh 값이 100이 아닌 파일 목록:")
        for file in files_with_non_100_soh:
            print(file)

        # 🔹 CSV 파일로 저장
        df_result = pd.DataFrame(files_with_non_100_soh, columns=['File_Path'])
        df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSOH 값이 100이 아닌 파일 목록이 저장됨: {output_file}")

    else:
        print("\nsoh가 100이 아닌 cell값을 찾지 못함.")
