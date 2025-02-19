import os
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 폴더 경로
data_folder = r"D:\SamsungSTF\Processed_Data\Merged"
output_base_folder = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250211\histogram"

# 히스토그램 저장 경로
output_folders = {
    "start_soh": os.path.join(output_base_folder, "start_soh"),
    "final_soh": os.path.join(output_base_folder, "final_soh"),
    "change_soh": os.path.join(output_base_folder, "change_soh")
}

# 출력 폴더가 없으면 생성
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# 차량 종류 리스트 (차량명: 단말기 리스트)
vehicle_dict = {
    'NiroEV': ['01241228149', '01241228151', '01241228153'],
    'Ioniq5': ['01241227999', '01241228003', '01241228005'],
    'Ioniq6': ['01241248713', '01241592904', '01241597763'],
    'KonaEV': ['01241228102', '01241228122', '01241228123'],
    'EV6': ['01241225206', '01241228049', '01241228050'],
    'GV60': ['01241228108', '01241228130', '01241228131'],
    'Porter2EV': ['01241228144', '01241228160', '01241228177'],
    'Bongo3EV': ['01241228162', '01241228179', '01241248642']
}

# 차량별 SOH 값 저장 딕셔너리
first_soh_values = {vehicle: [] for vehicle in vehicle_dict.keys()}
final_soh_values = {vehicle: [] for vehicle in vehicle_dict.keys()}

# 차량별 최초 & 최종 SOH 추출
for vehicle, device_ids in vehicle_dict.items():
    vehicle_path = os.path.join(data_folder, vehicle)

    if not os.path.exists(vehicle_path):
        print(f" 경로 없음: {vehicle_path}")
        continue

    for device_id in device_ids:
        # 해당 단말기 번호에 해당하는 모든 CSV 파일 찾기
        csv_files = [f for f in os.listdir(vehicle_path) if f.startswith(f"bms_{device_id}") or f.startswith(f"bms_altitude_{device_id}")]

        if not csv_files:
            continue

        # 최초 CSV 파일 (가장 오래된 연도-월 찾기)
        first_file = min(csv_files, key=lambda x: x.split('_')[-1].replace('.csv', ''))
        first_file_path = os.path.join(vehicle_path, first_file)

        # 최신 CSV 파일 (가장 최근 연도-월 찾기)
        latest_file = max(csv_files, key=lambda x: x.split('_')[-1].replace('.csv', ''))
        latest_file_path = os.path.join(vehicle_path, latest_file)

        try:
            # 최초 SOH 추출
            df_first = pd.read_csv(first_file_path, memory_map=True)
            df_first.columns = df_first.columns.str.strip().str.upper()  # 컬럼명을 대문자로 변환

            if 'SOH' in df_first.columns:
                df_first['SOH'] = pd.to_numeric(df_first['SOH'], errors='coerce')  # 숫자로 변환
                df_first = df_first.dropna(subset=['SOH'])  # NaN 값 제거

                if not df_first.empty:
                    first_soh_values[vehicle].append(df_first['SOH'].iloc[0])  # 첫 번째 행의 SOH 값 저장

            # 최종 SOH 추출
            df_final = pd.read_csv(latest_file_path, memory_map=True)
            df_final.columns = df_final.columns.str.strip().str.upper()  # 컬럼명을 대문자로 변환

            if 'SOH' in df_final.columns:
                df_final['SOH'] = pd.to_numeric(df_final['SOH'], errors='coerce')  # 숫자로 변환
                df_final = df_final.dropna(subset=['SOH'])  # NaN 값 제거

                if not df_final.empty:
                    final_soh_values[vehicle].append(df_final['SOH'].iloc[-1])  # 마지막 행의 SOH 값 저장

        except Exception as e:
            print(f" 파일 오류 {latest_file_path} 또는 {first_file_path}: {e}")

# SOH 변화량(ΔSOH) 계산
delta_soh_values = {
    vehicle: [first - final for first, final in zip(first_soh_values[vehicle], final_soh_values[vehicle])]
    for vehicle in vehicle_dict.keys()
}

# 히스토그램 생성 및 저장 함수
def save_histogram(data, title, filename, folder):
    if not data:
        print(f" 데이터 없음: {title}")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=10, edgecolor='black', alpha=0.75)
    plt.title(title)
    plt.xlabel("SOH 값")
    plt.ylabel("빈도")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_path = os.path.join(folder, filename)
    plt.savefig(save_path)
    plt.close()  # 그래프 창 닫기

    print(f" 저장 완료: {save_path}")

#  최초 SOH 히스토그램 생성
for vehicle, soh_values in first_soh_values.items():
    save_histogram(soh_values, f"{vehicle} 최초 SOH 분포", f"{vehicle}_최초_SOH_histogram.png", output_folders["start_soh"])

#  최종 SOH 히스토그램 생성
for vehicle, soh_values in final_soh_values.items():
    save_histogram(soh_values, f"{vehicle} 최종 SOH 분포", f"{vehicle}_최종_SOH_histogram.png", output_folders["final_soh"])

#  SOH 변화량(ΔSOH) 히스토그램 생성
for vehicle, soh_values in delta_soh_values.items():
    save_histogram(soh_values, f"{vehicle} SOH 변화량(ΔSOH) 분포", f"{vehicle}_SOH_변화량_histogram.png", output_folders["change_soh"])

print("\n 모든 히스토그램 저장 완료!")
