import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 데이터 폴더 경로
data_folder = r"D:\SamsungSTF\Processed_Data\Merged"
output_base_folder = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250211\histogram"

# 히스토그램 저장 경로
output_folders = {
    "initial_soh": os.path.join(output_base_folder, "initial_soh"),
    "final_soh": os.path.join(output_base_folder, "final_soh"),
    "degradation_soh": os.path.join(output_base_folder, "degradation_soh")
}

# 출력 폴더가 없으면 생성
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# 차량 종류 리스트 (차량명: 단말기 리스트)
vehicle_dict = {
    'NiroEV': ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155'],
    'Ioniq5': ['01241227999', '01241228003', '01241228005', '01241228007', '01241228009', '01241228014',
               '01241228016', '01241228020', '01241228021', '01241228024', '01241228025', '01241228026', '01241228030',
               '01241228037', '01241228044', '01241228046', '01241228047', '01241248780', '01241248782',
               '01241248790', '01241248811', '01241248815', '01241248817', '01241248820', '01241248827',
               '01241364543', '01241364560', '01241364570', '01241364581', '01241592867', '01241592868',
               '01241592878', '01241592896', '01241592907', '01241597801', '01241597802', '01241248919',
               '01241321944'],
    'Ioniq6': ['01241248713', '01241592904', '01241597763', '01241597804'],
    'KonaEV': ['01241228102', '01241228122', '01241228123', '01241228156', '01241228197', '01241228203',
               '01241228204', '01241248726', '01241248727', '01241364621', '01241124056'],
    'EV6': ['01241225206', '01241228049', '01241228050', '01241228051', '01241228053',
            '01241228054', '01241228055', '01241228057', '01241228059', '01241228073', '01241228075',
            '01241228076', '01241228082', '01241228084', '01241228085', '01241228086', '01241228087',
            '01241228090', '01241228091', '01241228092', '01241228094', '01241228095', '01241228097',
            '01241228098', '01241228099', '01241228103', '01241228104', '01241228106', '01241228107',
            '01241228114', '01241228124', '01241228132', '01241228134', '01241248679', '01241248818',
            '01241248831', '01241248833', '01241248842', '01241248843', '01241248850', '01241248860',
            '01241248876', '01241248877', '01241248882', '01241248891', '01241248892', '01241248900',
            '01241248903', '01241248908', '01241248909', '01241248912', '01241248913', '01241248921', '01241248924',
            '01241248926', '01241248927', '01241248929', '01241248932', '01241248933', '01241248934',
            '01241321943', '01241321947', '01241364554', '01241364575', '01241364592', '01241364627',
            '01241364638', '01241364714', '01241248928'],
    'GV60': ['01241228108', '01241228130', '01241228131', '01241228136', '01241228137', '01241228138'],
    'Porter2EV' : ['01241228144', '01241228160', '01241228177', '01241228188', '01241228192', '01241228171'],
    'Bongo3EV' : ['01241228162', '01241228179', '01241248642', '01241248723', '01241248829']
}

# 차량별 SOH 값 저장 딕셔너리
first_soh_values = {vehicle: [] for vehicle in vehicle_dict.keys()}
final_soh_values = {vehicle: [] for vehicle in vehicle_dict.keys()}

# 차량별 최초 & 최종 SOH 추출
total_files = sum(len(device_ids) for device_ids in vehicle_dict.values())  # 총 단말기 개수
progress_bar = tqdm(total=total_files, desc="Processing Vehicles", unit="device", bar_format="{l_bar}{bar} [Remaining: {remaining}]")

for vehicle, device_ids in vehicle_dict.items():
    vehicle_path = os.path.join(data_folder, vehicle)

    if not os.path.exists(vehicle_path):
        print(f" 경로 없음: {vehicle_path}")
        progress_bar.update(len(device_ids))
        continue

    for device_id in device_ids:
        # 해당 단말기 번호에 해당하는 모든 CSV 파일 찾기
        csv_files = [f for f in os.listdir(vehicle_path) if f.startswith(f"bms_{device_id}") or f.startswith(f"bms_altitude_{device_id}")]

        if not csv_files:
            progress_bar.update(1)
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

        progress_bar.update(1)

progress_bar.close()

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

    # 히스토그램의 x축 범위 조정
    min_value, max_value = min(data) - 0.05, max(data) + 0.05
    bins = np.arange(min_value, max_value+0.1, 0.1)  # 0.1 단위 간격으로 설정

    # 히스토그램 범위 조정: 소수점 한 자리까지 (bins 간격 0.1)
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.75, rwidth=0.5)  # rwidth로 막대 두께 조절
    plt.title(title)
    plt.xlabel("SOH")
    plt.ylabel("frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # x축 범위를 적절히 설정
    plt.xlim(min_value, max_value)

    # x축 눈금 조정
    plt.xticks(np.round(np.arange(min_value, max_value, 0.1), 1), rotation=45) # X축 레이블을 0.1 단위로 표시

    save_path = os.path.join(folder, filename)
    plt.savefig(save_path, bbox_inches='tight')  # 그래프 저장 (여백 자동 조절)
    plt.close()  # 그래프 창 닫기

    print(f" 저장 완료: {save_path}")

#  최초 SOH 히스토그램 생성
for vehicle, soh_values in first_soh_values.items():
    save_histogram(soh_values, f"{vehicle} Initial SOH Distribution", f"{vehicle}_Initial_SOH_histogram.png", output_folders["initial_soh"])

#  최종 SOH 히스토그램 생성
for vehicle, soh_values in final_soh_values.items():
    save_histogram(soh_values, f"{vehicle} Final SOH Distribution", f"{vehicle}_Final_SOH_histogram.png", output_folders["final_soh"])

#  SOH 변화량(ΔSOH) 히스토그램 생성
for vehicle, soh_values in delta_soh_values.items():
    save_histogram(soh_values, f"{vehicle} SOH Degradation Distribution", f"{vehicle}_SOH_Degradation_histogram.png", output_folders["degradation_soh"])

print("\n 모든 히스토그램 저장 완료")
