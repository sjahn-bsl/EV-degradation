import os
import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 🔹 Trip CSV 파일들이 저장된 폴더 경로
trip_data_folder = r"D:\SamsungSTF\Processed_Data\TripByTrip"

# 🔹 차량별 초기 배터리 용량 설정 (cell 단위로 변환)
Q_initial_map = {
    'Bongo3EV': 180 / 3,
    'EV6': 120.6 / 2,
    'GV60': 111.2 / 2,
    'Ioniq5': 120.6 / 2,
    'Ioniq6': 111.2 / 2,
    'KonaEV': 180.9 / 3,
    'NiroEV': 180 / 3,
    'Porter2EV': 180 / 3
}

# 🔹 차량별 병렬(Parallel) 개수 설정
parallel_map = {
    'Bongo3EV': 3,
    'EV6': 2,
    'GV60': 2,
    'Ioniq5': 2,
    'Ioniq6': 2,
    'KonaEV': 3,
    'NiroEV': 3,
    'Porter2EV': 3
}

# 🔹 차량 모델과 device_id 매핑
device_to_model = {}
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
    'Porter2EV': ['01241228144', '01241228160', '01241228177', '01241228188', '01241228192', '01241228171'],
    'Bongo3EV': ['01241228162', '01241228179', '01241248642', '01241248723', '01241248829']
}

for model, devices in vehicle_dict.items():
    for device in devices:
        device_to_model[device] = model

def calculate_soh(file_path):
    try:
        file_name = os.path.basename(file_path)

        # 🔹 정규표현식으로 device_id 추출
        match = re.search(r'bms(?:_altitude)?_(\d+)-\d{4}-\d{2}-trip-\d+', file_name)
        if match:
            device_id = match.group(1)
        else:
            print(f"🚨 {file_name}: device_id를 추출할 수 없습니다!")
            return {"file": file_name, "error": "Invalid file name format"}

        vehicle_model = device_to_model.get(device_id, None)
        if vehicle_model is None:
            print(f"🚨 {file_name}: device_id {device_id}가 vehicle_dict에서 찾을 수 없습니다!")
            return {"file": file_name, "error": f"Device ID {device_id} not found in vehicle_dict"}

        Q_initial = Q_initial_map.get(vehicle_model, None)
        if Q_initial is None:
            print(f"🚨 {file_name}: vehicle_model {vehicle_model}에 대한 Q_initial을 찾을 수 없습니다!")
            return {"file": file_name, "error": f"No Q_initial found for {vehicle_model}"}

        parallel_count = parallel_map.get(vehicle_model, None)
        if parallel_count is None:
            print(f"🚨 {file_name}: 병렬 개수 정보를 찾을 수 없습니다!")
            return {"file": file_name, "error": f"No parallel count found for {vehicle_model}"}

        # 🔹 CSV 파일 불러오기
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df['soc'] = df['soc'] * 0.01  # SOC 값을 0~1 범위로 변환
        df['cell_current'] = df['pack_current'] / parallel_count  # 🚨 차량별 Parallel 고려하여 cell_current 계산

        # 🔹 SOC 초기값 및 최종값 계산
        soc_initial = df['soc'].iloc[0]  # SOC 첫 번째 값
        soc_end = df['soc'].iloc[-1]     # SOC 마지막 값

        # 🔹 SOC 변화량 계산
        delta_soc = soc_end - soc_initial
        if abs(delta_soc) < 0.2:
            return None  # SOC 변화량이 너무 적으면 제외

        # 🔹 Q_current 계산 (trapz 이용)
        df['time_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
        Q_current = abs(np.trapz(df['cell_current'], df['time_seconds'])) / abs(delta_soc) / 3600  # Ah 변환

        # 🔹 SOH 계산
        SOH = (Q_current / Q_initial) * 100

        return {
            "file": file_name, "device_id": device_id, "vehicle_model": vehicle_model,
            "soc_initial": soc_initial, "soc_end": soc_end,  # 🚀 추가된 SOC 초기값 및 최종값
            "delta_SOC": delta_soc, "Q_current (Ah)": Q_current, "Q_initial (Ah)": Q_initial, "SOH (%)": SOH
        }
    except Exception as e:
        return {"file": file_name, "error": str(e)}

if __name__ == "__main__":
    trip_files = [os.path.join(trip_data_folder, f) for f in os.listdir(trip_data_folder) if f.endswith(".csv")]
    results = []

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(calculate_soh, file): file for file in trip_files}
        with tqdm(total=len(trip_files), desc="Processing SOH", unit="file") as pbar:
            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)

    results_df = pd.DataFrame(results)

    # 🔹 변경된 CSV 파일명 설정
    output_csv_path = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224\BMS_SOC SOH 20%_results.csv"

    # 🔹 폴더가 없으면 생성
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 🔹 결과 저장 (CSV)
    results_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ SOH 계산 완료! 결과가 '{output_csv_path}' 파일에 저장되었습니다.")