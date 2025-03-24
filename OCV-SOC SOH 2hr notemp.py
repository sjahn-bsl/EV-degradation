import os
import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.interpolate import interp1d

# Trip CSV 파일들이 저장된 폴더 경로
trip_data_folder = r"E:\SamsungSTF\Processed_Data\TripByTrip_soc_2hr_notemp"
trip_files = []
for root, _, files in os.walk(trip_data_folder):
    for file in files:
        if file.endswith(".csv"):
            trip_files.append(os.path.join(root, file))

# 결과 저장용 딕셔너리
results_dict = {
    0.2: [],
    0.3: [],
    0.4: []
}

# OCV-SOC 데이터 로드 (엑셀 파일에서 SOC-OCV 시트 읽기)
ocv_soc_df = pd.read_excel(
    r"C:\Users\BSL\Desktop\EV 열화 인자 분석\250306\NE_Cell_Characterization_performance.xlsx",
    sheet_name="SOC-OCV"
)

# 엑셀의 SOC(%) 값이 있는 행을 확인하고 불필요한 행 제거 (B8:B108, D8:D108 사용)
ocv_soc_df = ocv_soc_df.iloc[7:108]  # 8번째 행부터 108번째 행까지 사용

# 방전 방향 OCV 데이터 사용 (엑셀 기준 B열 = 1번째 열, D열 = 3번째 열)
soc_values = ocv_soc_df.iloc[:, 1].astype(float, errors='ignore').dropna().values  # SOC (0~100%) → B열
ocv_values = ocv_soc_df.iloc[:, 3].astype(float, errors='ignore').dropna().values  # 방전 방향 OCV (V) → D열

# OCV-SOC 변환 함수 생성 (보간법 사용)
ocv_to_soc_interp = interp1d(ocv_values, soc_values, kind="linear", fill_value="extrapolate")

# 차량별 초기 배터리 용량 설정 (cell 단위로 변환)
Q_initial_map = {
    'Bongo3EV': 180 / 3, 'EV6': 120.6 / 2, 'GV60': 111.2 / 2,
    'Ioniq5': 120.6 / 2, 'Ioniq6': 111.2 / 2, 'KonaEV': 180.9 / 3,
    'NiroEV': 180 / 3, 'Porter2EV': 180 / 3
}

# 차량별 병렬(Parallel) 개수 설정
parallel_map = {
    'Bongo3EV': 3, 'EV6': 2, 'GV60': 2,
    'Ioniq5': 2, 'Ioniq6': 2, 'KonaEV': 3,
    'NiroEV': 3, 'Porter2EV': 3
}

# 차량별 배터리 Pack 1개에 들어있는 셀 개수
battery_cells = {
    'Bongo3EV': 90, 'EV6': 192, 'GV60': 192, 'Ioniq5': 180, 'Ioniq6': 192,
    'KonaEV': 98, 'NiroEV': 96, 'Porter2EV': 90
}

# 차량 모델과 device_id 매핑
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

for model, devices in vehicle_dict.items(): # model: 차량 모델, devices: 해당 차량 모델의 device_id 리스트
    for device in devices:
        device_to_model[device] = model # device_id를 device_to_model 딕셔너리에 저장함.

def calculate_SOH_OCV(file_path, threshold):
    try:
        file_name = os.path.basename(file_path)

        # 정규표현식으로 파일이름에서 device_id 추출
        match = re.search(r'bms(?:_altitude)?_(\d+)-\d{4}-\d{2}-trip-\d+', file_name) # (?:_altitude)?: _altitude가 있어도 되고 없어도 됨.
        if match:
            device_id = match.group(1) # 첫 번째 캡처 그룹 (device_id)
        else:
            print(f" {file_name}: device_id를 추출할 수 없습니다!")
            return {"file": file_name, "error": "Invalid file name format"}

        vehicle_model = device_to_model.get(device_id, None) # device_id를 사용하여 차량 모델을 찾음
        if vehicle_model is None: # 차량 모델을 찾을 수 없으면 오류 메세지를 반환합니다.
            print(f" {file_name}: device_id {device_id}가 vehicle_dict에서 찾을 수 없습니다!")
            return {"file": file_name, "error": f"Device ID {device_id} not found in vehicle_dict"}

        Q_initial = Q_initial_map.get(vehicle_model, None) # 차량 모델별 초기 배터리 용량을 찾음
        if Q_initial is None: # 값이 없으면 오류 메시지를 반환하고 종료합니다.
            print(f" {file_name}: vehicle_model {vehicle_model}에 대한 Q_initial을 찾을 수 없습니다!")
            return {"file": file_name, "error": f"No Q_initial found for {vehicle_model}"}

        parallel_count = parallel_map.get(vehicle_model, None) # 차량 모델별 pack의 병렬 개수를 찾음
        if parallel_count is None: # 값이 없으면 오류 메시지를 반환하고 종료합니다.
            print(f" {file_name}: 병렬 개수 정보를 찾을 수 없습니다!")
            return {"file": file_name, "error": f"No parallel count found for {vehicle_model}"}

        # CSV 파일 불러오기
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time']) # time 컬럼을 datetime 형식으로 변환
        df['soc'] = df['soc'] * 0.01  # SOC 값을 0~1 범위로 변환
        df['cell_current'] = df['pack_current'] / parallel_count  # 차량별 배터리의 Parallel 고려하여 cell_current 계산

        trip_start_time = df['time'].iloc[0]  # 첫 번째 time 값
        soh_initial = df['soh'].iloc[0] if 'soh' in df.columns else None # 첫 번째 SOH 값
        odometer_initial = df['odometer'].iloc[0] if 'odometer' in df.columns else None  # 첫 번째 odometer 값

        # pack_volt0, pack_volt1 계산
        pack_volt0 = df['pack_volt'].iloc[0] # 첫 번째 pack 전압
        pack_volt1 = df['pack_volt'].iloc[-1] # 마지막 pack 전압

        # 배터리 셀 개수를 Trip 별로 동적으로 계산
        df['cell_count'] = df['cell_volt_list'].apply(lambda x: len(str(x).split(',')))

        # 첫 번째 데이터의 셀 개수를 사용 (모든 행이 동일한 구조를 가지므로)
        battery_count = df['cell_count'].iloc[0]

        # OCV 계산
        OCV0 = pack_volt0 / battery_count
        OCV1 = pack_volt1 / battery_count

        # SOC 계산 (OCV-SOC 변환)
        SOC0 = ocv_to_soc_interp(OCV0)/100 # OCV-SOC 변환 함수를 사용하여 OCV0 값을 SOC0으로 변환
        SOC1 = ocv_to_soc_interp(OCV1)/100 # OCV-SOC 변환 함수를 사용하여 OCV1 값을 SOC1으로 변환

        # SOC 변화량 계산
        delta_SOC_OCV = (SOC1 - SOC0) # SOC 차이를 0~1 범위로 변환

        # SOC_BMS 초기값 계산
        soc_initial = df['soc'].iloc[0]
        if soc_initial == 0:
            non_zero_soc = df['soc'][df['soc'] > 0]
            if not non_zero_soc.empty:
                soc_initial = non_zero_soc.iloc[0]
            else:
                return {"file": file_name, "error": "SOC values are all zero"}
        # SOC_BMS 최종값 설정
        soc_end = df['soc'].iloc[-1]
        if soc_end == 0:
            non_zero_soc = df['soc'][df['soc'] > 0]
            if not non_zero_soc.empty:
                soc_end = non_zero_soc.iloc[-1]
            else:
                return {"file": file_name, "error": "SOC values are all zero"}

        # SOC 변화량 계산
        delta_BMS_SOC = soc_end - soc_initial

        if abs(delta_SOC_OCV) < threshold: # soc가 줄어드는 경우+증가하는 경우
            return None  # SOC 변화량이 너무 적으면 제외

        # 주어진 Trip동안 사용된 전류 데이터를 trapz로 적분하여 Q_current 계산
        df['time_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
        Q_current = np.trapz(df['cell_current'], df['time_seconds']) / -(delta_SOC_OCV) / 3600  # Ah 변환

        # SOH 계산
        SOH_OCV = (Q_current / Q_initial) * 100

        # 결과 반환
        return {
            "file": file_name, "device_id": device_id, "vehicle_model": vehicle_model,
            "time": trip_start_time,"odometer (km)": odometer_initial,
            "Number of cells": battery_count,
            "pack_volt0": pack_volt0, "pack_volt1": pack_volt1, "OCV0": OCV0, "OCV1": OCV1, "SOC0": SOC0, "SOC1": SOC1, "BMS_SOC0": soc_initial, "BMS_SOC1": soc_end,
            "delta_SOC": delta_SOC_OCV, "delta_BMS_SOC": delta_BMS_SOC, "Q_current (Ah)": Q_current, "Q_initial (Ah)": Q_initial,
            "SOH_BMS (%)": soh_initial, "SOH_OCV (%)": SOH_OCV
        }
    except Exception as e:
        return {"file": file_name, "error": str(e)}

if __name__ == "__main__":
    trip_files = []
    for root, _, files in os.walk(trip_data_folder):
        for file in files:
            if file.endswith(".csv"):
                trip_files.append(os.path.join(root, file))

    thresholds = [0.2, 0.3, 0.4]
    results_by_threshold = {th: [] for th in thresholds}

    for threshold in thresholds:
        print(f"\n Processing SOC Δ ≥ {int(threshold * 100)}% ...")
        with ProcessPoolExecutor() as executor:
            future_to_file = {
                executor.submit(calculate_SOH_OCV, file, threshold): file for file in trip_files
            }
            with tqdm(total=len(future_to_file), desc=f"Threshold {int(threshold*100)}%", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    result = future.result()
                    if result:
                        results_by_threshold[threshold].append(result)
                    pbar.update(1)

    # 결과 저장
    for threshold, results in results_by_threshold.items():
        df = pd.DataFrame(results)
        output_csv_path = fr"C:\Users\BSL\Desktop\EV 열화 인자 분석\250306\OCV-SOC SOH results 2hr notemp {int(threshold * 100)}%.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"SOC Δ ≥ {int(threshold * 100)}% 결과 저장 완료: {output_csv_path}")