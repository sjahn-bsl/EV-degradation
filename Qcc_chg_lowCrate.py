import os
import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.interpolate import interp1d

# Trip CSV 파일 경로
trip_data_folder = r"C:\Users\BSL\Desktop\EV 열화 인자 분석\Charging_merged"
trip_files = []
for root, _, files in os.walk(trip_data_folder):
    for file in files:
        if file.endswith(".csv"):
            trip_files.append(os.path.join(root, file))

# OCV-SOC 데이터 로드
ocv_soc_df = pd.read_excel(
    r"C:\Users\BSL\Desktop\EV 열화 인자 분석\250306\NE_Cell_Characterization_performance.xlsx",
    sheet_name="SOC-OCV"
)
ocv_soc_df = ocv_soc_df.iloc[7:108]
soc_values = ocv_soc_df.iloc[:, 1].astype(float).dropna().values
ocv_values = ocv_soc_df.iloc[:, 2].astype(float).dropna().values
ocv_to_soc_interp = interp1d(ocv_values, soc_values, kind="linear", fill_value="extrapolate")

# 차량별 초기 용량 및 병렬 수
Q_initial_map = {
    'Bongo3EV': 180 / 3, 'EV6': 120.6 / 2, 'GV60': 111.2 / 2,
    'Ioniq5': 120.6 / 2, 'Ioniq6': 111.2 / 2, 'KonaEV': 180.9 / 3,
    'NiroEV': 180 / 3, 'Porter2EV': 180 / 3
}
parallel_map = {
    'Bongo3EV': 3, 'EV6': 2, 'GV60': 2,
    'Ioniq5': 2, 'Ioniq6': 2, 'KonaEV': 3,
    'NiroEV': 3, 'Porter2EV': 3
}
battery_cells = {
    'Bongo3EV': 90, 'EV6': 192, 'GV60': 192, 'Ioniq5': 180, 'Ioniq6': 192,
    'KonaEV': 98, 'NiroEV': 96, 'Porter2EV': 90
}

# device_id와 모델 매핑
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

def calculate_Qcc_lowCrate(file_path, threshold):
    try:
        file_name = os.path.basename(file_path)
        match = re.search(r'bms(?:_altitude)?_(\d+)_\d{4}-\d{2}-trip', file_name)
        if not match:
            return {"file": file_name, "error": "Invalid filename format"}
        device_id = match.group(1)
        vehicle_model = device_to_model.get(device_id)
        if not vehicle_model:
            return {"file": file_name, "error": "Device not found"}
        Q_initial = Q_initial_map.get(vehicle_model)
        parallel_count = parallel_map.get(vehicle_model)
        if not Q_initial or not parallel_count:
            return {"file": file_name, "error": "Missing vehicle config"}

        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df['soc'] = df['soc'] * 0.01
        df['cell_current'] = df['pack_current'] / parallel_count

        df_trip = df.copy()

        trip_start_time = df['time'].iloc[0]
        soh_initial = df['soh'].iloc[0] if 'soh' in df.columns else None
        odometer_initial = df['odometer'].iloc[0] if 'odometer' in df.columns else None

        # 평균 C-rate 계산
        avg_c_rate = df['cell_current'].abs().mean() / Q_initial
        if avg_c_rate > 0.1:
            return None  # 완속 충전이 아니면 제외

        pack_volt0 = df['pack_volt'].iloc[0]
        pack_volt1 = df['pack_volt'].iloc[-1]

        df['cell_count'] = df['cell_volt_list'].apply(lambda x: len(str(x).split(',')))
        battery_count = df['cell_count'].iloc[0]

        OCV0 = pack_volt0 / battery_count
        OCV1 = pack_volt1 / battery_count

        SOC0 = ocv_to_soc_interp(OCV0) / 100
        SOC1 = ocv_to_soc_interp(OCV1) / 100
        delta_SOC = SOC1 - SOC0

        soc_initial = df['soc'].iloc[0]
        soc_end = df['soc'].iloc[-1]
        delta_BMS_SOC = soc_end - soc_initial

        if delta_SOC < threshold:
            return None

        df_trip['time_seconds'] = (df_trip['time'] - df_trip['time'].iloc[0]).dt.total_seconds()
        Q_current = -np.trapz(df_trip['cell_current'], df_trip['time_seconds']) / delta_SOC / 3600
        SOH_OCV = (Q_current / Q_initial) * 100

        return {
            "file": file_name, "device_id": device_id, "vehicle_model": vehicle_model,
            "C_rate_mean": avg_c_rate,
            "time": trip_start_time, "odometer(km)": odometer_initial,
            "pack_volt0": pack_volt0, "pack_volt1": pack_volt1,
            "OCV0": OCV0, "OCV1": OCV1,
            "SOC0": SOC0, "SOC1": SOC1,
            "BMS_SOC0": soc_initial, "BMS_SOC1": soc_end,
            "delta_SOC": delta_SOC, "delta_BMS_SOC": delta_BMS_SOC,
            "Q_current(Ah)": Q_current, "Q_initial(Ah)": Q_initial,
            "SOH_OCV(%)": SOH_OCV, "SOH_BMS(%)": soh_initial
        }
    except Exception as e:
        return {"file": file_name, "error": str(e)}

if __name__ == "__main__":
    thresholds = [0.05, 0.1, 0.2]  # 원하는 threshold 구간
    results_by_threshold = {th: [] for th in thresholds}

    for threshold in thresholds:
        print(f"\nProcessing threshold ΔSOC ≥ {threshold:.2f}...")
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(calculate_Qcc_lowCrate, f, threshold): f for f in trip_files}
            with tqdm(total=len(futures), desc=f"Threshold {threshold:.2f}", unit="file") as pbar:
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        results_by_threshold[threshold].append(res)
                    pbar.update(1)

    for threshold, results in results_by_threshold.items():
        df = pd.DataFrame(results)
        output_csv = rf"C:\Users\BSL\Desktop\EV 열화 인자 분석\Qcc_lowCrate_results_threshold_{int(threshold*100)}pct.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"SOC Δ ≥ {threshold:.2f} 결과 저장 완료: {output_csv}")
