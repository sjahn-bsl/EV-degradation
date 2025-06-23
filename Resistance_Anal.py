import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from vehicle_dict import vehicle_dict

# 입력 및 출력 경로
root_input_folder = r"E:\SamsungSTF\Processed_Data\TripByTrip"
output_csv = r"D:\SamsungSTF\Processed_Data\Domain\R_peak_cell_Anal.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# 병렬 셀 수 맵
parallel_map = {
    'Bongo3EV': 3, 'EV6': 2, 'GV60': 2,
    'Ioniq5': 2, 'Ioniq6': 2, 'KonaEV': 3,
    'NiroEV': 3, 'Porter2EV': 3
}

# device_id → vehicle_type 역맵 생성
device_to_vehicle = {
    device_id: vehicle_type
    for vehicle_type, device_list in vehicle_dict.items()
    for device_id in device_list
}

# 파라미터 (논문 조건 기반)
MAX_INITIAL_CURRENT = 1.25  # A
MIN_DELTA_I = 3  # A
MIN_DURATION = 4  # 샘플 기준 (2초 샘플링 시 8초)

results = []

# 카운터 초기화
count_total = 0
count_all_acc_segments = 0
count_valid_duration_zones = 0
count_initialcurrent_pass = 0
count_deltacurrent_pass = 0
count_didt_pass = 0
count_final_R_calculated = 0

# 전체 파일 목록 수집
all_files = [f for f in os.listdir(root_input_folder) if f.endswith('.csv')]

with tqdm(total=len(all_files), desc="전체 차량 처리 중", unit="file") as pbar:
    for filename in all_files:
        try:
            file_path = os.path.join(root_input_folder, filename)
            df = pd.read_csv(file_path)
            count_total += 1

            match = re.search(r'bms(?:_altitude)?_(\d+)', filename)
            device_id = match.group(1) if match else "unknown"

            vehicle_type = device_to_vehicle.get(device_id, None)
            if vehicle_type is None:
                pbar.update(1)
                continue

            N_p = parallel_map.get(vehicle_type)
            if N_p is None:
                pbar.update(1)
                continue

            required_cols = {'pack_current', 'pack_volt', 'cell_volt_list', 'time', 'acceleration'}
            if not required_cols.issubset(df.columns):
                pbar.update(1)
                continue

            df['cell_count'] = df['cell_volt_list'].apply(lambda x: len(str(x).split(',')))
            N_s = df['cell_count'].mode().iloc[0]

            df['cell_volt'] = df['pack_volt'] / N_s
            df['cell_current'] = df['pack_current'] / N_p

            df['I_smooth'] = df['cell_current'].rolling(window=50, min_periods=1).mean()
            df['dI_dt'] = df['I_smooth'].diff()

            # 가속 구간 정의를 변경: cell_current > 0인 연속 구간을 후보로 사용
            df['current_flag'] = (df['cell_current'] > 0).astype(int)
            df['flag_shift'] = df['current_flag'].shift().fillna(0)
            df['flag_change'] = df['current_flag'] != df['flag_shift']
            change_indices = df.index[df['flag_change']].tolist() + [len(df)]

            R_list = []
            for k in range(len(change_indices) - 1):
                start = change_indices[k]
                end = change_indices[k + 1]

                if df.loc[start, 'current_flag'] == 1:
                    count_all_acc_segments += 1

                    if (end - start) >= MIN_DURATION:
                        count_valid_duration_zones += 1

                        if abs(df.loc[start, 'cell_current']) < MAX_INITIAL_CURRENT:
                            count_initialcurrent_pass += 1

                            delta_I = df.loc[end - 1, 'cell_current'] - df.loc[start, 'cell_current']
                            if delta_I >= MIN_DELTA_I:
                                count_deltacurrent_pass += 1

                                dI_segment = df['dI_dt'].iloc[start:end]
                                if (dI_segment >= 0).all():
                                    count_didt_pass += 1

                                    V_start = df.loc[start, 'cell_volt']
                                    V_end = df.loc[end - 1, 'cell_volt']
                                    R_cell = -(V_end - V_start) / delta_I
                                    R_list.append(R_cell)
                                    count_final_R_calculated += 1

            if R_list:
                if 'soc' in df.columns and not df['soc'].isnull().all():
                    soc_start = df['soc'].iloc[0]
                    soc_end = df['soc'].iloc[-1]
                    delta_soc = abs(soc_end - soc_start)
                else:
                    delta_soc = np.nan

                results.append({
                    'device_id': device_id,
                    'vehicle_type': vehicle_type,
                    'filename': filename,
                    'cell_count': N_s,
                    'num_acc_zones': len(R_list),
                    'R_cell_peak_avg': np.mean(R_list),
                    'R_cell_peak_std': np.std(R_list),
                    'delta_soc': delta_soc,
                    'start_time': df['time'].iloc[0],
                    'SOH_BMS': df['soh'].iloc[0]
                })

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

        pbar.update(1)

summary_df = pd.DataFrame(results)
summary_df["start_time"] = pd.to_datetime(summary_df["start_time"], errors='coerce')
summary_df = summary_df.sort_values(by=["device_id", "start_time"])
summary_df.to_csv(output_csv, index=False)
print(f"[완료] 모든 차량에 대한 R_peak (논문 기준) 결과 저장 완료: {output_csv}")

print("\n[필터링 요약]")
print(f"전체 파일 수                     : {count_total}")
print(f"총 가속 구간 수                   : {count_all_acc_segments}")
print(f"지속시간 8초 이상 유효 구간 수     : {count_valid_duration_zones}")
print(f"초기 전류 조건 통과 수             : {count_initialcurrent_pass}")
print(f"전류 증가량 조건 통과 수           : {count_deltacurrent_pass}")
print(f"dI/dt 단조 증가 조건 통과 수       : {count_didt_pass}")
print(f"최종 R 계산 완료 수                : {count_final_R_calculated}")
