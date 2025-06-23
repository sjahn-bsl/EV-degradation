import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

# 입력 및 출력 경로 설정
input_folder = r"E:\SamsungSTF\Processed_Data\Charging_merged\EV6"
output_csv = r"D:\SamsungSTF\Processed_Data\Domain\Zchg_ist"

# 상위 디렉토리가 존재하지 않으면 생성
output_dir = os.path.dirname(output_csv)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# 필터 파라미터 정의
V_MIN = 3.7
V_MAX = 3.9
I_MIN = -8.86
I_MAX = -7.14
SOC_MAX = 100
Q_rated = 56.168  # Ah

# 차량별 병렬 셀 수 정의
parallel_map = {
    'Bongo3EV': 3, 'EV6': 2, 'GV60': 2,
    'Ioniq5': 2, 'Ioniq6': 2, 'KonaEV': 3,
    'NiroEV': 3, 'Porter2EV': 3
}

device_dict = defaultdict(list)

# 결과 저장용 딕셔너리 및 필터 통과 카운터
device_dict = defaultdict(list)
count_total = 0
count_voltage_pass = 0
count_current_pass = 0
count_soc_pass = 0
count_final_ech = 0

def calculate_Zchg_for_file(file_path):
    global count_voltage_pass, count_current_pass, count_soc_pass, count_final_ech
    try:
        df = pd.read_csv(file_path)

        vehicle_type = os.path.basename(os.path.dirname(file_path))
        N_p = parallel_map.get(vehicle_type)
        if N_p is None:
            return None

        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            try:
                df['time'] = pd.to_datetime(df['time'], format='%y-%m-%d %H:%M:%S')
            except ValueError:
                df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

        df['cell_count'] = df['cell_volt_list'].apply(lambda x: len(str(x).split(',')))
        battery_count = df['cell_count'].iloc[0]
        df['cell_volt'] = df['pack_volt'] / battery_count
        df['cell_current'] = df['pack_current'] / N_p

        # 1. 전압 조건 필터
        start_cell_volt = df['cell_volt'].iloc[0]
        end_cell_volt = df['cell_volt'].iloc[-1]
        if start_cell_volt > V_MIN or end_cell_volt < V_MAX:  # ✔ 부등호 방향도 Ech와 동일하게 맞춤
            return None
        count_voltage_pass += 1

        # 2. 전류 조건 필터 (Ech 방식으로 수정: 전압 조건 구간에서만 필터링)
        df_voltage_valid = df[
            (df['cell_volt'] >= V_MIN) & (df['cell_volt'] <= V_MAX)
            ].copy()

        invalid_current = df_voltage_valid[
            (df_voltage_valid['pack_current'] <= I_MIN) |
            (df_voltage_valid['pack_current'] >= I_MAX)
            ]
        if len(invalid_current) >= 5:
            return None
        count_current_pass += 1

        # 3. SOC 초기값 조건 필터
        if 'soc' not in df.columns or df['soc'].isnull().all():
            return None
        soc_initial = df['soc'].iloc[0]
        soc_end = df['soc'].iloc[-1]
        if soc_initial > SOC_MAX:  # 부등호 방향도 Ech와 동일하게 맞춤
            return None
        count_soc_pass += 1

        # Z_CHG 계산 (전압 범위 [3.7V, 3.9V] 구간에서만)
        df_zchg = df[
            (df['cell_volt'] >= V_MIN) & (df['cell_volt'] <= V_MAX)
        ].copy()

        if df_zchg.empty or len(df_zchg) < 2:
            return None

        df_zchg['delta_V'] = df_zchg['cell_volt'].diff()
        df_zchg['Z_chg_inst'] = df_zchg['delta_V'] / df_zchg['cell_current'].abs()
        Z_CHG = df_zchg['Z_chg_inst'].iloc[1:].replace([np.inf, -np.inf], np.nan).dropna().mean()

        if pd.isna(Z_CHG):
            return None

        # Ech 계산
        df['dt'] = df['time'].diff().dt.total_seconds().fillna(0)
        df['power_cell'] = df['cell_volt'] * df['cell_current'].abs()
        df['energy_ws'] = df['power_cell'] * df['dt']
        ech_wh = df['energy_ws'].sum() / 3600
        ech_wh_per_sec = ech_wh / df['dt'].sum() if df['dt'].sum() > 0 else np.nan

        # 온도 평균
        ini_mod_temp_avg = df['mod_temp_avg'].iloc[0] if 'mod_temp_avg' in df.columns else None
        end_mod_temp_avg = df['mod_temp_avg'].iloc[-1] if 'mod_temp_avg' in df.columns else None

        delta_soc = abs(soc_end - soc_initial)

        count_final_ech += 1

        return {
            'Z_CHG': Z_CHG,
            'avg_pack_current': df['pack_current'].mean(),
            'battery_count': battery_count,
            'trip_length': len(df),
            'odometer_start': df['odometer'].iloc[0] if 'odometer' in df.columns else None,
            'start_time': df['time'].iloc[0],
            'start_cell_volt': start_cell_volt,
            'end_cell_volt': end_cell_volt,
            'vehicle_type': vehicle_type,
            'delta_soc': delta_soc,
            'soc_initial': soc_initial,
            'soc_end': soc_end,
            'ech_wh': ech_wh,
            'ech_wh_per_sec': ech_wh_per_sec,
            'ini_mod_temp_avg': ini_mod_temp_avg,
            'end_mod_temp_avg': end_mod_temp_avg,
            'valid_duration': df['dt'].sum()
        }

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None


# 반복 처리
all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

with tqdm(total=len(all_files), desc="[Z_CHG] Processing", unit="file") as pbar:
    for filename in all_files:
        count_total += 1
        file_path = os.path.join(input_folder, filename)

        match = re.search(r'bms(?:_altitude)?_(\d+)-', filename)
        if not match:
            print(f"[SKIP] device_id 추출 실패: {filename}")
            pbar.update(1)
            continue
        device_id = match.group(1)

        result = calculate_Zchg_for_file(file_path)

        if result is not None:
            trip_info = {
                'device_id': device_id,
                'filename': filename,
                'Z_CHG': result['Z_CHG'],
                'avg_pack_current': result['avg_pack_current'],
                'Ech_cell_Wh': result['ech_wh'],
                'Ech_cell_Wh_per_sec': result['ech_wh_per_sec'],
                'delta_soc(%)': result['delta_soc'],
                'soc_initial(%)': result['soc_initial'],
                'soc_end(%)': result['soc_end'],
                'start_cell_volt': result['start_cell_volt'],
                'end_cell_volt': result['end_cell_volt'],
                'ini_mod_temp_avg': result['ini_mod_temp_avg'],
                'end_mod_temp_avg': result['end_mod_temp_avg'],
                'cell_count': result['battery_count'],
                'samples': result['trip_length'],
                'valid_duration_sec': result['valid_duration'],
                'start_time': result['start_time'],
                'vehicle_type': result['vehicle_type']
            }

            device_dict[device_id].append(trip_info)

        pbar.update(1)

# 증가율 계산
results = []
for device_id, trip_list in device_dict.items():
    sorted_list = sorted(trip_list, key=lambda x: x['filename'])
    fresh_val = sorted_list[0]['Z_CHG']
    for trip in sorted_list:
        trip['Z_CHG_increase(%)'] = (trip['Z_CHG'] - fresh_val) / fresh_val * 100
        results.append(trip)

summary_df = pd.DataFrame(results)
summary_df.to_csv(output_csv, index=False)

# 필터 통계 출력
print("\n===== 필터링 결과 요약 =====")
print(f"전체 파일 수: {count_total}")
print(f"전압 조건 통과: {count_voltage_pass}")
print(f"전류 조건 통과: {count_current_pass}")
print(f"SOC 조건 통과: {count_soc_pass}")
print(f"최종 Z_CHG 계산 완료: {count_final_ech}")
print(f"결과 저장 완료: {output_csv}")

