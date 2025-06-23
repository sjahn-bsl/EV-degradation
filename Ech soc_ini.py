import os
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm

# 입력 및 출력 경로
input_folder = r"E:\SamsungSTF\Processed_Data\Charging_merged\EV6"
output_csv = r"D:\SamsungSTF\Processed_Data\Domain\Ech_soc_ini.csv"

# 상위 디렉토리가 존재하지 않으면 생성
output_dir = os.path.dirname(output_csv)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# EV6 기준 변수
N_p = 2  # 병렬 셀 수
V_MIN = 3.7
V_MAX = 3.9
I_MIN = -8.86
I_MAX = -7.14
Q_rated = 56.168  # Ah

# 결과 저장용 딕셔너리 및 필터 통과 카운터
device_dict = defaultdict(list)
count_total = 0
count_voltage_pass = 0
count_current_pass = 0
count_soc_pass = 0
count_final_ech = 0

# 처리 대상 파일 목록
all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

with tqdm(total=len(all_files), desc="EV6 파일 처리", unit="file") as pbar:
    for filename in all_files:
        file_path = os.path.join(input_folder, filename)
        count_total += 1

        try:
            df = pd.read_csv(file_path)

            # time 파싱 및 정렬
            if 'time' not in df.columns or df['time'].isnull().all():
                pbar.update(1)
                continue
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)
            df.sort_values('time', inplace=True)

            # 셀 수 계산
            if 'cell_volt_list' not in df.columns or df['cell_volt_list'].isnull().all():
                pbar.update(1)
                continue
            df['cell_count'] = df['cell_volt_list'].apply(lambda x: len(str(x).split(',')))
            battery_count = int(df['cell_count'].iloc[0])

            # 전압/전류/Crate 계산
            df['cell_volt'] = df['pack_volt'] / battery_count
            df['cell_current'] = df['pack_current'] / N_p
            df['C_rate'] = df['pack_current'].abs() / Q_rated

            # 필터 조건 1: 전압 범위
            start_cell_volt = df['cell_volt'].iloc[0]
            end_cell_volt = df['cell_volt'].iloc[-1]

            # 🔹 시작 및 종료 시점의 평균 모듈 온도 추출
            ini_mod_temp_avg = df['mod_temp_avg'].iloc[0] if 'mod_temp_avg' in df.columns else None
            end_mod_temp_avg = df['mod_temp_avg'].iloc[-1] if 'mod_temp_avg' in df.columns else None

            if start_cell_volt > V_MIN or end_cell_volt < V_MAX:
                pbar.update(1)
                continue
            count_voltage_pass += 1

            # 필터 조건 2: 전류 범위 (pack_current 기준)
            df_voltage_valid = df[
                (df['cell_volt'] >= V_MIN) & (df['cell_volt'] <= V_MAX)
            ].copy()
            invalid_current = df_voltage_valid[
                (df_voltage_valid['pack_current'] < I_MIN) |
                (df_voltage_valid['pack_current'] > I_MAX)
            ]
            if len(invalid_current) >= 5:
                pbar.update(1)
                continue
            count_current_pass += 1

            # 필터 조건 3: soc_initial ≤ 48
            if 'soc' not in df.columns or df['soc'].isnull().all():
                pbar.update(1)
                continue
            soc_initial = df['soc'].iloc[0]
            if soc_initial > 48:
                pbar.update(1)
                continue
            count_soc_pass += 1

            # 최종 유효 데이터로 Ech 계산
            df_valid = df_voltage_valid[
                (df_voltage_valid['pack_current'] >= I_MIN) &
                (df_voltage_valid['pack_current'] <= I_MAX)
            ].copy()
            if len(df_valid) < 2:
                pbar.update(1)
                continue

            df_valid['dt'] = df_valid['time'].diff().dt.total_seconds().fillna(0)
            df_valid['power_cell'] = df_valid['cell_volt'] * df_valid['cell_current'].abs()
            df_valid['energy_ws'] = df_valid['power_cell'] * df_valid['dt']
            ech_wh = df_valid['energy_ws'].sum() / 3600
            ech_wh_per_sec = ech_wh / df_valid['dt'].sum()

            soc_end = df['soc'].iloc[-1]
            delta_soc = abs(soc_end - soc_initial)

            device_id = re.search(r'bms(?:_altitude)?_(\d+)-', filename).group(1)

            trip_info = {
                'device_id': device_id,
                'filename': filename,
                'Ech_cell_Wh': ech_wh,
                'Ech_cell_Wh_per_sec': ech_wh_per_sec,
                'delta_soc(%)': delta_soc,
                'soc_initial(%)': soc_initial,
                'soc_end(%)': soc_end,
                'start_cell_volt': start_cell_volt,
                'end_cell_volt': end_cell_volt,
                'ini_mod_temp_avg': ini_mod_temp_avg,
                'end_mod_temp_avg': end_mod_temp_avg,
                'cell_count': battery_count,
                'samples': len(df_valid),
                'valid_duration_sec': df_valid['dt'].sum(),
                'start_time': df['time'].iloc[0]
            }

            device_dict[device_id].append(trip_info)
            count_final_ech += 1

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
        finally:
            pbar.update(1)

# 손실률 계산 및 저장
results = []
for device_id, trip_list in device_dict.items():
    sorted_list = sorted(trip_list, key=lambda x: x['start_time'])
    fresh_val = sorted_list[0]['Ech_cell_Wh']
    for trip in sorted_list:
        trip['Ech_cell_loss(%)'] = (fresh_val - trip['Ech_cell_Wh']) / fresh_val * 100
        results.append(trip)

summary_df = pd.DataFrame(results)
summary_df.to_csv(output_csv, index=False)

# 필터링 통계 출력
print("\n===== 필터링 결과 요약 =====")
print(f"전체 파일 수: {count_total}")
print(f"전압 조건 통과: {count_voltage_pass}")
print(f"전류 조건 통과: {count_current_pass}")
print(f"SOC 초기값 조건 통과: {count_soc_pass}")
print(f"최종 Ech 계산된 파일 수: {count_final_ech}")
print(f"결과 저장 완료: {output_csv}")
