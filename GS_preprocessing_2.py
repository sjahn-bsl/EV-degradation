import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from GS_vehicle_dict import vehicle_dict
from datetime import timedelta

def get_vehicle_type(device_no, vehicle_dict):
    """
    device_no를 받아서 어떤 차종인지 반환.
    만약 vehicle_dict에 없으면 "Unknown" 반환
    """
    for vtype, dev_list in vehicle_dict.items():
        if device_no in dev_list:
            return vtype
    return "Unknown"


def read_file_with_detected_encoding(file_path):
    """
    파일 인코딩을 추론하여 DataFrame으로 읽어들이는 함수.
    UTF-8 -> ISO-8859-1 -> Python engine UTF-8 순으로 시도.
    """
    try:
        return pd.read_csv(file_path, encoding='utf-8')  # 1) C engine, UTF-8
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding='iso-8859-1')  # 2) ISO-8859-1
        except Exception:
            # 3) Python engine + UTF-8
            try:
                return pd.read_csv(file_path, encoding='utf-8', engine='python')
            except Exception as e:
                print(f"Failed to read file {file_path} with Python engine due to: {e}")
                return None


def fill_altitude(df):
    """
    altitude 컬럼에 대하여,
      1) 처음 나온 altitude 값으로 이전 구간(NaN)을 채우고
      2) 중간 구간(NaN)은 선형 보간
      3) 마지막 altitude 값으로 이후 구간(NaN)을 채우는 함수
    """
    df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
    if df['altitude'].notnull().sum() == 0:
        return df

    first_valid_idx = df['altitude'].first_valid_index()
    last_valid_idx = df['altitude'].last_valid_index()

    # 1) 처음 발견된 altitude 값으로 이전 구간(NaN) 채우기
    first_value = df.loc[first_valid_idx, 'altitude']
    df.loc[:first_valid_idx, 'altitude'] = df.loc[:first_valid_idx, 'altitude'].fillna(first_value)

    # 2) 마지막 발견된 altitude 값으로 이후 구간(NaN) 채우기
    last_value = df.loc[last_valid_idx, 'altitude']
    df.loc[last_valid_idx:, 'altitude'] = df.loc[last_valid_idx:, 'altitude'].fillna(last_value)

    # 3) 중간 구간 선형 보간
    df['altitude'] = df['altitude'].interpolate(method='linear')
    return df


def split_into_periods_keep_midnight(df, period_days=7,
                                     midnight_gap_sec=60,   # 자정 ±1분
                                     continuity_sec=10,     # 10초 미만이면 연속
                                     pad_nextday_rows=15):  # 다음날 처음 15행
    """
    df를 'period_days' 간격으로 나누면서:
      - 첫 구간(i=0)일 경우:
          자정 이전부터 이미 진행 중이던 이벤트 -> 제거
          그 뒤에도 다음날 초반 15행 이어붙이기
      - 이후 구간(i>0)은
          자정 ±1분 & 직전 시점과 10초 미만으로 이어지는 구간은 cross
          + 다음날 초반 15행 이어붙이기
    """

    if df.empty:
        return {}

    df = df.sort_values('time').reset_index(drop=True)

    start_time = df['time'].min()
    end_time = df['time'].max()

    period_frames = {}
    period_idx = 1

    current_start = start_time
    while current_start <= end_time:
        current_end = current_start + pd.Timedelta(days=period_days)

        subdf = df[(df['time'] >= current_start) & (df['time'] < current_end)].copy()
        if subdf.empty:
            current_start = current_end
            period_idx += 1
            continue

        # -----------------------------
        # (i=0) => 자정 이전 이벤트 제거
        # -----------------------------
        if period_idx == 1:
            # “이미 진행 중이던 이벤트” 제거 로직:
            #   - 보통 current_start가 자정이라면, 이전 데이터는 필터링에서 자연스럽게 제외
            #   - 필요하다면 subdf 맨 앞에서 자정 이전부터 이어져 내려온 chunk를 제거할 수도 있음
            pass

        # -----------------------------
        # 자정 경계 넘어가는 이벤트(cross) + 다음날 초반 15행 포함
        # => i=0 이든 i>0 이든 '다음 period' next_df를 확인 후 처리
        # -----------------------------
        # subdf의 마지막 시간
        last_time_subdf = subdf['time'].iloc[-1]

        # 현재 구간 끝 시점(current_end) 이후(next_df)
        next_df = df[df['time'] >= current_end].copy()
        if not next_df.empty:
            # next_df의 첫 시각
            first_time_after_end = next_df['time'].iloc[0]

            # 1) 자정 ±1분 안이고
            if (first_time_after_end - current_end) < pd.Timedelta(seconds=midnight_gap_sec):
                # 2) 연속(10초 미만)
                if (first_time_after_end - last_time_subdf).total_seconds() < continuity_sec:
                    # cross
                    crossing_idx = []
                    prev_t = last_time_subdf
                    for idx2 in next_df.index:
                        curr_t = next_df.loc[idx2, 'time']
                        if (curr_t - prev_t).total_seconds() < continuity_sec:
                            crossing_idx.append(idx2)
                            prev_t = curr_t
                        else:
                            break
                    if crossing_idx:
                        crossing_df = next_df.loc[crossing_idx].copy()
                        subdf = pd.concat([subdf, crossing_df], ignore_index=True)

            # (공통) 다음날 초반부 15행 붙이기
            pad_rows = next_df.head(pad_nextday_rows)
            if not pad_rows.empty:
                subdf = pd.concat([subdf, pad_rows], ignore_index=True)
                subdf.drop_duplicates(subset='time', inplace=True)
                subdf.sort_values('time', inplace=True)

        # 저장
        period_frames[period_idx] = subdf

        # 다음 구간 준비
        current_start = current_end
        period_idx += 1

    return period_frames


def process_device_folder(device_folder_path, save_path, vehicle_type,
                          altitude=False, period_days=7):
    """
    1) CSV 병합
    2) period_days 단위 split_into_periods_keep_midnight 로 나눔
       - i=0: 자정 이전 이벤트 제거, + 다음날 15행
       - i>0: cross + 다음날 15행
    3) CSV 저장
    """
    device_no = os.path.basename(device_folder_path)  # 폴더 이름이 단말기번호
    vehicle_model = get_vehicle_type(device_no, vehicle_type)

    # -----------------------------
    # 1) CSV 파일 모아 읽기
    # -----------------------------
    csv_files = []
    for root, dirs, files in os.walk(device_folder_path):
        for f in files:
            if f.endswith(".csv"):
                if altitude:
                    # altitude=True -> 'bms'+'altitude'가 파일명에 포함
                    if "bms" in f and "altitude" in f:
                        csv_files.append(os.path.join(root, f))
                else:
                    # altitude=False -> 'bms' 포함 & 'altitude' 미포함
                    if "bms" in f and "altitude" not in f:
                        csv_files.append(os.path.join(root, f))

    if len(csv_files) < 20:
        print(f"[{device_no}] CSV count({len(csv_files)}) < 20. Skipping...")
        return

    if not csv_files:
        print(f"[{device_no}] No CSV files found. Skipping...")
        return

    dfs = []
    for file_path in csv_files:
        df = read_file_with_detected_encoding(file_path)
        if df is None:
            continue

        # 불필요한 'Unnamed' 등 제거
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]

        # time 중복 제거
        df = df.drop_duplicates(subset='time')

        # time -> datetime 변환
        date_formats = ['%y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S']
        for fmt in date_formats:
            try:
                df['time'] = pd.to_datetime(df['time'], format=fmt, errors='raise')
                break
            except ValueError:
                continue
        else:
            print(f"[{device_no}] Time format error in file {file_path}. Skipping...")
            continue

        # 정렬
        df = df.sort_values(by='time').reset_index(drop=True)
        dfs.append(df)

    if not dfs:
        print(f"[{device_no}] All CSV files invalid or empty. Skipping...")
        return

    combined_df = pd.concat(dfs, ignore_index=True).sort_values(by='time').reset_index(drop=True)

    # -----------------------------
    # 2) 전처리: speed, accel, Power_data 등
    # -----------------------------
    combined_df['time_diff'] = combined_df['time'].diff().dt.total_seconds()

    # speed(m/s) = emobility_spd(km/h) * 0.27778
    if 'emobility_spd' in combined_df.columns:
        combined_df['speed'] = combined_df['emobility_spd'] * 0.27778
    else:
        combined_df['speed'] = 0

    # accel = delta speed / delta time
    combined_df['acceleration'] = combined_df['speed'].diff() / combined_df['time_diff']
    if len(combined_df) > 1:
        combined_df.at[0, 'acceleration'] = (
            combined_df.at[1, 'speed'] - combined_df.at[0, 'speed']
        ) / combined_df.at[1, 'time_diff']
        combined_df.at[len(combined_df) - 1, 'acceleration'] = (
            combined_df.at[len(combined_df) - 1, 'speed'] -
            combined_df.at[len(combined_df) - 2, 'speed']
        ) / combined_df.at[len(combined_df) - 1, 'time_diff']
    combined_df['acceleration'] = combined_df['acceleration'].fillna(0)

    # Power_data = pack_volt * pack_current
    pack_volt = combined_df['pack_volt'] if 'pack_volt' in combined_df.columns else 0
    pack_current = combined_df['pack_current'] if 'pack_current' in combined_df.columns else 0
    combined_df['Power_data'] = pack_volt * pack_current

    # altitude 보정
    if altitude:
        combined_df = fill_altitude(combined_df)

    # -----------------------------
    # 3) N일 단위 분할
    # -----------------------------
    splitted = split_into_periods_keep_midnight(
        combined_df,
        period_days=period_days,
        midnight_gap_sec=60,   # 자정 ±1분
        continuity_sec=10,     # 10초 미만이면 연속
        pad_nextday_rows=15    # 다음날 첫 15행
    )

    if not splitted:
        print(f"[{device_no}] All data removed or no valid splitting.")
        return

    # -----------------------------
    # 4) CSV 저장
    # -----------------------------
    device_save_folder = os.path.join(save_path, vehicle_model)
    os.makedirs(device_save_folder, exist_ok=True)

    for p_idx, grp in splitted.items():
        if grp.empty:
            continue

        # altitude 여부에 따라 열 구성 분기
        if altitude:
            data_save = grp[
                ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp',
                 'chrg_cnt', 'chrg_cnt_q', 'cumul_energy_chrgd',
                 'cumul_energy_chrgd_q', 'mod_temp_list', 'odometer',
                 'op_time', 'soc', 'soh', 'chrg_cable_conn',
                 'altitude', 'cell_volt_list', 'min_deter',
                 'pack_volt', 'pack_current', 'Power_data']
            ].copy()
        else:
            data_save = grp[
                ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp',
                 'chrg_cnt', 'chrg_cnt_q', 'cumul_energy_chrgd',
                 'cumul_energy_chrgd_q', 'mod_temp_list', 'odometer',
                 'op_time', 'soc', 'soh', 'chrg_cable_conn',
                 'pack_volt', 'pack_current', 'cell_volt_list', 'min_deter',
                 'Power_data']
            ].copy()

        output_name = f"{'bms_altitude' if altitude else 'bms'}_{device_no}_d{int(p_idx)}.csv"
        output_path = os.path.join(device_save_folder, output_name)
        data_save.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[{device_no}] Period {p_idx} -> {output_path} (rows={len(data_save)})")


def merge_bms_data_by_device(start_path, save_path,
                             vehicle_type,
                             altitude=False,
                             period_days=7):
    """
    start_path 안의 디바이스 폴더 각각 병렬 처리:
      1) CSV 병합
      2) N일 단위 split_into_periods_keep_midnight() 로직
         -> (i=0)도 다음날 15행 추가
         -> (i>0)는 cross + 15행 추가
      3) 결과 CSV 저장
    """
    if vehicle_type is None:
        vehicle_type = {}

    device_folders = [
        d for d in os.listdir(start_path)
        if os.path.isdir(os.path.join(start_path, d))
    ]
    if not device_folders:
        print("No device folders found. Check the start_path.")
        return

    total_devices = len(device_folders)
    print(f"Found {total_devices} device folders. Starting... (period_days={period_days})")

    with ProcessPoolExecutor() as executor:
        future_to_device = {}
        for device_folder_name in device_folders:
            device_folder_path = os.path.join(start_path, device_folder_name)
            future = executor.submit(
                process_device_folder,
                device_folder_path,
                save_path,
                vehicle_type,
                altitude,
                period_days
            )
            future_to_device[future] = device_folder_name

        # 진행도 표시용 tqdm
        with tqdm(total=total_devices, desc="Merging by device") as pbar:
            for future in as_completed(future_to_device):
                device_folder_name = future_to_device[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[{device_folder_name}] Error: {e}")
                pbar.update(1)

    print("=== All device folders processed. ===")


def process_files_trip_by_trip(start_path, save_path):
    """
    CSV 파일을 trip 단위로 분할/저장하는 메인 함수.
    """
    csv_files = [os.path.join(root, file)
                 for root, _, files in os.walk(start_path)
                 for file in files if file.endswith('.csv')]
    total_files = len(csv_files)

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        future_to_file = {}
        with ProcessPoolExecutor() as executor:
            for file_path in csv_files:
                future = executor.submit(process_wrapper, file_path, save_path)
                future_to_file[future] = file_path

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'File {file_path} generated an exception: {exc}')
                finally:
                    pbar.update(1)

    print("Processing complete")


def process_wrapper(file_path, save_path):
    """
    단일 파일을 처리하기 위한 래퍼 함수.
    오류 발생 시 예외처리를 담당합니다.
    """
    try:
        process_single_file(file_path, save_path)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise


def process_single_file(file_path, save_path):
    """
    단일 CSV 파일에 대한 실제 로직 (대략적인 예시)
    """
    try:
        # CSV 로드
        data = pd.read_csv(file_path)

        ###################################################################
        # (1) mod_temp_list에서 평균 모듈 온도 컬럼 생성
        ###################################################################
        if 'mod_temp_list' in data.columns:
            data['mod_temp_avg'] = data['mod_temp_list'].apply(
                lambda x: np.mean([float(temp) for temp in str(x).split(',')])
            )
        else:
            data['mod_temp_avg'] = np.nan

        # altitude 컬럼 유무에 따른 file_prefix 파싱
        if 'altitude' in data.columns:
            parts = file_path.split(os.sep)
            file_name = parts[-1]
            name_parts = file_name.split('_')
            device_no = name_parts[2]
            year_month = name_parts[3][:7]
            file_prefix = f"bms_altitude_{device_no}-{year_month}-trip-"
        else:
            parts = file_path.split(os.sep)
            file_name = parts[-1]
            name_parts = file_name.split('_')
            device_no = name_parts[1]
            year_month = name_parts[2][:7]
            file_prefix = f"bms_{device_no}-{year_month}-trip-"

        # (2) device_no → vehicle_type 매핑
        vehicle_type = get_vehicle_type(device_no, vehicle_dict)

        # (3) 이미 해당 device_no, year_month 파일이 있으면 스킵
        vehicle_save_path = os.path.join(save_path, vehicle_type)
        os.makedirs(vehicle_save_path, exist_ok=True)

        existing_files = [
            f for f in os.listdir(vehicle_save_path)
            if f.startswith(file_prefix)
        ]
        if existing_files:
            print(f"Files {device_no} and {year_month} already exist in {vehicle_type} folder. Skipping.")
            return

        # time 컬럼 datetime 변환 시도
        try:
            data['time'] = pd.to_datetime(data['time'], format='%y-%m-%d %H:%M:%S')
        except ValueError:
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

        # (4) Trip 구간 분할 (chrg_cable_conn, 시간 간격)
        cut = []
        if data.loc[0, 'chrg_cable_conn'] == 0:
            cut.append(0)
        for i in range(len(data) - 1):
            if data.loc[i, 'chrg_cable_conn'] != data.loc[i + 1, 'chrg_cable_conn']:
                cut.append(i + 1)
            if (data.loc[i + 1, 'time'] - data.loc[i, 'time']).total_seconds() > 10:
                if (i + 1) not in cut:
                    cut.append(i + 1)
        if data.loc[len(data) - 1, 'chrg_cable_conn'] == 0:
            cut.append(len(data) - 1)

        cut = sorted(set(cut))

        # (5) Trip별 처리
        trip_counter = 1
        for idx in range(len(cut) - 1):
            start_idx = cut[idx]
            end_idx = cut[idx + 1] - 1

            # 주행 상태(0) 구간만 처리
            if data.loc[start_idx, 'chrg_cable_conn'] != 0:
                continue

            # 기본 trip 슬라이싱
            trip = data.loc[start_idx:end_idx, :]

            # (5-1) Trip 기본 유효성 체크
            if not check_trip_base_conditions(trip):
                continue

            # (5-2) Trip 확장 조건 + expand(앞뒤 30행)
            expanded_trip = check_time_gap_conditions(data, start_idx, end_idx)
            if expanded_trip is None:
                continue

            ###################################################################
            # (5-3) "확장된 Trip" 평균 모듈 온도도 20~28℃ 범위인지 재확인
            ###################################################################
            expanded_temp_mean = expanded_trip['mod_temp_avg'].mean()
            if not (20 <= expanded_temp_mean <= 28):
                # 조건을 만족하지 않으면 skip
                continue
            ###################################################################

            # (5-4) 최종 Trip 저장
            filename = f"{file_prefix}{trip_counter}.csv"
            expanded_trip.to_csv(os.path.join(vehicle_save_path, filename), index=False)
            print(f"Trip saved: {os.path.join(vehicle_type, filename)}")
            trip_counter += 1

    except Exception as e:
        print(f"[ERROR] {file_path} 처리 중 오류: {e}")
        return


def check_trip_base_conditions(trip):
    """
    기존에 주어진 trip 유효성 체크 로직 + 추가된 모듈 온도 조건
    """
    # 1) 빈 데이터프레임
    if trip.empty:
        return False

    # 2) 가속도 비정상
    if (trip['acceleration'] > 9.0).any():
        return False

    # 3) 주행 시간(5분 이상), 이동 거리(3km 이상)
    t = trip['time']
    t_diff = t.diff().dt.total_seconds().fillna(0)
    v = trip['speed']
    distance = (v * t_diff).cumsum().iloc[-1]

    if (t.iloc[-1] - t.iloc[0]).total_seconds() < 300:
        return False
    if distance < 3000:
        return False

    # 4) 소비 에너지 (>= 1.0 kWh)
    power = trip['Power_data']
    data_energy = (power * t_diff / 3600 / 1000).cumsum().iloc[-1]
    if data_energy < 1.0:
        return False

    # 5) 0속도 연속 300초
    zero_speed_duration = 0
    for i in range(len(trip) - 1):
        if v.iloc[i] == 0:
            zero_speed_duration += (t.iloc[i + 1] - t.iloc[i]).total_seconds()
            if zero_speed_duration >= 300:
                return False
        else:
            zero_speed_duration = 0

    # 6) 모듈 온도 조건
    # (a) Trip 최초 1분 평균 온도
    first_min_mask = (t - t.iloc[0]) <= pd.Timedelta(minutes=1)
    trip_first_min = trip.loc[first_min_mask]
    if trip_first_min.empty:
        return False  # 1분 미만이면 제외
    first_min_temp_mean = trip_first_min['mod_temp_avg'].mean()
    if not (20 <= first_min_temp_mean <= 28):
        return False

    # (b) Trip 전체 평균 온도
    whole_trip_temp_mean = trip['mod_temp_avg'].mean()
    if not (20 <= whole_trip_temp_mean <= 28):
        return False

    return True


def check_time_gap_conditions(data, start_idx, end_idx):
    """
    1) Trip 시작 시점과 이전 행의 timestamp 차이 >= 2시간(7200초)
    2) Trip 끝 시점과 다음 행의 timestamp 차이 >= 1시간(3600초)
    ---------------------------------------------------------
    조건을 만족하면, Trip 앞뒤로 30행씩 확장해서 리턴.
    만족하지 못하면 None.
    """
    if start_idx == 0 or end_idx == (len(data) - 1):
        return None

    trip_start_time = data.loc[start_idx, 'time']
    trip_end_time = data.loc[end_idx, 'time']

    prev_time = data.loc[start_idx - 1, 'time']
    next_time = data.loc[end_idx + 1, 'time']

    # (1) 2시간 이상
    if (trip_start_time - prev_time).total_seconds() < 7200:
        return None

    # (2) 1시간 이상
    if (next_time - trip_end_time).total_seconds() < 3600:
        return None

    expanded_start = max(start_idx - 30, 0)
    expanded_end = min(end_idx + 30, len(data) - 1)

    return data.loc[expanded_start:expanded_end, :]
