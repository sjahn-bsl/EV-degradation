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

def process_trip_by_trip_soc_2hr_with_charging(start_path, save_path, save_path2):
    """
    CSV 파일을 trip 단위로 분할/저장하는 메인 함수 (SOC 추정용 2시간 조건)
    start_path에서 모든 csv파일을 탐색, 각 csv 파일을 Trip 단위로 변환 후 save_path에 저장
    """
    csv_files = [os.path.join(root, file) #파일 경로를 절대 경로로 변환
                 for root, _, files in os.walk(start_path) #os.walk(start_path): start_path 내부의 모든 디렉터리 탐색
                 for file in files if file.endswith('.csv')] # csv 파일만 탐색
    total_files = len(csv_files) # 총 csv 파일 개수를 저장

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar: #진행률바를 출력하는 라이브러리
        future_to_file = {} # future 객체(비동기 실행 객체)와 파일 경로를 매핑, 실행된 각 프로세스의 결과를 추적
        with ProcessPoolExecutor() as executor: # 병렬 처리(Python 멀티프로세싱) 지원
            for file_path in csv_files: # 각 csv파일을 병렬로 처리
                future = executor.submit(process_wrapper_2hr_with_charging, file_path, save_path, save_path2) # 각 csv 파일을 process_wrapper() 함수에 전달하여 비동기 실행, 비동기 실행=여러 파일을 동시에 처리
                future_to_file[future] = file_path  #future 객체와 해당 파일 경로를 저장(오류 발생 시 어떤 파일인지 추적 가능)

            for future in as_completed(future_to_file): # 병렬 처리된 파일들의 결과 확인, as_completed(future_to_file): 모든 병렬 작업이 완료될 때까지 대기
                file_path = future_to_file[future]
                try:
                    future.result() #비동기 작업의 실행 결과를 가져옴
                except Exception as exc:
                    print(f'File {file_path} generated an exception: {exc}') # 특정파일에서 오류가 발생하면 오류 메세지를 출력
                finally:
                    pbar.update(1) # tqdm 진행률 바를 1 증가시킴

    print("Processing complete")

def process_wrapper_2hr_with_charging(file_path, save_path, save_path2):
    """
    단일 파일을 처리하는 래퍼 함수 (SOC 추정용 2시간 확장 조건 적용)
    """
    try:
        process_single_file_2hr_with_charging(file_path, save_path, save_path2) # process_single_file 호출
    except Exception as e: # 실행 도중 오류가 발생할 경우 예외 처리
        print(f"Error processing file {file_path}: {e}")
        raise # raise를 사용하여 예외를 다시 발생시킴

def process_single_file_2hr_with_charging(file_path, save_path, save_path2):
    """
    단일 CSV 파일에 대한 실제 로직 (대략적인 예시)
    1. csv파일을 로드 2. mod_temp_list에서 평균 모듈 온도 계산 3. device_no, year_month 추출 4. 이미 존재하는 파일인지 확인 후 중복 저장 방지
    5. chrg_cable_conn 및 시간간격(10초 이상)을 기준으로 Trip 분할 6. Trip 유효성 검사 7. Trip 확장 조건 확인 후 저장
    """
    try: #예외 발생 시 프로그램이 중단되지 않도록 처리
        # CSV 파일을 Pandas 데이터프레임을 불러옴
        data = pd.read_csv(file_path)

        ###################################################################
        # (1) mod_temp_list에서 평균 모듈 온도 컬럼 생성
        ###################################################################
        if 'mod_temp_list' in data.columns: #mod_temp_list는 여러 개의 모듈 온도를 콤마로 구분한 문자열.
            data['mod_temp_avg'] = data['mod_temp_list'].apply(
                lambda x: np.mean([float(temp) for temp in str(x).split(',')]) # 각각의 값을 실수(float)로 변환 후 평균값을 저장
            )
        else:
            data['mod_temp_avg'] = np.nan

        # altitude 컬럼 유무에 따른 file_prefix 파싱, 목적: 파일 구분 (고도 포함 데이터 vs 일반 데이터)
        if 'altitude' in data.columns: #altitude 컬럼이 있으면 파일명에 bms_altitude_ 접두사 추가
            parts = file_path.split(os.sep)
            file_name = parts[-1]
            name_parts = file_name.split('_')
            device_no = name_parts[2]
            year_month = name_parts[3][:7]
            file_prefix = f"bms_altitude_{device_no}_{year_month}-trip-"
        else: # altitude 컬럼이 없으면 bms_접두사 사용
            parts = file_path.split(os.sep)
            file_name = parts[-1]
            name_parts = file_name.split('_')
            device_no = name_parts[1]
            year_month = name_parts[2][:7]
            file_prefix = f"bms_{device_no}_{year_month}-trip-"

        # (2) device_no → vehicle_type 매핑
        vehicle_type = get_vehicle_type(device_no, vehicle_dict) # vehicle_dict를 참고하여 device_no에 해당하는 차량 모델명 가져옴.

        # (3) 이미 해당 device_no, year_month 파일이 있으면 스킵
        vehicle_save_path = os.path.join(save_path, vehicle_type)
        charging_save_path = os.path.join(save_path2, vehicle_type)
        os.makedirs(vehicle_save_path, exist_ok=True)
        os.makedirs(charging_save_path, exist_ok=True)

        existing_drive_files = [
            f for f in os.listdir(vehicle_save_path)
            if f.startswith(file_prefix) and not f.startswith(file_prefix + "charging-")
        ]
        existing_chrg_files = [
            f for f in os.listdir(charging_save_path)
            if f.startswith(file_prefix + "charging-")
        ]

        skip_drive = bool(existing_drive_files)
        skip_chrg = bool(existing_chrg_files)

        if skip_drive:
            print(f"[SKIP] Driving Trips already exist for {device_no}, {year_month} in {vehicle_type}")
        if skip_chrg:
            print(f"[SKIP] Charging Trips already exist for {device_no}, {year_month} in {vehicle_type}")

        # time 컬럼 datetime 변환 시도
        try:
            data['time'] = pd.to_datetime(data['time'], format='%y-%m-%d %H:%M:%S')
        except ValueError:
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

        # (4) Trip 구간 분할 (chrg_cable_conn이 변경되는 시점, 시간 간격이 10초 이상 차이나는 경우)
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

        # (5) Trip별 처리, Trip 유효성 검사 및 저장
        trip_counter = 1 # 저장할 Trip의 순번을 관리
        charging_trip_counter = 1 # 저장할 충전 Trip의 순번을 관리

        for idx in range(len(cut) - 1): # 분할된 Trip 개수만큼 반복
            start_idx = cut[idx] # 현재 Trip의 시작 인덱스
            end_idx = cut[idx + 1] - 1 # 현재 Trip의 종료 인덱스 (다음 cut 인덱스의 바로 전)

            # 기본 trip 슬라이싱
            trip = data.loc[start_idx:end_idx, :] # Trip 데이터 추출

            # 충전 상태로 시작하는 Trip은 별도 저장
            if data.loc[start_idx, 'chrg_cable_conn'] != 0:
                if skip_chrg:
                    continue
                # 충전 조건 검사
                if not check_charge_conditions(trip):
                    continue
                charging_filename = f"{file_prefix}charging-{charging_trip_counter}.csv"
                trip.to_csv(os.path.join(charging_save_path, charging_filename), index=False)
                print(f"Charging Trip saved: {os.path.join(vehicle_type, charging_filename)}")
                charging_trip_counter += 1
                continue

            if skip_drive:
                continue

            # (5-1) Trip 기본 유효성 체크
            if not check_trip_base_conditions_2hr_with_charging(trip):
                continue # 유효하지 않은 Trip이면 처리하지 않고 넘어감

            # (5-2) Trip 확장 조건 + expand(앞뒤 30행)
            expanded_trip = check_time_gap_conditions_2hr_with_charging(data, start_idx, end_idx)
            if expanded_trip is None:
                continue # 확장 불가능한 Trip은 저장하지 않음

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

def check_charge_conditions(trip):
    """
    주어진 trip 데이터에서 충전 상태 조건을 만족하는지 검사
    기준:
    1. pack_current < 0 (충전 전류)
    2. speed == 0 (정차 상태)
    3. 해당 상태가 일정 시간 이상 지속됨 (5분 이상)
    4. 일시적인 충전 중단(1분 이하)은 무시
    """
    try:
        trip = trip.copy().reset_index(drop=True)
        trip['charging'] = 0

        # datetime 변환 보장
        if not pd.api.types.is_datetime64_any_dtype(trip['time']):
            trip['time'] = pd.to_datetime(trip['time'], format='%y-%m-%d %H:%M:%S', errors='coerce')
            trip['time'] = pd.to_datetime(trip['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # 충전 상태 조건 설정
        trip['charging'] = ((trip['pack_current'] < 0) & (trip['speed'] == 0)).astype(int)

        # charging 상태 변화 포착
        change_points = [0]
        for i in range(1, len(trip)):
            if trip.loc[i, 'charging'] != trip.loc[i - 1, 'charging']:
                change_points.append(i)
        change_points.append(len(trip) - 1)

        # 일시적 공백 무시 (5분 이내 gap은 연결)
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            if trip.loc[start, 'charging'] == 0:
                time_diff = trip.loc[end, 'time'] - trip.loc[start, 'time'] # 시간 변화량 계산
                pack_volt_diff = abs(trip.loc[end, 'pack_volt'] - trip.loc[start, 'pack_volt']) # pack_volt 변화량 계산
                if time_diff <= pd.Timedelta(minutes=5) and pack_volt_diff < 20: # 조건: 5분 이하이고, 전압 변화가 20V 미만일 때만 붙이기
                    trip.loc[start:end, 'charging'] = 1

        # 5분 이상 충전 지속 구간 존재하는지 검사
        trip['charging_shift'] = trip['charging'].shift().fillna(0)
        trip['change'] = trip['charging'] != trip['charging_shift']
        change_indices = trip.index[trip['change']].tolist() + [trip.index[-1]]

        for i in range(len(change_indices) - 1):
            start = change_indices[i]
            end = change_indices[i + 1]
            if trip.loc[start, 'charging'] == 1:
                duration = trip.loc[end, 'time'] - trip.loc[start, 'time']
                if duration >= pd.Timedelta(minutes=5):
                    return True  # 유효한 충전 구간 존재

        return False

    except Exception as e:
        import traceback
        print("[ERROR] check_charge_conditions() 실패:")
        traceback.print_exc()
        return False


def check_trip_base_conditions_2hr_with_charging(trip):
    """
    기존에 주어진 trip 유효성 체크 로직 + 추가된 모듈 온도 조건: Trip 시작 1분간 평균 온도 및 전체 평균 온도
    """
    # 1) 빈 데이터프레임 체크, Trip이 생성되지 않았거나, 필터링 과정에서 데이터가 전부 삭제된 경우를 처리
    if trip.empty:
        return False

    # 2) 가속도 비정상, EV에서 비정상적인 급가속은 배터리 열화 원인이 될 가능성이 큼
    if (trip['acceleration'] > 9.0).any(): # 9.0m/s^2를 초과하는 행이 하나라도 있으면 무효한 Trip
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

    # 6) 모듈 온도 조건, Trip이 시작할 때 배터리 온도가 정상적인 범위인지 확인
    # (a) Trip 최초 1분 평균 온도
    first_min_mask = (t - t.iloc[0]) <= pd.Timedelta(minutes=1)
    trip_first_min = trip.loc[first_min_mask]
    if trip_first_min.empty:
        return False  # 1분 미만이면 제외
    first_min_temp_mean = trip_first_min['mod_temp_avg'].mean() #평균 온도가 20~28도 범위 이내에 있어야 함
    if not (20 <= first_min_temp_mean <= 28):
        return False

    # (b) Trip 전체 평균 온도
    whole_trip_temp_mean = trip['mod_temp_avg'].mean() # Trip 전체의 평균값이 20~28도 범위 내에 있어야 함
    if not (20 <= whole_trip_temp_mean <= 28):
        return False

    return True

def check_time_gap_conditions_2hr_with_charging(data, start_idx, end_idx):
    """
    1) 24시간(86400초) >= Trip 시작 시점과 이전 행의 timestamp 차이 >= 2시간(7200초)
    2) 24시간(86400초) >= Trip 끝 시점과 다음 행의 timestamp 차이 >= 2시간(7200초)

        단, 시간 간격이 24시간 이상이더라도 아래 예외 조건을 만족하면 확장 허용:
        - 조건1 예외: 31행과 30행의 SOC 차이가 ≤ 1%
        - 조건2 예외: -31행과 -32행의 SOC 차이가 ≤ 1%
    ---------------------------------------------------------
    조건을 만족하면, Trip 앞뒤로 30행씩 확장해서 리턴.
    만족하지 못하면 None.
    """
    if start_idx == 0 or end_idx == (len(data) - 1): #Trip이 파일의 첫번째 또는 마지막 행과 연결된 경우 확장하지 않음.
        return None
    # Trip 시작/종료 시점의 timestamp 가져오기
    trip_start_time = data.loc[start_idx, 'time']
    trip_end_time = data.loc[end_idx, 'time']
    # 이전 및 다음 timestamp 가져오기, 이전/다음 데이터와 비교해서 Trip의 시작/종료가 충분히 떨어져 있는지 확인
    prev_time = data.loc[start_idx - 1, 'time']
    next_time = data.loc[end_idx + 1, 'time']
    prev_gap = (trip_start_time - prev_time).total_seconds()
    next_gap = (next_time - trip_end_time).total_seconds()

    # SOC 예외 조건 준비 (전제: 'soc' 컬럼이 존재하고 길이가 충분함)
    try:
        soc_diff_prev = abs(data.loc[start_idx, 'soc'] - data.loc[start_idx-1, 'soc'])
    except:
        soc_diff_prev = 999  # 비교 불가능하면 큰 값으로 간주

    try:
        soc_diff_next = abs(data.loc[end_idx+1, 'soc'] - data.loc[end_idx, 'soc'])
    except:
        soc_diff_next = 999

    # 이전 구간 검사 (2시간 이상, 24시간 미만 또는 예외 허용)
    if not (7200 <= prev_gap < 86400 or (prev_gap >= 86400 and soc_diff_prev <= 0.5)):
        return None

    # 다음 구간 검사 (2시간 이상, 24시간 미만 또는 예외 허용)
    if not (7200 <= next_gap < 86400 or (next_gap >= 86400 and soc_diff_next <= 0.5)):
        return None

    # Trip을 확장하여 더 많은 데이터 확보
    expanded_start = max(start_idx - 30, 0)
    expanded_end = min(end_idx + 30, len(data) - 1)

    return data.loc[expanded_start:expanded_end, :]
