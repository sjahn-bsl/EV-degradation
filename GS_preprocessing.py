import os
import glob
import pandas as pd
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_data_by_vehicle(folder_path, vehicle_dict, selected_car):
    """
    주어진 folder_path에서 selected_car에 해당하는 vehicle_dict 내의 ID를 이용해
    bms 혹은 bms_altitude 파일을 재귀적으로 찾은 뒤,
    해당 파일들의 경로를 vehicle_files 딕셔너리에 담아 반환합니다.
    """
    vehicle_files = {}
    # 선택된 차량이 vehicle_dict에 없으면 처리 중단
    if selected_car not in vehicle_dict:
        print(f"Selected vehicle '{selected_car}' not found in vehicle_dict.")
        return vehicle_files

    # 선택된 차량에 대응하는 ID 리스트 가져오기
    ids = vehicle_dict[selected_car]
    all_files = []
    for vid in ids:
        # bms, bms_altitude 파일 형식으로 glob 패턴 생성
        patterns = [
            os.path.join(folder_path, f"**/bms_{vid}-*"),
            os.path.join(folder_path, f"**/bms_altitude_{vid}-*")
        ]
        # 각 패턴별로 매칭되는 모든 파일을 all_files에 추가
        for pattern in patterns:
            all_files += glob.glob(pattern, recursive=True)

    vehicle_files[selected_car] = all_files
    return vehicle_files

def get_vehicle_type(device_no, vehicle_dict):
    """
    device_no를 받아서 어떤 차종인지 반환.
    만약 vehicle_dict에 없으면 "Unknown" 반환
    """
    for vtype, dev_list in vehicle_dict.items():
        if device_no in dev_list:
            return vtype
    return "Unknown"

def process_device_folders(source_paths, destination_root, altitude=False):
    """
    주어진 source_paths(폴더 경로) 아래의 파일들을 순회하며,
    altitude 여부에 따라 다른 파일명(bms_altitude vs bms)을 필터링하고,
    디바이스 번호와 날짜정보를 이용해 destination_root 경로에 정리하는 함수입니다.
    """
    for root, dirs, files in os.walk(source_paths):
        # 폴더 내에 더 이상 하위 디렉터리가 없는지(leaf 디렉터리) 확인
        if not dirs:
            # altitude 여부에 따라 다른 파일명 필터 적용
            if altitude:
                filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' in f]
            else:
                filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' not in f]

            filtered_files.sort()

            device_no, year_month = None, None

            for file in filtered_files:
                file_path = os.path.join(root, file)
                parts = file_path.split(os.sep)
                file_name = parts[-1]
                name_parts = file_name.split('_')

                # altitude 플래그에 따라 device_no를 다르게 파싱
                device_no = name_parts[1] if not altitude else name_parts[2]
                date_parts = name_parts[2].split('-') if not altitude else name_parts[3].split('-')
                year_month = '-'.join(date_parts[:2])

                # 목적지 폴더 생성(디바이스 번호와 연-월 기준)
                save_folder = os.path.join(destination_root, device_no, year_month)
                os.makedirs(save_folder, exist_ok=True)

                destination_file_path = os.path.join(save_folder, file)

                # 파일 이동
                shutil.move(file_path, destination_file_path)
                print(f"Moved {file} to {destination_file_path}")

def delete_zero_kb_files(root_dir):
    """
    주어진 루트 디렉토리 내의 모든 하위 디렉토리를 순회하며 크기가 0KB인 파일을 삭제합니다.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                # 파일 크기가 0이면 삭제
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    print(f"삭제됨: {file_path}")
            except OSError as e:
                print(f"파일 삭제 실패: {file_path} - 오류: {e}")

def read_file_with_detected_encoding(file_path):
    """
    파일 인코딩을 추론하여 DataFrame으로 읽어들이는 함수.
    UTF-8 -> ISO-8859-1 -> 기타 등등 순으로 시도.
    """
    try:
        # 가장 먼저 UTF-8 인코딩(C engine) 시도
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # UTF-8 실패 시 ISO-8859-1 인코딩 시도
        try:
            return pd.read_csv(file_path, encoding='iso-8859-1')
        except Exception as e:
            # 둘 다 실패하면 Python engine + UTF-8 인코딩 시도
            try:
                return pd.read_csv(file_path, encoding='utf-8', engine='python')
            except Exception as e:
                print(f"Failed to read file {file_path} with Python engine due to: {e}")
                return None

def process_files(start_path, save_path, vehicle_type, altitude=False):
    """
    start_path부터 시작하여 하위 폴더(leaf) 단위로 process_folder 함수를 호출해
    CSV 파일들을 병합/처리한 뒤, save_path로 저장하는 메인 함수입니다.
    ProcessPoolExecutor를 통해 병렬 처리하며 tqdm로 진행도를 표시합니다.
    """
    # 전체 폴더 개수를 계산(하위 폴더 중 leaf 디렉터리만)
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)]) #start_path 내의 "leaf 디렉토리(하위 폴더가 없는 폴더)" 개수를 계산하는 코드

    with tqdm(total=total_folders, desc="Processing folders", unit="folder") as pbar: #total_folders 개수만큼 진행 바를 만들고, 각 폴더 처리 시 업데이트 됨.
        with ProcessPoolExecutor() as executor: # 여러 개의 폴더를 동시에 처리(병렬 처리)하여 실행 속도를 최적화
            futures = []
            # 모든 경로를 순회하며 leaf 디렉터리를 찾는다.
            for root, dirs, files in os.walk(start_path):
                if not dirs:
                    # leaf 디렉터리면 process_folder 실행을 위한 future 등록
                    futures.append(
                        executor.submit(process_folder, root, files, save_path, vehicle_type, altitude)
                    )

            # 모든 스레드의 결과 수집
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                finally:
                    pbar.update(1)

    print("모든 폴더의 파일 처리가 완료되었습니다.")

def fill_altitude(df):
    """
    altitude 컬럼에 대하여,
      1) 처음 나온 altitude 값으로 이전 구간(NaN)을 채우고
      2) 중간 구간 NaN은 선형 보간(linear interpolation)
      3) 마지막 altitude 값으로 이후 구간(NaN)을 채우는 함수
    """
    # 혹시 모를 타입 변환(문자열 등) 방지
    df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')

    # altitude에 유효한 값이 전혀 없다면 그대로 반환
    if df['altitude'].notnull().sum() == 0:
        return df

    # 처음 & 마지막으로 altitude가 유효한 값인 인덱스 찾기
    first_valid_idx = df['altitude'].first_valid_index()
    last_valid_idx = df['altitude'].last_valid_index()

    # 1) 처음 발견된 altitude 값으로 이전 구간 채우기
    first_value = df.loc[first_valid_idx, 'altitude']
    df.loc[:first_valid_idx, 'altitude'] = df.loc[:first_valid_idx, 'altitude'].fillna(first_value)

    # 2) 마지막 발견된 altitude 값으로 이후 구간 채우기
    last_value = df.loc[last_valid_idx, 'altitude']
    df.loc[last_valid_idx:, 'altitude'] = df.loc[last_valid_idx:, 'altitude'].fillna(last_value)

    # 3) 중간 구간은 linear interpolation
    df['altitude'] = df['altitude'].interpolate(method='linear')

    return df

def process_folder(root, files, save_path, vehicle_dict, altitude): #root: 현재 처리 중인 폴더 경로, files: root 폴더 내의 파일 리스트, save_path: 처리된 CSV를 저장할 폴더, vehicle_dict: 차량 번호와 모델명을 매칭하는 딕셔너리
    """
    주어진 root(leaf 디렉토리)에 있는 CSV 파일을 필터링 후,
    병합/처리하여 최종 CSV로 저장하는 함수입니다.
    altitude 여부에 따라 bms_altitude / bms 파일을 구분하여 처리합니다.
    """
    # altitude 여부에 따른 파일 필터
    if altitude:
        filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' in f] # altitude가 True이면 고도 정보가 포함된 BMS 파일만 선택
    else:
        filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' not in f] # altitude가 False이면 일반 BMS 파일만 선택

    filtered_files.sort()
    dfs = []
    device_no, year_month = None, None

    # 각 CSV 파일을 순회하며 데이터프레임 변환
    for file in filtered_files:
        file_path = os.path.join(root, file)
        parts = file_path.split(os.sep)
        file_name = parts[-1]
        name_parts = file_name.split('_')

        # altitude 여부에 따라 device_no, date_parts 위치가 다름
        device_no = name_parts[1] if not altitude else name_parts[2] # 파일명에서 device_no 추출
        date_parts = name_parts[2].split('-') if not altitude else name_parts[3].split('-')
        year_month = '-'.join(date_parts[:2]) # 파일명에서 year_month 추출

        # 단말기번호에 대응되는 모델명 가져오기 (없으면 'Unknown')
        vehicle_model = get_vehicle_type(device_no, vehicle_dict)

        save_folder = os.path.join(save_path, vehicle_model) #save_path/{차량 모델명} 형식으로 폴더를 생성하여 데이터를 저장
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        output_file_name = f"{'bms_altitude' if altitude else 'bms'}_{device_no}_{year_month}.csv"
        output_file_path = os.path.join(save_folder, output_file_name)

        # 이미 동일한 파일이 있으면 스킵
        if os.path.exists(output_file_path):
            print(f"File {output_file_name} already exists in {save_folder}. Skipping...")
            return

        # 파일 인코딩 자동 인식 후 DataFrame 로드
        df = read_file_with_detected_encoding(file_path)
        if df is not None:
            # 'Unnamed' 등 불필요한 컬럼 제거
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            # time 중복 제거
            df = df.drop_duplicates(subset='time')

            # time 컬럼을 datetime으로 변환 (여러 포맷 시도)
            date_formats = ['%y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S']
            for fmt in date_formats:
                try:
                    df['time'] = pd.to_datetime(df['time'], format=fmt, errors='raise')
                    break
                except ValueError:
                    continue
            else:
                # datetime 변환 실패 시 스킵
                print(f"Time format error in file {file_path}. Skipping this file.")
                continue

            # time 컬럼 기준 정렬
            df = df.sort_values(by='time').reset_index(drop=True)
            dfs.append(df)

    # 불러온 여러 파일들을 concat하여 최종 파일로 저장
    if dfs and device_no and year_month and not os.path.exists(output_file_path):
        combined_df = pd.concat(dfs, ignore_index=True)
        # time_diff: 인접 행 간 시간 차이(초 단위)
        combined_df['time_diff'] = combined_df['time'].diff().dt.total_seconds()

        # speed(속도, m/s) = emobility_spd(km/h) * 0.27778
        combined_df['speed'] = combined_df['emobility_spd'] * 0.27778

        # 가속도(acceleration) 계산
        combined_df['acceleration'] = combined_df['speed'].diff() / combined_df['time_diff']

        # 처음 행과 마지막 행의 가속도도 보정
        if len(combined_df) > 1:
            combined_df.at[0, 'acceleration'] = (combined_df.at[1, 'speed'] - combined_df.at[0, 'speed']) / \
                                                combined_df.at[1, 'time_diff']
            combined_df.at[len(combined_df) - 1, 'acceleration'] = (
                combined_df.at[len(combined_df) - 1, 'speed'] - combined_df.at[len(combined_df) - 2, 'speed']
            ) / combined_df.at[len(combined_df) - 1, 'time_diff']

        combined_df['acceleration'] = combined_df['acceleration'].fillna(0)

        # Power_data = pack_volt * pack_current
        combined_df['Power_data'] = combined_df['pack_volt'] * combined_df['pack_current']

        if altitude:
            # altitude 보정(결측치 보간) 진행
            combined_df = fill_altitude(combined_df)
            # altitude 컬럼 포함해서 저장할 컬럼 선택
            data_save = combined_df[
                ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp',
                 'chrg_cnt', 'chrg_cnt_q', 'cumul_energy_chrgd',
                 'cumul_energy_chrgd_q', 'mod_temp_list', 'odometer',
                 'op_time', 'soc', 'soh', 'chrg_cable_conn',
                 'altitude', 'cell_volt_list', 'min_deter',
                 'pack_volt', 'pack_current', 'Power_data']
            ].copy()
        else:
            # altitude 컬럼 없는 경우
            data_save = combined_df[
                ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp',
                 'chrg_cnt', 'chrg_cnt_q', 'cumul_energy_chrgd',
                 'cumul_energy_chrgd_q', 'mod_temp_list', 'odometer',
                 'op_time', 'soc', 'soh', 'chrg_cable_conn',
                 'pack_volt', 'cell_volt_list', 'min_deter',
                 'pack_current', 'Power_data']
            ].copy()

        # 최종 CSV 저장
        data_save.to_csv(output_file_path, index=False)

def process_files_trip_by_trip(start_path, save_path): #save_path: Trip 단위로 분할된 데이터를 저장할 폴더 경로
    """
    CSV 파일을 trip 단위로 분할/저장하는 메인 함수.
    start_path(월별 병합된 BMS 데이터)부터 하위 디렉토리를 탐색하며 모든 CSV 파일을 찾고,
    각 파일에 대해 process_wrapper 함수를 호출합니다.
    """
    # 전체 CSV 파일 개수 파악
    csv_files = [os.path.join(root, file)
                 for root, _, files in os.walk(start_path) # os.walk(start_path)를 사용하여 start_path내의 모든 .csv파일을 리스트에 저장
                 for file in files if file.endswith('.csv')]
    total_files = len(csv_files) # csv 파일 개수를 total_files 변수에 저장(진행 바 표시를 위해)

    # tqdm를 이용해 진행 상황 출력
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar: #Processing files라는 설명과 함께 진행 바를 출력
        # 파일들을 병렬 처리
        future_to_file = {}
        with ProcessPoolExecutor() as executor: # 여러 개의 CSV 파일들을 동시에 처리
            for file_path in csv_files:
                future = executor.submit(process_wrapper, file_path, save_path) # 개별 파일을 process_wrapper()에 전달하여 처리
                future_to_file[future] = file_path

            for future in as_completed(future_to_file): # 처리가 완료된 파일들을 순차적으로 확인
                file_path = future_to_file[future]
                try:
                    future.result() #결과 가져오기
                except Exception as exc:
                    print(f'File {file_path} generated an exception: {exc}')
                finally:
                    pbar.update(1) # 진행 바 업데이트

    print("Processing complete") # 모든 파일이 Trip 단위로 처리 완료되었음을 알리는 메세지 출력

def process_wrapper(file_path, save_path): #file_path를 process_single_file()에 전달하여 Trip 단위로 변환 및 저장
    """
    단일 파일을 처리하기 위한 래퍼 함수.
    오류 발생 시 예외처리를 담당합니다.
    """
    try:
        process_single_file(file_path, save_path) # process_single_file 호출
    except Exception as e: # 실행 도중 오류가 발생할 경우 예외 처리
        print(f"Error processing file {file_path}: {e}")
        raise # raise를 사용하여 예외를 다시 발생시킴

def process_single_file(file_path, save_path):
    """
    단일 CSV 파일에 대한 실제 로직:
    1) altitude 컬럼 유무에 따라 device_no, year_month 파싱
    2) 이미 동일 device_no & year_month로 만들어진 파일이 있으면 스킵
    3) CSV 파일을 일정 조건(trip)별로 쪼갬
    4) 조건이 충족되는 trip만 CSV로 저장
    """
    try:
        # CSV 로드 (기본 pandas.read_csv, 인코딩 가정)
        data = pd.read_csv(file_path)

        # altitude 컬럼 존재 여부에 따라 파일명 규칙이 달라짐, device_no와 year_month를 파일명에서 추출
        if 'altitude' in data.columns:
            parts = file_path.split(os.sep)
            file_name = parts[-1]
            name_parts = file_name.split('_')
            device_no = name_parts[2]
            year_month = name_parts[3][:7]
        else:
            parts = file_path.split(os.sep)
            file_name = parts[-1]
            name_parts = file_name.split('_')
            device_no = name_parts[1]
            year_month = name_parts[2][:7]

        # 이미 해당 device_no와 year_month 관련 파일이 존재하면 스킵
        altitude_file_pattern = f"bms_altitude_{device_no}-{year_month}-trip-"
        non_altitude_file_pattern = f"bms_{device_no}-{year_month}-trip-"
        existing_files = [
            f for f in os.listdir(save_path)
            if f.startswith(altitude_file_pattern) or f.startswith(non_altitude_file_pattern) # save_path내에서 device_no와 year_month가 동일한 Trip 파일이 이미 존재하는지 확인
        ]

        if existing_files:
            print(f"Files {device_no} and {year_month} already exist. Skipping all related files.")
            return

        cut = []

        # 1) 충전 케이블 연결 상태(chrg_cable_conn)가 0 -> 1 혹은 1 -> 0으로 변할 때 trip 분리
        if data.loc[0, 'chrg_cable_conn'] == 0:
            cut.append(0)
        for i in range(len(data) - 1):
            if data.loc[i, 'chrg_cable_conn'] != data.loc[i + 1, 'chrg_cable_conn']:
                cut.append(i + 1)
        if data.loc[len(data) - 1, 'chrg_cable_conn'] == 0:
            cut.append(len(data) - 1)

        # 2) 시간 간격이 10초 이상 차이나면 다른 trip으로 인식
        cut_time = pd.Timedelta(seconds=10)

        # time 컬럼 datetime 변환 시도
        try:
            data['time'] = pd.to_datetime(data['time'], format='%y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"Date format error in file {file_path}: {e}")
                return

        # 인접 행 간 time 차이가 10초 이상이면 trip 분할
        for i in range(len(data) - 1):
            if data.loc[i + 1, 'time'] - data.loc[i, 'time'] > cut_time:
                cut.append(i + 1)

        # 중복된 Trip 분할 지점 제거 후 시간 순서대로 정렬
        cut = list(set(cut))
        cut.sort()

        if not cut:
            print(f"No cuts found in file: {file_path}")
            return None

        trip_counter = 1  # trip 번호(1부터 시작)

        # trip 구간별로 잘라내어 저장
        for i in range(len(cut) - 1):
            # 케이블 연결이 0(주행 상태)인 구간만 trip으로 인식
            if data.loc[cut[i], 'chrg_cable_conn'] == 0:
                trip = data.loc[cut[i]:cut[i + 1] - 1, :]

                # trip 조건(거리, 시간, 에너지 등) 검증
                if not check_trip_conditions(trip):
                    continue

                # 파일 이름 생성(bms_altitude 또는 bms)
                if 'altitude' in data.columns:
                    filename = f"bms_altitude_{device_no}-{year_month}-trip-{trip_counter}.csv"
                else:
                    filename = f"bms_{device_no}-{year_month}-trip-{trip_counter}.csv"

                # 유효한 trip 데이터만 save_path에 csv 파일로 저장
                os.makedirs(save_path, exist_ok=True)
                trip.to_csv(os.path.join(save_path, filename), index=False)
                trip_counter += 1

        # 마지막 구간 처리
        if cut:
            trip = data.loc[cut[-1]:, :]

            if check_trip_conditions(trip):
                duration = trip['time'].iloc[-1] - trip['time'].iloc[0]
                # 5분 이상 + 케이블 연결 0인 경우에만 저장
                if duration >= pd.Timedelta(minutes=5) and data.loc[cut[-1], 'chrg_cable_conn'] == 0:
                    if 'altitude' in data.columns:
                        filename = f"bms_altitude_{device_no}-{year_month}-trip-{trip_counter}.csv"
                    else:
                        filename = f"bms_{device_no}-{year_month}-trip-{trip_counter}.csv"

                    print(f"Files {device_no} and {year_month} successfully processed.")
                    os.makedirs(save_path, exist_ok=True)
                    trip.to_csv(os.path.join(save_path, filename), index=False)

    except Exception as e:
        # 오류가 발생하면 문제 파일명을 출력
        print(f"[ERROR] {file_path} 처리 중 오류가 발생했습니다: {e}")
        return


def check_trip_conditions(trip):
    """
    trip의 각종 조건(가속도, 총 운행 시간, 거리, 소비 에너지, 정지 상태 지속 시간 등)을 확인하여
    유효한 trip인지 True/False로 반환합니다.
    """
    # 빈 DataFrame이면 False
    if trip.empty:
        return False

    # 가속도가 9.0m/s^2 이상 발생하는 행이 있으면 비정상으로 간주
    if (trip['acceleration'] > 9.0).any():
        return False

    # time 컬럼 datetime 변환 확인
    v = trip['speed']
    date_formats = ['%y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S']
    for date_format in date_formats:
        try:
            t = pd.to_datetime(trip['time'], format=date_format)
            break
        except ValueError:
            continue
    else:
        print("Date format error in trip conditions")
        return False

    # 시간차(초)를 이용해 이동 거리 계산 (속도= m/s)
    t_diff = t.diff().dt.total_seconds().fillna(0)
    distance = (v * t_diff).cumsum().iloc[-1]

    # 운행 시간 300초, 이동거리 3000m, 소비 에너지 1.0kWh 이상이어야 유효
    time_limit = 300
    distance_limit = 3000
    energy_limit = 1.0

    if (t.iloc[-1] - t.iloc[0]).total_seconds() < time_limit or distance < distance_limit:
        return False

    # Power_data를 이용해 소비 에너지(kWh) 계산
    data_energy = (trip['Power_data'] * t_diff / 3600 / 1000).cumsum().iloc[-1]
    if data_energy < energy_limit:
        return False

    # 정지 상태(속도=0)가 연속 300초 이상이면 비정상으로 처리
    zero_speed_duration = 0
    for i in range(len(trip) - 1):
        if trip['speed'].iloc[i] == 0:
            zero_speed_duration += (t.iloc[i + 1] - t.iloc[i]).total_seconds()
            if zero_speed_duration >= 300:
                return False
        else:
            zero_speed_duration = 0

    return True