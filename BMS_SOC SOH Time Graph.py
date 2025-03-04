import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.dates as mdates  # 날짜 포맷을 위한 모듈 추가
from tqdm import tqdm

# PyCharm 백엔드 문제 해결
matplotlib.use('TkAgg')

# 저장할 폴더 경로
output_folder = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224"
# 저장할 파일 경로 변경
output_file = os.path.join(output_folder, "SOH vs Time 20% Graph.png")  # 저장할 파일 경로

# SOH 정보 파일 로드
soh_results_path = os.path.join(output_folder, "BMS_SOC SOH 40%_results.csv")
soh_df = pd.read_csv(soh_results_path)

# file_name 목록 가져오기
file_names = soh_df["file"].tolist()
soh_values = dict(zip(soh_df["file"], soh_df["SOH (%)"]))  # SOH 값 매핑
# 단말기번호 추출 (파일명에 따라 다르게 처리)
device_ids = {}
for file in file_names:
    parts = file.split("_")  # 언더스코어 기준으로 나누기

    if parts[1] == "altitude":
        device_id = parts[2]  # "bms_altitude_단말기번호_..." → 3번째 요소가 단말기 번호
    else:
        device_id = parts[1]  # "bms_단말기번호_..." → 2번째 요소가 단말기 번호

    device_ids[file] = device_id  # 딕셔너리에 저장

# 단말기번호별 색상을 위한 컬러맵 생성
unique_devices = sorted(set(device_ids.values()))  # 고유한 단말기번호 리스트를 정렬
device_colors = {dev: plt.cm.jet(i / (len(unique_devices) - 1)) for i, dev in enumerate(unique_devices)}

# 데이터 폴더
trip_data_folder = r"D:\SamsungSTF\Processed_Data\TripByTrip"

# 데이터 로딩 (time 데이터의 시작과 끝만 가져옴)
trip_data = []
print("Trip 데이터 로딩 중...")
for file_name in tqdm(file_names, desc="Loading Trip Files", unit="file"):
    file_path = os.path.join(trip_data_folder, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, usecols=["time"])  # 필요한 컬럼만 로드
        if "time" in df.columns and len(df) > 1:
            df = df.iloc[[0, -1]]  # 첫 번째와 마지막 데이터만 가져오기
            df["SOH"] = soh_values[file_name]
            df["file"] = file_name
            df["device_id"] = device_ids[file_name]
            trip_data.append(df)
        else:
            print(f"time 컬럼 없음 또는 데이터 부족: {file_name}")
    else:
        print(f"파일 없음: {file_name}")

# 데이터프레임 병합
if trip_data:
    final_df = pd.concat(trip_data, ignore_index=True)
    print(f"데이터 크기: {final_df.shape}")  # 데이터 개수 확인

    # time 컬럼을 datetime 형식으로 변환 후 정렬
    final_df["time"] = pd.to_datetime(final_df["time"])
    final_df = final_df.sort_values(by=["device_id", "time"])  # 같은 단말기번호끼리 정렬

    # 그래프 크기 조정 (가로, 세로)
    plt.figure(figsize=(20, 10))
    print("그래프 그리는 중...")

    for device_id, subset in final_df.groupby("device_id"):
        color = device_colors[device_id]  # 같은 단말기번호끼리 같은 색상 지정
        subset = subset.sort_values(by="time")  # 시간순 정렬

        # 🔹 **점과 선을 함께 그리기**
        plt.plot(subset["time"], subset["SOH"], color=color, linestyle='-', alpha=0.6, linewidth=1,
                 label=device_id)  # 선 연결
        plt.scatter(subset["time"], subset["SOH"], color=color, alpha=0.8, marker='o')  # 점 표시

    # x축 눈금 개수 조정 (7일 간격으로 설정)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # 7일 단위로 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # 날짜 포맷 변경
    plt.xticks(rotation=45)  # x축 레이블 45도 회전

    plt.xlabel("Time")
    plt.ylabel("SOH (%)")
    plt.title("SOH vs Time (Grouped by Device ID, Start & End Points)")

    # 🔹 **모든 범례를 표시하도록 설정**
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small", title="Device ID", ncol=5)  # ncol 값 증가

    plt.grid()

    # 그래프 저장 경로 설정
    print(f"그래프 저장 중... ({output_file})")
    plt.savefig(output_file, bbox_inches="tight")  # 지정된 폴더에 저장
    print("그래프 저장 완료")

    # 그래프 출력
    print("그래프 출력 중...")
    plt.show()
    print("그래프 출력 완료")

else:
    print("유효한 데이터가 없습니다.")
