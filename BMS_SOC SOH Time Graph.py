import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.dates as mdates  # 날짜 포맷을 위한 모듈 추가
from tqdm import tqdm

# PyCharm 백엔드 문제 해결
matplotlib.use('TkAgg')

# SOH 정보 파일 로드
soh_results_path = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224\BMS_SOC SOH 40%_results.csv"
soh_df = pd.read_csv(soh_results_path)

# file_name 목록 가져오기
file_names = soh_df["file"].tolist()
soh_values = dict(zip(soh_df["file"], soh_df["SOH (%)"]))  # SOH 값 매핑
device_ids = {file: file.split("_")[1] for file in file_names}  # 단말기번호 추출

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
    final_df = final_df.sort_values(by="time")

    # 그래프 그리기
    plt.figure(figsize=(16, 6))
    print("그래프 그리는 중...")

    for device_id, subset in final_df.groupby("device_id"):
        color = device_colors[device_id]  # 같은 단말기번호끼리 같은 색상 지정
        plt.plot(subset["time"], subset["SOH"], label=device_id, alpha=0.6, color=color, marker='o', linestyle='-')

    # ✅ x축 눈금 개수 조정 (하루 간격으로 설정)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # 하루 단위로 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # 날짜 포맷 변경
    plt.xticks(rotation=45)  # x축 레이블 45도 회전

    # ✅ 너무 많은 눈금이 생성되는 경우 자동 최적화
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xlabel("Time")
    plt.ylabel("SOH (%)")
    plt.title("SOH vs Time (Grouped by Device ID, Start & End Points)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="x-small", title="Device ID", ncol=3)
    plt.grid()

    print("그래프 저장 중...")
    plt.savefig("optimized_output.png", bbox_inches="tight")  # 범례가 잘리지 않도록 설정
    print("그래프 저장 완료")

    # 그래프 출력
    print("그래프 출력 중...")
    plt.show()
    print("그래프 출력 완료")

else:
    print("유효한 데이터가 없습니다.")
