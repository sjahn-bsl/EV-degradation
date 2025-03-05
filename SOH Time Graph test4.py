import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.dates as mdates  # 날짜 포맷을 위한 모듈 추가
import re
from tqdm import tqdm

# PyCharm 백엔드 문제 해결
matplotlib.use('TkAgg')

# 저장할 폴더 경로
output_folder = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250224"
output_file = os.path.join(output_folder, "SOH vs Time 40% Graph 01241248927 & 01241248924.png")  # 저장할 파일 경로

# SOH 정보 파일 로드
soh_results_path = os.path.join(output_folder, "BMS_SOC SOH 40%_results.csv")
soh_df = pd.read_csv(soh_results_path)

# 특정 단말기 리스트 (두 개의 단말기 선택)
target_device_ids = ["01241248927", "01241248924"]
filtered_files = soh_df[soh_df["file"].str.contains('|'.join(target_device_ids))]  # 두 개의 단말기 포함된 파일 선택

# SOH 값 매핑
file_names = filtered_files["file"].tolist()
soh_values = dict(zip(filtered_files["file"], filtered_files["SOH (%)"]))


# trip 번호 추출 함수
def extract_trip_number(file_name):
    match = re.search(r"trip-(\d+)", file_name)
    return int(match.group(1)) if match else None


# 데이터 폴더
trip_data_folder = r"D:\SamsungSTF\Processed_Data\TripByTrip"

# 색상 지정
device_colors = {
    "01241248927": "blue",
    "01241248924": "red"
}

# 데이터 로딩
trip_data = []
print(f"단말기 {target_device_ids}의 데이터 로딩 중...")

for file_name in tqdm(file_names, desc="Loading Trip Files", unit="file"):
    file_path = os.path.join(trip_data_folder, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, usecols=["time"])
        if "time" in df.columns and len(df) > 1:
            df = df.iloc[[0, -1]]
            df["SOH"] = soh_values[file_name]
            df["file"] = file_name
            df["trip_number"] = extract_trip_number(file_name)  # trip 번호 추가

            # 단말기 ID 추출
            for dev_id in target_device_ids:
                if dev_id in file_name:
                    df["device_id"] = dev_id
                    trip_data.append(df)
        else:
            print(f"time 컬럼 없음 또는 데이터 부족: {file_name}")
    else:
        print(f"파일 없음: {file_name}")

# 데이터프레임 병합
if trip_data:
    final_df = pd.concat(trip_data, ignore_index=True)
    print(f"데이터 크기: {final_df.shape}")

    # NaT 값 제거 후 정렬
    final_df["time"] = pd.to_datetime(final_df["time"], errors='coerce')
    final_df = final_df.dropna(subset=["time"])
    final_df = final_df.sort_values(by=["device_id", "time"])

    # 그래프 그리기
    plt.figure(figsize=(20, 10))
    print(f"단말기 {target_device_ids} 그래프 그리는 중...")

    # 범례 정보 저장할 리스트
    legend_labels = []

    # 두 개의 단말기 각각 다른 색상으로 그래프 표시
    for device_id, subset in final_df.groupby("device_id"):
        color = device_colors.get(device_id, "black")  # 지정된 색상, 없으면 검은색
        subset = subset.sort_values(by="time")

        for _, row in subset.iterrows():
            trip_label = f"Device {device_id} - Trip {row['trip_number']}"
            if trip_label not in legend_labels:
                plt.scatter(row["time"], row["SOH"], color=color, marker='o', s=20, label=trip_label)
                legend_labels.append(trip_label)
            else:
                plt.scatter(row["time"], row["SOH"], color=color, marker='o', s=20)

        plt.plot(subset["time"], subset["SOH"], color=color, linestyle='-', alpha=1.0, linewidth=1.5)

    # x축 눈금 조정
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    plt.xlabel("Time")
    plt.ylabel("SOH (%)")
    plt.title(f"SOH vs Time for Devices {', '.join(target_device_ids)}")

    # 범례 추가 (trip 정보를 포함)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize="small", title="Device ID - Trip Number", frameon=True, shadow=True)

    plt.grid()

    # 그래프 저장
    print(f"그래프 저장 중... ({output_file})")
    plt.savefig(output_file, bbox_inches="tight")
    print("그래프 저장 완료")

    # 그래프 출력
    print("그래프 출력 중...")
    plt.show()
    print("그래프 출력 완료")

else:
    print(f"유효한 데이터가 없습니다. 단말기 {target_device_ids}에 해당하는 파일이 없거나, 데이터가 부족합니다.")
