import os
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 파일 및 출력 폴더 설정
base_folder = r"C:\Users\6211s\OneDrive\Desktop\kentech\EV 열화 인자 분석\250308"
SOC_deltas = ["20%", "30%", "40%"]  # 다양한 ΔSOC 값

# 폴더 생성
os.makedirs(base_folder, exist_ok=True)

# ΔSOC별 데이터 처리 및 그래프 생성
for delta in SOC_deltas:
    input_file = os.path.join(base_folder, f"BMS_SOC SOH results2 {delta}.csv")
    output_file = os.path.join(base_folder, f"SOH_cal vs Odometer {delta}.png")

    # 데이터 로드
    df = pd.read_csv(input_file)

    # 데이터 정렬
    df = df.sort_values(by=["device_id", "odometer (km)"])

    # 단말기별 그래프 생성
    unique_devices = df["device_id"].unique()
    total_devices = len(unique_devices)  # 총 단말기 개수 계산

    plt.figure(figsize=(30, 10))

    for device_id in unique_devices:
        subset = df[df["device_id"] == device_id]
        plt.plot(subset["odometer (km)"], subset["SOH_cal (%)"], marker='o', linestyle='-', label=f"Device {device_id}")

    # 그래프 설정
    plt.xlabel("Odometer (km)")
    plt.ylabel("SOH_cal (%)")
    plt.title(f"SOH_cal vs Odometer per Device (Total Devices: {total_devices}, ΔSOC={delta})")  # ΔSOC 정보 추가
    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize="small", ncol=3)
    plt.grid()
    plt.tight_layout()

    # 그래프 저장
    plt.savefig(output_file, bbox_inches="tight")
    print(f"그래프가 저장되었습니다: {output_file}")

    # 그래프 출력
    plt.show()
