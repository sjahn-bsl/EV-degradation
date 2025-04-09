import os
import pandas as pd
import matplotlib.pyplot as plt

# Trip CSV 파일 리스트
trip_files = [
    r"\\BSL\Shared_Drive\SamsungSTF\Processed_Data\TripByTrip_soc_2hr\EV6\bms_01241228103-2023-06-trip-3.csv",
    r"\\BSL\Shared_Drive\SamsungSTF\Processed_Data\TripByTrip_soc_2hr\Ioniq5\bms_01241228030-2023-09-trip-5.csv",
    r"\\BSL\Shared_Drive\SamsungSTF\Processed_Data\TripByTrip_soc_2hr\KonaEV\bms_01241228156-2023-05-trip-3.csv",
    r"\\BSL\Shared_Drive\SamsungSTF\Processed_Data\TripByTrip_soc_2hr\KonaEV\bms_01241228156-2023-07-trip-11.csv",
]

# Trip별로 따로 그리기
for file_path in trip_files:
    try:
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])

        # Figure 생성
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Trip Data: {os.path.basename(file_path)}", fontsize=14)

        # (1) SOC vs Time (검은 점)
        axs[0].plot(df['time'], df['soc'], color='black', marker='o', linestyle='None', markersize=3)
        axs[0].set_ylabel('SOC (%)')
        axs[0].set_title('SOC vs. Time')
        axs[0].grid(True)

        # (2) Pack Current vs Time (파란 선)
        axs[1].plot(df['time'], df['pack_current'], color='blue')
        axs[1].set_ylabel('Pack Current (A)')
        axs[1].set_title('Pack Current vs. Time')
        axs[1].grid(True)

        # (3) Pack Voltage vs Time (빨간 선)
        axs[2].plot(df['time'], df['pack_volt'], color='red')
        axs[2].set_ylabel('Pack Voltage (V)')
        axs[2].set_title('Pack Voltage vs. Time')
        axs[2].set_xlabel('Time')
        axs[2].tick_params(axis='x', rotation=45)
        axs[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 제목 공간 확보
        plt.show()

    except Exception as e:
        print(f"[에러] {file_path}: {e}")
