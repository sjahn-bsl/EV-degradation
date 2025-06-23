import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# 설정
device_id = "01241228076"
input_folder = r"E:\SamsungSTF\Processed_Data\TripByTrip"
MIN_DURATION = 4
MIN_DELTA_I = 3
MAX_INITIAL_CURRENT = 1.25
N_p = 2  # EV6 병렬 셀 수

# 파일 목록 수집
file_info = []
for fname in os.listdir(input_folder):
    if device_id not in fname or not fname.endswith(".csv"):
        continue
    fpath = os.path.join(input_folder, fname)
    try:
        df = pd.read_csv(fpath)
        start_time = pd.to_datetime(df["time"].iloc[0], errors="coerce")
        if pd.isna(start_time):
            continue
        file_info.append((fname, start_time))
    except:
        continue

file_info = sorted(file_info, key=lambda x: x[1])

# 결과 저장
resistances = []
peak_idx = 0

for fname, trip_time in tqdm(file_info, desc=f"{device_id} 처리 중", unit="file"):
    file_path = os.path.join(input_folder, fname)
    try:
        df = pd.read_csv(file_path)

        if not {'pack_current', 'pack_volt', 'acceleration', 'cell_volt_list', 'time'}.issubset(df.columns):
            continue

        file_month = trip_time.strftime("%b")

        df['cell_count'] = df['cell_volt_list'].apply(lambda x: len(str(x).split(',')))
        N_s = df['cell_count'].mode().iloc[0]
        df['cell_volt'] = df['pack_volt'] / N_s
        df['cell_current'] = df['pack_current'] / N_p
        df['I_smooth'] = df['cell_current'].rolling(window=50, min_periods=1).mean()
        df['dI_dt'] = df['I_smooth'].diff()

        # 가속 구간 정의: cell_current > 0인 연속 구간
        df['current_flag'] = (df['cell_current'] > 0).astype(int)
        df['flag_shift'] = df['current_flag'].shift().fillna(0)
        df['flag_change'] = df['current_flag'] != df['flag_shift']
        change_indices = df.index[df['flag_change']].tolist() + [len(df)]

        for i in range(len(change_indices) - 1):
            start = change_indices[i]
            end = change_indices[i + 1]
            if df.loc[start, 'current_flag'] == 1 and (end - start) >= MIN_DURATION:
                I0 = df.loc[start, 'cell_current']
                I1 = df.loc[end - 1, 'cell_current']
                delta_I = I1 - I0

                if abs(I0) < MAX_INITIAL_CURRENT and delta_I >= MIN_DELTA_I:
                    dI_segment = df['dI_dt'].iloc[start:end]
                    if (dI_segment >= 0).all():
                        V0 = df.loc[start, 'cell_volt']
                        V1 = df.loc[end - 1, 'cell_volt']
                        delta_V = V1 - V0
                        R = -delta_V / delta_I
                        resistances.append((peak_idx, R, file_month))
                        peak_idx += 1


    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        continue

# DataFrame 생성
r_df = pd.DataFrame(resistances, columns=["peak_idx", "R", "month"])
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
r_df["month"] = pd.Categorical(r_df["month"], categories=month_order, ordered=True)

# 이동 평균
avg_r_by_peak = r_df.groupby("peak_idx")["R"].mean().rolling(window=30, min_periods=5).mean()

# Month별로 alpha 조절
month_alpha = {month: 0.1 + i * 0.07 for i, month in enumerate(month_order)}

# ----------------------------- 그래프 -----------------------------
plt.figure(figsize=(14, 6))
for month in month_order:
    month_data = r_df[r_df["month"] == month]
    if not month_data.empty:
        alpha = month_alpha[month]
        plt.scatter(month_data["peak_idx"], month_data["R"], color="gray", alpha=alpha, label=month, s=10)

plt.plot(avg_r_by_peak.index, avg_r_by_peak.values, color="lime", linewidth=2.5, label="Average Resistance")

plt.xlabel("Current Peak Number (Time Ordered)")
plt.ylabel("R [Ω]")
plt.title(f"Device {device_id} Resistance over Time (Gray deepens by Month)", fontsize=18)
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.show()
