import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import plotly.graph_objects as go

# -------- 설정 --------
device_id = "01241228076"
input_folder = r"E:\SamsungSTF\Processed_Data\TripByTrip"
MIN_DURATION = 4
MIN_DELTA_I = 3
MAX_INITIAL_CURRENT = 5
N_p = 2

# -------- 결과 저장 --------
results = []

# -------- Trip 파일 수집 --------
file_list = sorted([f for f in os.listdir(input_folder) if device_id in f and f.endswith(".csv")])

for fname in tqdm(file_list, desc=f"Processing {device_id}", unit="file"):
    try:
        fpath = os.path.join(input_folder, fname)
        df = pd.read_csv(fpath)

        required_cols = {'pack_current', 'pack_volt', 'acceleration', 'cell_volt_list', 'soc', 'time'}
        if not required_cols.issubset(df.columns):
            continue

        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        trip_time = df['time'].iloc[0]

        df['cell_count'] = df['cell_volt_list'].apply(lambda x: len(str(x).split(',')))
        N_s = df['cell_count'].mode().iloc[0]
        df['cell_volt'] = df['pack_volt'] / N_s
        df['cell_current'] = df['pack_current'] / N_p
        df['I_smooth'] = df['cell_current'].rolling(window=50, min_periods=1).mean()
        df['dI_dt'] = df['I_smooth'].diff()

        if 'mod_temp_list' in df.columns:
            df['mod_temp_avg'] = df['mod_temp_list'].apply(
                lambda x: np.mean([float(t) for t in str(x).split(',')]) if pd.notna(x) else np.nan
            )
        else:
            df['mod_temp_avg'] = np.nan

        # -------- 가속 구간 정의 및 R 계산 (Resistance_Anal 방식 적용) --------
        df['current_flag'] = (df['cell_current'] > 0).astype(int)
        df['flag_shift'] = df['current_flag'].shift().fillna(0)
        df['flag_change'] = df['current_flag'] != df['flag_shift']
        change_indices = df.index[df['flag_change']].tolist() + [len(df)]

        for i in range(len(change_indices) - 1):
            s = change_indices[i]
            e = change_indices[i + 1]
            if df.loc[s, 'current_flag'] == 1 and (e - s) >= MIN_DURATION:
                I0 = df.loc[s, 'cell_current']
                I1 = df.loc[e - 1, 'cell_current']
                delta_I = I1 - I0

                if abs(I0) >= MAX_INITIAL_CURRENT:
                    continue
                if delta_I < MIN_DELTA_I:
                    continue
                if not (df['dI_dt'].iloc[s:e] >= 0).all():
                    continue

                V0 = df.loc[s, 'cell_volt']
                V1 = df.loc[e - 1, 'cell_volt']
                delta_V = V1 - V0
                R = -delta_V / delta_I

                soc_avg = df.loc[s:e, 'soc'].mean()
                temp_avg = df.loc[s:e, 'mod_temp_avg'].mean()
                results.append((trip_time, soc_avg, temp_avg, R))


    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        continue

# -------- DataFrame 생성 --------
df3d = pd.DataFrame(results, columns=["time", "soc", "temp", "R"]).dropna()
df3d["time_num"] = mdates.date2num(df3d["time"])

# -------- Plot 1: R vs Time & SOC --------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df3d["time_num"], df3d["soc"], df3d["R"], c=df3d["R"], cmap='viridis_r', s=8)
ax.set_xlabel("Time")
ax.set_ylabel("SOC [%]")
ax.set_zlabel("R_peak [Ω]")
ax.set_title("R vs Time & SOC", fontsize=24)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.colorbar(sc, shrink=0.6, label="R_peak [Ω]")
plt.tight_layout()
plt.show()

# -------- Plot 2: R vs Time & Temp --------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df3d["time_num"], df3d["temp"], df3d["R"], c=df3d["R"], cmap='viridis_r', s=8)
ax.set_xlabel("Time")
ax.set_ylabel("Temp [°C]")
ax.set_zlabel("R_peak [Ω]")
ax.set_title("R vs Time & Temperature", fontsize=24)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.colorbar(sc, shrink=0.6, label="R_peak [Ω]")
plt.tight_layout()
plt.show()

# -------- Plot 3: R vs SOC & Temp --------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df3d["temp"], df3d["soc"], df3d["R"], c=df3d["R"], cmap='viridis_r', s=8)
ax.set_xlabel("Temp [°C]")
ax.set_ylabel("SOC [%]")
ax.set_zlabel("R_peak [Ω]")
ax.set_title("R vs SOC & Temperature", fontsize=24)
plt.colorbar(sc, shrink=0.6, label="R_peak [Ω]")
plt.tight_layout()
plt.show()
