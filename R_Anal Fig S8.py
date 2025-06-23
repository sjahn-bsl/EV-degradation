import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import re

# -------- 설정 --------
file_path = r"E:\SamsungSTF\Processed_Data\TripByTrip\bms_01241225206-2023-01-trip-20.csv"
MIN_DURATION = 4
MIN_DELTA_I = 3.0
MAX_INITIAL_CURRENT = 1.25
N_p = 2

# -------- 데이터 불러오기 및 계산 --------
df = pd.read_csv(file_path)
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])

df['cell_count'] = df['cell_volt_list'].apply(lambda x: len(str(x).split(',')))
N_s = df['cell_count'].mode().iloc[0]
df['cell_volt'] = df['pack_volt'] / N_s
df['cell_current'] = df['pack_current'] / N_p
df['I_smooth'] = df['cell_current'].rolling(window=50, min_periods=1).mean()
df['dI_dt'] = df['I_smooth'].diff()

# -------- 가속 구간 탐색 (수정: cell_current > 0 기반) --------
df['current_flag'] = (df['cell_current'] > 0).astype(int)
df['flag_change'] = df['current_flag'].diff().fillna(0).astype(bool)
change_indices = df.index[df['flag_change']].tolist() + [len(df)]

acc_windows = []
R_list = []

for i in range(len(change_indices) - 1):
    s = change_indices[i]
    e = change_indices[i + 1]
    if df.loc[s, 'current_flag'] == 1 and (e - s) >= MIN_DURATION:
        delta_I = df.loc[e - 1, 'cell_current'] - df.loc[s, 'cell_current']
        dI_segment = df['dI_dt'].iloc[s:e]
        if abs(df.loc[s, 'cell_current']) < MAX_INITIAL_CURRENT and delta_I >= MIN_DELTA_I and (dI_segment >= 0).all():
            acc_windows.append((s, e))
            V_start = df.loc[s, 'cell_volt']
            V_end = df.loc[e - 1, 'cell_volt']
            R_cell = -(V_end - V_start) / delta_I
            R_list.append(R_cell)

# -------- 제목 구성 --------
basename = os.path.basename(file_path)
match = re.search(r'bms(?:_altitude)?_(\d+)-(\d{4}-\d{2})', basename)
if match:
    device_id = match.group(1)
    year_month = match.group(2)
    title_str = f"R_Anal Acc. Zone - Device {device_id} ({year_month})"
else:
    title_str = "R_Anal Acc. Zone"

# 색상 지정
voltage_color = '#CD534C'
current_color = '#0073C2'

# -------- (1) 전체 구간 Figure --------
fig1, axs1 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig1.suptitle(title_str + " - total")

axs1[0].plot(df['time'], df['cell_volt'], color=voltage_color, label="Voltage")
axs1[1].plot(df['time'], df['cell_current'], color=current_color, label="Current")

for s, e in acc_windows:
    axs1[0].plot(df['time'].iloc[s:e], df['cell_volt'].iloc[s:e], color='cyan')
    axs1[1].plot(df['time'].iloc[s:e], df['cell_current'].iloc[s:e], color='cyan')
    axs1[0].axvline(df['time'].iloc[s], linestyle='--', color=voltage_color, alpha=0.4)
    axs1[0].axvline(df['time'].iloc[e - 1], linestyle='--', color=voltage_color, alpha=0.4)
    axs1[1].axvline(df['time'].iloc[s], linestyle='--', color=current_color, alpha=0.4)
    axs1[1].axvline(df['time'].iloc[e - 1], linestyle='--', color=current_color, alpha=0.4)

axs1[0].set_ylabel("Voltage [V]")
axs1[1].set_ylabel("Current [A]")
axs1[1].set_xlabel("Time")
axs1[0].legend()
axs1[1].legend()
axs1[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axs1[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# -------- (2) 확대 구간 Figure (모든 구간 반복) --------
for idx, (s, e) in enumerate(acc_windows):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"{title_str} - part #{idx + 1}")

    # Voltage plot
    v1 = df['cell_volt'].iloc[s]
    v2 = df['cell_volt'].iloc[e - 1]
    t = df['time'].iloc[s]
    axs[0].plot(df['time'].iloc[s:e], df['cell_volt'].iloc[s:e], color=voltage_color)
    axs[0].axvline(t, linestyle='--', color=voltage_color, alpha=0.5)
    axs[0].axvline(df['time'].iloc[e - 1], linestyle='--', color=voltage_color, alpha=0.5)
    axs[0].annotate("", xy=(t, v2), xytext=(t, v1), arrowprops=dict(arrowstyle='<->', color='black'))
    axs[0].text(t, (v1 + v2)/2, f"$\\Delta V_{{acc}}$ = {abs(v1 - v2):.4f} V", fontsize=9)
    axs[0].set_ylabel("Voltage [V]")
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # Current plot
    i1 = df['cell_current'].iloc[s]
    i2 = df['cell_current'].iloc[e - 1]
    axs[1].plot(df['time'].iloc[s:e], df['cell_current'].iloc[s:e], color=current_color)
    axs[1].axvline(t, linestyle='--', color=current_color, alpha=0.5)
    axs[1].axvline(df['time'].iloc[e - 1], linestyle='--', color=current_color, alpha=0.5)
    axs[1].annotate("", xy=(t, i2), xytext=(t, i1), arrowprops=dict(arrowstyle='<->', color='black'))
    axs[1].text(t, (i1 + i2)/2, f"$\\Delta I_{{acc}}$ = {abs(i2 - i1):.4f} A", fontsize=9)
    axs[1].set_ylabel("Current [A]")
    axs[1].set_xlabel("Time")
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# -------- R 출력 --------
for idx, R in enumerate(R_list):
    print(f"R_cell #{idx + 1}: {R:.5f} ohm")
