import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress

# 경로 설정
input_file = r"D:\SamsungSTF\Processed_Data\Domain\R_peak_cell_Anal.csv"
output_folder = r"D:\SamsungSTF\Processed_Data\Domain\R_peak_cell_Anal_plots"
os.makedirs(output_folder, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv(input_file)
df["start_time"] = pd.to_datetime(df["start_time"], errors='coerce')
df = df.dropna(subset=["start_time", "R_cell_peak_avg", "device_id", "vehicle_type"])
df = df[df["vehicle_type"] == "EV6"]
df = df.sort_values(by=["device_id", "start_time"])

# 기준 날짜
cutoff = pd.to_datetime("2023-08-01")

early_devices = []
late_devices = []

for device_id in df["device_id"].unique():
    device_data = df[df["device_id"] == device_id]
    if device_data["start_time"].min() < cutoff:
        early_devices.append(device_id)
    else:
        late_devices.append(device_id)

# ---------- 첫 번째 그래프 ----------
plt.figure(figsize=(30, 10))
for device_id in early_devices:
    subset = df[df["device_id"] == device_id]
    plt.plot(subset["start_time"], subset["R_cell_peak_avg"], marker='o', linestyle='-', label=f"Device {device_id}")

    x = subset["start_time"].map(pd.Timestamp.toordinal)
    y = subset["R_cell_peak_avg"]
    valid = (~x.isna()) & (~y.isna())
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) >= 2 and x_valid.nunique() > 1:
        slope, intercept, r, p, _ = linregress(x_valid, y_valid)
        if p < 0.05 and not x_valid.empty and not y_valid.empty:
            last_idx = x_valid.index[-1]
            last_time = subset.loc[last_idx, "start_time"]
            last_val = y_valid.loc[last_idx]
            plt.annotate(f"p={p:.3f}", (last_time, last_val), fontsize=8, color="gray")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("R_cell_peak_avg")
plt.title("EV6 - R_cell_peak_avg vs Time (Device started before 2023-08)", fontsize=20)
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize="small", ncol=3)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "EV6_R_vs_time_early_devices.png"), bbox_inches="tight")
plt.show()

# ---------- 두 번째 그래프 ----------
plt.figure(figsize=(30, 10))
for device_id in late_devices:
    subset = df[df["device_id"] == device_id]
    plt.plot(subset["start_time"], subset["R_cell_peak_avg"], marker='o', linestyle='-', label=f"Device {device_id}")

    x = subset["start_time"].map(pd.Timestamp.toordinal)
    y = subset["R_cell_peak_avg"]
    valid = (~x.isna()) & (~y.isna())
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) >= 2 and x_valid.nunique() > 1:
        slope, intercept, r, p, _ = linregress(x_valid, y_valid)
        if p < 0.05 and not x_valid.empty and not y_valid.empty:
            last_idx = x_valid.index[-1]
            last_time = subset.loc[last_idx, "start_time"]
            last_val = y_valid.loc[last_idx]
            plt.annotate(f"p={p:.3f}", (last_time, last_val), fontsize=8, color="gray")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("R_cell_peak_avg")
plt.title("EV6 - R_cell_peak_avg vs Time (Device started from 2023-08)")
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize="small", ncol=3)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "EV6_R_vs_time_late_devices.png"), bbox_inches="tight")
plt.show()