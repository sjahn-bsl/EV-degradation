import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress

# 파일 경로 설정
input_file = r"D:\SamsungSTF\Processed_Data\Domain\Zchg_ist.csv"
output_folder = r"D:\SamsungSTF\Processed_Data\Domain\Zchg_plots_EV6"
os.makedirs(output_folder, exist_ok=True)

# 데이터 불러오기 및 전처리
df = pd.read_csv(input_file)
df["start_time"] = pd.to_datetime(df["start_time"], errors='coerce')
df = df.dropna(subset=["start_time", "Z_CHG", "Z_CHG_increase(%)", "device_id", "avg_pack_current"])
df = df.sort_values(by=["device_id", "start_time"])

# 이상치 제거 (Z_CHG 기준)
q1 = df["Z_CHG"].quantile(0.25)
q3 = df["Z_CHG"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df_filtered = df[(df["Z_CHG"] >= lower_bound) & (df["Z_CHG"] <= upper_bound)]

# 전체 범위 그래프
plt.figure(figsize=(30, 10))
significant_devices = []

for device_id in df_filtered["device_id"].unique():
    subset = df_filtered[df_filtered["device_id"] == device_id]
    plt.plot(subset["start_time"], subset["Z_CHG"], marker='o', linestyle='-', label=f"Device {device_id}")

    if len(subset) >= 2:
        x = subset["start_time"].map(pd.Timestamp.toordinal)
        y = subset["Z_CHG"]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        if p_value < 0.05:
            significant_devices.append(device_id)
            last_time = subset["start_time"].iloc[-1]
            last_val = subset["Z_CHG"].iloc[-1]
            plt.annotate(f"p={p_value:.3f}", (last_time, last_val), fontsize=8, color="gray")

# 그래프 설정
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Z_CHG")
plt.title("Z_CHG vs Time (All avg_pack_current ranges)")
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize="small", ncol=3)
plt.grid()
plt.tight_layout()

# 저장
output_path = os.path.join(output_folder, "Z_CHG_vs_time_ALL_range_with_significant_pvalue.png")
plt.savefig(output_path, bbox_inches="tight")
plt.show()

# 두 번째 그래프: delta_soc ≥ 60인 단말기만
df_soc60 = df_filtered[df_filtered["delta_soc(%)"] >= 60]
plt.figure(figsize=(30, 10))
significant_devices_soc60 = []

for device_id in df_soc60["device_id"].unique():
    subset = df_soc60[df_soc60["device_id"] == device_id]
    plt.plot(subset["start_time"], subset["Z_CHG"], marker='o', linestyle='-', label=f"Device {device_id}")

    if len(subset) >= 2:
        x = subset["start_time"].map(pd.Timestamp.toordinal)
        y = subset["Z_CHG"]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        if p_value < 0.05:
            significant_devices_soc60.append(device_id)
            last_time = subset["start_time"].iloc[-1]
            last_val = subset["Z_CHG"].iloc[-1]
            plt.annotate(f"p={p_value:.3f}", (last_time, last_val), fontsize=8, color="gray")

# 그래프 설정
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Z_CHG")
plt.title("Z_CHG vs Time (Only delta_soc ≥ 60%)")
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize="small", ncol=3)
plt.grid()
plt.tight_layout()

# 저장
output_path_soc60 = os.path.join(output_folder, "Z_CHG_vs_time_SOC60_only.png")
plt.savefig(output_path_soc60, bbox_inches="tight")
plt.show()
