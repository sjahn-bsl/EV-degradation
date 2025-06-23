import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress

# 파일 경로 설정
input_file = r"D:\SamsungSTF\Processed_Data\Domain\Ech_cell_summary_with_loss.csv"
output_folder = r"D:\SamsungSTF\Processed_Data\Domain\Ech_plots_EV6"
os.makedirs(output_folder, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv(input_file)
df["start_time"] = pd.to_datetime(df["start_time"], errors='coerce')

df = df[df["Ech_cell_Wh"] > 10]
df = df.dropna(subset=["start_time", "Ech_cell_Wh", "device_id"])
df = df.sort_values(by=["device_id", "start_time"])

# 단말기 리스트
unique_devices = df["device_id"].unique()

# 1. 전체 단말기 Ech vs Time (p-value 포함, p < 0.05만 표시)
plt.figure(figsize=(30, 10))
significant_devices = []

for device_id in unique_devices:
    subset = df[df["device_id"] == device_id]
    plt.plot(subset["start_time"], subset["Ech_cell_Wh"], marker='o', linestyle='-', label=f"Device {device_id}")

    if len(subset) >= 2:
        x = subset["start_time"].map(pd.Timestamp.toordinal)
        y = subset["Ech_cell_Wh"]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        if p_value < 0.05:
            significant_devices.append(device_id)
            last_time = subset["start_time"].iloc[-1]
            last_val = y.iloc[-1]
            plt.annotate(f"p={p_value:.3f}", (last_time, last_val), fontsize=8, color="gray")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Ech_cell_Wh")
plt.title("Ech_cell_Wh vs Time per Device (p < 0.05 only)")
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize="small", ncol=3)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Ech_vs_time_all_with_pvalue.png"), bbox_inches="tight")
plt.show()

# 2. delta_soc(%) ≥ 60 조건을 만족하는 데이터만 사용한 그래프 (p < 0.05만 표시)
df_soc_filtered = df[(df["delta_soc(%)"] >= 60)]
filtered_devices = df_soc_filtered["device_id"].unique()
plt.figure(figsize=(30, 10))
for device_id in filtered_devices:
    subset = df_soc_filtered[df_soc_filtered["device_id"] == device_id]
    plt.plot(subset["start_time"], subset["Ech_cell_Wh"], marker='o', linestyle='-', label=f"Device {device_id}")

    if len(subset) >= 2 and subset["start_time"].nunique() > 1:
        x = subset["start_time"].map(pd.Timestamp.toordinal)
        y = subset["Ech_cell_Wh"]
        try:
            slope, intercept, r, p, _ = linregress(x, y)
            if p < 0.05:
                last_time = subset["start_time"].iloc[-1]
                last_val = y.iloc[-1]
                plt.annotate(f"p={p:.3f}", (last_time, last_val), fontsize=8, color="gray")
        except Exception as e:
            print(f"[linregress 오류] {device_id}: {e}")
            continue

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Ech_cell_Wh")
plt.title("Ech_cell_Wh vs Time (delta_soc ≥ 60%, p < 0.05 only)")
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize="small", ncol=3)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Ech_vs_time_soc60plus.png"), bbox_inches="tight")
plt.show()

# 3. p-value < 0.05 단말기만
plt.figure(figsize=(30, 10))

for device_id in unique_devices:
    subset = df[df["device_id"] == device_id]

    if len(subset) >= 2 and subset["start_time"].nunique() > 1:
        x = subset["start_time"].map(pd.Timestamp.toordinal)
        y = subset["Ech_cell_Wh"]
        slope, intercept, r, p, _ = linregress(x, y)

        if p < 0.05:
            df_sub = df[df["device_id"] == device_id]
            plt.plot(df_sub["start_time"], df_sub["Ech_cell_Wh"], marker='o', linestyle='-', label=f"Device {device_id}")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Ech_cell_Wh")
plt.title("Ech_cell_Wh vs Time (p < 0.05 only, filenames shown)")
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize="small", ncol=2)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Ech_vs_time_pvalue_below_0.05_filenames.png"), bbox_inches="tight")
plt.show()

