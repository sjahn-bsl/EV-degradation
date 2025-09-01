import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
from scipy.integrate import quad
import matplotlib.pyplot as plt
from Aging_Model import k_Cal, k_Cyc_High_T, k_Cyc_Low_T_Current, k_Cyc_Low_T_High_SOC
from tqdm import tqdm

# ==== 공통 열화 적분 함수 ====
def integrate_k_cal(t, T, SOC):
    return k_Cal(T, SOC) / (2 * np.sqrt(max(t, 1e-6)))

def integrate_k_cycHighT(phi_tot, T):
    return k_Cyc_High_T(T) / (2 * np.sqrt(max(phi_tot, 1e-6)))

def integrate_k_cycLowT(phi_ch, T, I_Ch):
    return k_Cyc_Low_T_Current(T, I_Ch) / (2 * np.sqrt(max(phi_ch, 1e-6)))

def integrate_k_cycLowTHighSOC(T, I_Ch, SOC):
    return k_Cyc_Low_T_High_SOC(T, I_Ch, SOC)

# ==== 공통 열화량 계산 ====
def calculate_loss(file_path):
    df = pd.read_csv(file_path)

    # 시간 변환
    df['time'] = pd.to_datetime(df['time'])
    df['Time (seconds)'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

    # current 변환
    df['Current(A)'] = -df['pack_current'] / 2.0
    df['SOC'] = df['soc']
    df['Temp(K)'] = df['int_temp'] + 273.15


    # ===== 시동 꺼짐 구간 감지 & 전류/전압 0으로 =====
    time_diff = df['Time (seconds)'].diff().fillna(0)
    # 예: 10초 이상 건너뛰면 시동 꺼짐으로 간주
    idle_threshold = 10

    idle_idx = time_diff > idle_threshold
    df.loc[idle_idx, 'Current(A)'] = 0.0

    calendar_losses, cyc_high_T_losses, cyc_low_T_losses, cyc_low_T_high_SOC_losses, total_losses = [], [], [], [], []
    chr_cap, cap = 0, 0
    prev_time = 0.0
    prev_chr_cap, prev_cap = 0, 0
    time_list = []

    for i in tqdm(range(len(df)), desc=f"Processing {os.path.basename(file_path)}"):
        SOC = df.loc[i, 'SOC']
        current = df.loc[i, 'Current(A)']
        time_hr = df.loc[i, 'Time (seconds)'] / 3600
        T_row = df.loc[i, 'Temp(K)']
        dt = time_hr - prev_time

        if current > 0:
            chr_cap += current * dt
        cap += abs(current) * dt
        time_list.append(time_hr)

        calendar_loss, _ = quad(lambda t: integrate_k_cal(t, T_row, SOC), prev_time, time_hr)
        cyc_high_T_loss, _ = quad(lambda phi_tot: integrate_k_cycHighT(phi_tot, T_row), prev_cap, cap)
        if current > 0:  # 충전일 때만
            cyc_low_T_loss, _ = quad(lambda phi_ch: integrate_k_cycLowT(phi_ch, T_row, current), prev_chr_cap, chr_cap)
            if SOC > 80:
                cyc_low_T_high_SOC_loss, _ = quad(lambda phi_ch: integrate_k_cycLowTHighSOC(T_row, current, SOC),
                                                  prev_chr_cap, chr_cap)
            else:
                cyc_low_T_high_SOC_loss = 0
        else:
            cyc_low_T_loss = 0
            cyc_low_T_high_SOC_loss = 0

        # scaling
        cyc_high_T_loss *= np.sqrt(3 / 55.6)
        cyc_low_T_loss *= np.sqrt(3 / 55.6)
        cyc_low_T_high_SOC_loss *= 3 / 55.6

        total_loss = calendar_loss + cyc_high_T_loss + cyc_low_T_loss + cyc_low_T_high_SOC_loss

        calendar_losses.append(calendar_loss)
        cyc_high_T_losses.append(cyc_high_T_loss)
        cyc_low_T_losses.append(cyc_low_T_loss)
        cyc_low_T_high_SOC_losses.append(cyc_low_T_high_SOC_loss)
        total_losses.append(total_loss)

        prev_time = time_hr
        prev_cap = cap
        prev_chr_cap = chr_cap

    return {
        "time": time_list,
        "total": np.cumsum(total_losses),
        "cal": np.cumsum(calendar_losses),
        "cyc_high": np.cumsum(cyc_high_T_losses),
        "cyc_low": np.cumsum(cyc_low_T_losses),
        "cyc_highSOC": np.cumsum(cyc_low_T_high_SOC_losses)
    }

# ==== 파일 경로 ====
DFC_path = r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\01241228087_DFC.csv"
NONDFC_path = r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\bms_01241228087_CR.csv"

save_dir = r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\Compare_DFC"
os.makedirs(save_dir, exist_ok=True)

# ==== 계산 ====
result_DFC = calculate_loss(DFC_path)
result_NONDFC = calculate_loss(NONDFC_path)

# ==== Q_loss 비교 시각화 ====
plt.figure(figsize=(10, 6))
plt.plot(result_DFC["time"], result_DFC["total"] * 100, label="DFC", color="blue")
plt.plot(result_NONDFC["time"], result_NONDFC["total"] * 100, label="NON DFC", color="red")
plt.xlabel("Time [h]")
plt.ylabel("Cumulative Q_loss [%]")
plt.title("Q_loss vs Time (DFC vs NON DFC)")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, "Q_loss_Comparison.png"), dpi=300)
plt.close()

# ==== 기여도 비교 시각화 ====
labels = ["Calendar", "Cyc High-T", "Cyc Low-T", "Cyc Low-T High SOC"]
DFC_vals = [result_DFC["cal"][-1], result_DFC["cyc_high"][-1], result_DFC["cyc_low"][-1], result_DFC["cyc_highSOC"][-1]]
NONDFC_vals = [result_NONDFC["cal"][-1], result_NONDFC["cyc_high"][-1], result_NONDFC["cyc_low"][-1], result_NONDFC["cyc_highSOC"][-1]]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, np.array(DFC_vals)*100, width, label="DFC")
plt.bar(x + width/2, np.array(NONDFC_vals)*100, width, label="NON DFC")
plt.ylabel("Contribution to Q_loss [%]")
plt.title("Loss Component Contribution (DFC vs NON DFC)")
plt.xticks(x, labels)
plt.legend()
plt.grid(True, axis='y')
plt.savefig(os.path.join(save_dir, "Loss_Contribution_Comparison.png"), dpi=300)
plt.close()

# ==== Loss Components vs Time 시각화 ==== (Engine_code Figure 2-2 스타일)
for label, result in [("DFC", result_DFC), ("NON DFC", result_NONDFC)]:
    plt.figure(figsize=(10, 6))
    plt.stackplot(
        result["time"],
        result["cal"] * 100,
        result["cyc_high"] * 100,
        result["cyc_low"] * 100,
        result["cyc_highSOC"] * 100,
        labels=["Calendar", "Cyc High-T", "Cyc Low-T", "Cyc Low-T High SOC"],
        colors=["blue", "orange", "green", "red"],
        alpha=0.6
    )

    # print로 최종 기여도 출력
    print(f"\n[기여도 분석] {label}")
    print(f"  Calendar: {result['cal'][-1] * 100:.6f}%")
    print(f"  Cyc High-T: {result['cyc_high'][-1] * 100:.6f}%")
    print(f"  Cyc Low-T: {result['cyc_low'][-1] * 100:.6f}%")
    print(f"  Cyc Low-T High SOC: {result['cyc_highSOC'][-1] * 100:.6f}%")

    plt.xlabel("Time [h]")
    plt.ylabel("Cumulative Component Loss [%]")
    plt.title(f"Loss Components vs Time ({label})")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, f"Loss_Components_vs_Time_{label}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

print("비교 완료. 결과 저장 경로:", save_dir)
