import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # 눈금 설정용
import numpy as np
import pandas as pd
from scipy.integrate import quad
from tqdm import tqdm

from Aging_Model import (
    k_Cal,
    k_Cyc_High_T,
    k_Cyc_Low_T_Current,
    k_Cyc_Low_T_High_SOC,
)

# ──────────────────────────────────────────────────────────────────────────────
# k 계수 확인 (디버깅)
# ──────────────────────────────────────────────────────────────────────────────
print("[k 계수 확인용 출력]")
print("k_Cal(298.15K, SOC=100):", k_Cal(298.15, 100))
print("k_Cyc_High_T(298.15K):", k_Cyc_High_T(298.15))
print("k_Cyc_Low_T_Current(298.15K, 2A):", k_Cyc_Low_T_Current(298.15, 2.0))
print(
    "k_Cyc_Low_T_High_SOC(298.15K, 2A, SOC=100):",
    k_Cyc_Low_T_High_SOC(298.15, 2.0, 100),
)

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 파일 경로
# ──────────────────────────────────────────────────────────────────────────────
file_paths = {
    4.2: [
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I27.8_V4.20.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I55.6_V4.20.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I111.2_V4.20.csv",
    ],
    4.15: [
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I27.8_V4.15.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I55.6_V4.15.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I111.2_V4.15.csv",
    ],
    4.1: [
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I27.8_V4.10.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I55.6_V4.10.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I111.2_V4.10.csv",
    ],
}

T_fixed = 298.15  # Kelvin (25°C)

# ──────────────────────────────────────────────────────────────────────────────
# 열화율(dQ/dt) 함수
# ──────────────────────────────────────────────────────────────────────────────
def integrate_k_cal(t, T, SOC):
    """Calendar aging: dQ/dt = k_Cal(T, SOC) / (2*sqrt(t))."""
    return k_Cal(T, SOC) / (2 * np.sqrt(max(t, 1e-6)))


def integrate_k_cycHighT(phi_tot, T):
    """High-T cycling: dQ/d(phi_tot) = k / (2*sqrt(phi_tot))."""
    return k_Cyc_High_T(T) / (2 * np.sqrt(max(phi_tot, 1e-6)))


def integrate_k_cycLowT(phi_ch, T, I_Ch):
    """Low-T cycling current term: dQ/d(phi_ch) = k / (2*sqrt(phi_ch))."""
    return k_Cyc_Low_T_Current(T, I_Ch) / (2 * np.sqrt(max(phi_ch, 1e-6)))


def integrate_k_cycLowTHighSOC(T, I_Ch, SOC):
    """Low-T & High-SOC cycling term."""
    return k_Cyc_Low_T_High_SOC(T, I_Ch, SOC)


# ──────────────────────────────────────────────────────────────────────────────
# 한 사이클 구간의 손실 계산
# ──────────────────────────────────────────────────────────────────────────────
def calculate_loss(data, initial_time=0, initial_chr_cap=0, initial_cap=0):
    """
    입력: 데이터프레임(한 사이클 구간), 이전까지의 누적 시간/전하량.
    출력: 항목별 손실, 누적 리스트, 최종 상태 등.
    """
    # 결과 버퍼
    calendar_losses = []               # 시간 적분(h)
    cyc_high_T_losses = []             # |I| 기반 누적 전하량 적분
    cyc_low_T_losses = []              # 충전 전류(+)만 적분
    cyc_low_T_high_SOC_losses = []     # SOC>82% 충전 전류 적분
    total_losses = []

    # 누적 변수
    chr_cap_list, cap_list, time_list = [], [], []
    chr_cap = initial_chr_cap
    cap = initial_cap
    prev_time = initial_time
    prev_chr_cap = chr_cap
    prev_cap = cap

    dqdt_list = []

    for i in range(len(data)):
        SOC = data.loc[i, "SOC"]
        current = data.loc[i, "Current(A)"]
        time_hr = data.loc[i, "Time (seconds)"] / 3600.0
        dt = time_hr - prev_time

        # 순간 열화율
        cal_rate = integrate_k_cal(max(time_hr, 1e-6), T_fixed, SOC)
        cyc_high_rate = integrate_k_cycHighT(max(cap, 1e-6), T_fixed)

        if current > 0:
            cyc_low_rate = integrate_k_cycLowT(max(chr_cap, 1e-6), T_fixed, current)
            if SOC > 82:
                cyc_highSOC_rate = integrate_k_cycLowTHighSOC(T_fixed, current, SOC)
            else:
                cyc_highSOC_rate = 0
        else:
            cyc_low_rate = 0
            cyc_highSOC_rate = 0

        dqdt_list.append(cal_rate + cyc_high_rate + cyc_low_rate + cyc_highSOC_rate)

        # 디버깅 로그
        if i % 10000 == 0:
            print(
                f"[{i}] dt={dt:.6f} hr, I={current:.3f} A, dQ={current * dt:.6e} Ah"
            )

        # 누적 전하량 업데이트
        if current > 0:
            chr_cap += current * dt
        cap += abs(current) * dt

        chr_cap_list.append(chr_cap)
        cap_list.append(cap)
        time_list.append(time_hr)

        # ── 항목별 손실 적분 (가독성용 변수명 정렬) ─────────────────────
        calendar_loss, _ = quad(
            lambda t: integrate_k_cal(t, T_fixed, SOC), prev_time, time_hr
        )
        cyc_high_T_loss, _ = quad(
            lambda phi_tot: integrate_k_cycHighT(phi_tot, T_fixed), prev_cap, cap
        )

        if current > 0:
            cyc_low_T_loss, _ = quad(
                lambda phi_ch: integrate_k_cycLowT(phi_ch, T_fixed, current),
                prev_chr_cap, chr_cap,
            )
            if SOC > 82:
                # integrand가 phi_ch에 의존하지 않아도, 적분구간이 dphi_ch이므로 이름만 맞춰 둠
                cyc_low_T_high_SOC_loss, _ = quad(
                    lambda phi_ch: integrate_k_cycLowTHighSOC(T_fixed, current, SOC),
                    prev_chr_cap, chr_cap,
                )
            else:
                cyc_low_T_high_SOC_loss = 0
        else:
            cyc_low_T_loss = 0
            cyc_low_T_high_SOC_loss = 0

        # 모델 용량 정규화
        scale_highT = np.sqrt(3 / 55.6)
        scale_lowT = np.sqrt(3 / 55.6)
        scale_highSOC = 3 / 55.6

        cyc_high_T_loss *= scale_highT
        cyc_low_T_loss *= scale_lowT
        cyc_low_T_high_SOC_loss *= scale_highSOC

        # 합산 및 버퍼 저장
        total_loss = (
            calendar_loss
            + cyc_high_T_loss
            + cyc_low_T_loss
            + cyc_low_T_high_SOC_loss
        )

        calendar_losses.append(calendar_loss)
        cyc_high_T_losses.append(cyc_high_T_loss)
        cyc_low_T_losses.append(cyc_low_T_loss)
        cyc_low_T_high_SOC_losses.append(cyc_low_T_high_SOC_loss)
        total_losses.append(total_loss)

        # 구간 종료 상태 업데이트
        prev_time = time_hr
        prev_cap = cap
        prev_chr_cap = chr_cap

    final_time = data["Time (seconds)"].iloc[-1] / 3600.0
    cumulative_total_losses = np.cumsum(total_losses)

    return (
        calendar_losses,
        cyc_high_T_losses,
        cyc_low_T_losses,
        cyc_low_T_high_SOC_losses,
        total_losses,
        chr_cap_list,
        cap_list,
        chr_cap,
        cap,
        final_time,
        time_list,
        cumulative_total_losses,
        dqdt_list,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 파일 처리(전체 100사이클)
# ──────────────────────────────────────────────────────────────────────────────
def process_file(file_path):
    # 1) 로드 & 컬럼명 통일
    data = pd.read_csv(file_path)
    data.rename(
        columns={
            "Time (s)": "Time (seconds)",
            "Current (A)": "Current(A)",
            "Voltage (V)": "Scaled Voltage(V)",
        },
        inplace=True,
    )

    # 2) 사이클 분할
    num_cycles = 100
    cycle_length = len(data) // num_cycles

    # 3) 누적 버퍼
    cumulative_losses_over_cycles = []
    initial_chr_cap = 0.0
    initial_cap = 0.0
    initial_time = 0.0

    all_time_list = []
    all_cumulative_loss_list = []
    all_dqdt_list = []

    all_calendar_losses = []
    all_cyc_high_T_losses = []
    all_cyc_low_T_losses = []
    all_cyc_low_T_high_SOC_losses = []

    cycle_start_indices = []

    # 4) 루프
    for cycle in tqdm(range(num_cycles), desc="Processing Cycles"):
        start_idx = cycle * cycle_length
        end_idx = (cycle + 1) * cycle_length if cycle < num_cycles - 1 else len(data)
        data_cycle = data.iloc[start_idx:end_idx].reset_index(drop=True)

        results = calculate_loss(
            data_cycle, initial_time, initial_chr_cap, initial_cap
        )
        (
            calendar_losses,
            cyc_high_T_losses,
            cyc_low_T_losses,
            cyc_low_T_high_SOC_losses,
            total_losses,
            _chr_cap_list,
            _cap_list,
            final_chr_cap,
            final_cap,
            final_time,
            time_list,
            cumulative_total_losses,
            dqdt_list,
        ) = results

        # 4-2) 사이클별 누적 Q_loss
        cycle_total_loss = np.sum(total_losses)
        if cycle == 0:
            cumulative_losses_over_cycles.append(cycle_total_loss)
        else:
            cumulative_losses_over_cycles.append(
                cumulative_losses_over_cycles[-1] + cycle_total_loss
            )

        # 4-3) 시간축 누적 Q_loss 이어붙이기
        cycle_start_indices.append(len(all_time_list))

        if not all_cumulative_loss_list:
            all_cumulative_loss_list.extend(cumulative_total_losses)
        else:
            last_cum_loss = all_cumulative_loss_list[-1]
            all_cumulative_loss_list.extend(last_cum_loss + cumulative_total_losses)

        all_time_list.extend(time_list)
        all_dqdt_list.extend(list(dqdt_list))

        all_calendar_losses.extend(calendar_losses)
        all_cyc_high_T_losses.extend(cyc_high_T_losses)
        all_cyc_low_T_losses.extend(cyc_low_T_losses)
        all_cyc_low_T_high_SOC_losses.extend(cyc_low_T_high_SOC_losses)

        # 4-4) 상태 업데이트
        initial_chr_cap = final_chr_cap
        initial_cap = final_cap
        initial_time = final_time

        # 길이 체크
        assert len(all_time_list) == len(all_dqdt_list), (
            f"Length mismatch: time_list={len(all_time_list)}, "
            f"dqdt_list={len(all_dqdt_list)}"
        )
        print(
            f"DEBUG: time_list len={len(all_time_list)}, "
            f"dqdt_list len={len(all_dqdt_list)}"
        )

    # 5) 반환
    return (
        cumulative_losses_over_cycles,
        all_time_list,
        all_cumulative_loss_list,
        cycle_start_indices,
        all_dqdt_list,
        all_calendar_losses,
        all_cyc_high_T_losses,
        all_cyc_low_T_losses,
        all_cyc_low_T_high_SOC_losses,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 출력 폴더
# ──────────────────────────────────────────────────────────────────────────────
save_dir = r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\Comprehensive Modeling"
os.makedirs(save_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────
colors = [
    [0.0, 0.450980392156863, 0.76078431372549],
    [0.937254901960784, 0.752941176470588, 0.0],
    [0.803921568627451, 0.325490196078431, 0.298039215686275],
    [0.125490196078431, 0.52156862745098, 0.305882352941177],
    [0.572549019607843, 0.368627450980392, 0.623529411764706],
    [0.882352941176471, 0.529411764705882, 0.152941176470588],
    [0.301960784313725, 0.733333333333333, 0.835294117647059],
    [0.933333333333333, 0.298039215686275, 0.592156862745098],
    [0.494117647058824, 0.380392156862745, 0.282352941176471],
    [0.454901960784314, 0.462745098039216, 0.470588235294118],
]

for vmax, paths in file_paths.items():
    # 한 번 처리해 캐시
    results = {file: process_file(file) for file in paths}

    # ── Figure 1: Q_loss vs Cycle ───────────────────────────
    plt.figure(figsize=(10, 6))
    for idx, file in enumerate(paths):
        (
            cyc_losses,
            time_list,
            qloss_list,
            cycle_start_indices,
            dqdt_list,
            calendar_losses,
            cyc_high_T_losses,
            cyc_low_T_losses,
            cyc_low_T_high_SOC_losses,
        ) = results[file]

        cyc_losses = np.array(cyc_losses) * 100.0
        qloss_list = np.array(qloss_list) * 100.0

        label_str = (
            file.split("\\")[-1]
            .replace("cc_cv_2rc_result_", "")
            .replace(".csv", "")
        )
        x_vals = range(1, len(cyc_losses) + 1)

        plt.fill_between(
            x_vals,
            cyc_losses,
            color=colors[idx % len(colors)],
            alpha=0.3,
            label=label_str,
        )
        plt.plot(x_vals, cyc_losses, color=colors[idx % len(colors)])

    plt.xlabel("Cycle Number")
    plt.ylabel("Cumulative Q_loss [%]")
    title_str = f"Q_loss vs Cycle (Vmax={vmax}V)"
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, title_str.replace(" ", "_") + ".png"), dpi=300)
    plt.close()

    # ── Figure 2: Q_loss vs Time ────────────────────────────
    plt.figure(figsize=(10, 6))
    for idx, file in enumerate(paths):
        (
            cyc_losses,
            time_list,
            qloss_list,
            cycle_start_indices,
            dqdt_list,
            calendar_losses,
            cyc_high_T_losses,
            cyc_low_T_losses,
            cyc_low_T_high_SOC_losses,
        ) = results[file]

        cyc_losses = np.array(cyc_losses) * 100.0
        qloss_list = np.array(qloss_list) * 100.0

        label_str = (
            file.split("\\")[-1]
            .replace("cc_cv_2rc_result_", "")
            .replace(".csv", "")
        )
        plt.fill_between(
            time_list,
            qloss_list,
            color=colors[idx % len(colors)],
            alpha=0.3,
            label=label_str,
        )
        plt.plot(time_list, qloss_list, color=colors[idx % len(colors)])

    plt.xlabel("Time [h]")
    plt.ylabel("Cumulative Q_loss [%]")
    title_str = f"Q_loss vs Time (Vmax={vmax}V)"
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, title_str.replace(" ", "_") + ".png"), dpi=300)
    plt.close()

    # ── Figure 2-2: Loss Components vs Time (파일별) ────────
    for idx, file in enumerate(paths):
        (
            cyc_losses,
            time_list,
            qloss_list,
            cycle_start_indices,
            dqdt_list,
            calendar_losses,
            cyc_high_T_losses,
            cyc_low_T_losses,
            cyc_low_T_high_SOC_losses,
        ) = results[file]

        label_str = (
            file.split("\\")[-1]
            .replace("cc_cv_2rc_result_", "")
            .replace(".csv", "")
        )

        plt.figure(figsize=(10, 6))
        plt.stackplot(
            time_list,
            np.cumsum(calendar_losses) * 100.0,
            np.cumsum(cyc_high_T_losses) * 100.0,
            np.cumsum(cyc_low_T_losses) * 100.0,
            np.cumsum(cyc_low_T_high_SOC_losses) * 100.0,
            labels=["Calendar", "Cyc High-T", "Cyc Low-T", "Cyc Low-T High SOC"],
            colors=["blue", "orange", "green", "red"],
            alpha=0.6,
        )

        calendar_final = np.cumsum(calendar_losses)[-1]
        cyc_high_T_final = np.cumsum(cyc_high_T_losses)[-1]
        cyc_low_T_final = np.cumsum(cyc_low_T_losses)[-1]
        cyc_low_T_high_SOC_final = np.cumsum(cyc_low_T_high_SOC_losses)[-1]

        print(f"\n[기여도 분석] {label_str} @ Vmax={vmax}V")
        print(f"  Calendar: {calendar_final * 100:.3f}%")
        print(f"  Cyc High-T: {cyc_high_T_final * 100:.3f}%")
        print(f"  Cyc Low-T: {cyc_low_T_final * 100:.3f}%")
        print(f"  Cyc Low-T High SOC: {cyc_low_T_high_SOC_final * 100:.6f}%")

        plt.xlabel("Time [h]")
        plt.ylabel("Cumulative Component Loss [%]")
        plt.title(f"Loss Components vs Time (Vmax={vmax}V, {label_str})")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(
            save_dir, f"Loss_Components_vs_Time_Vmax{vmax}_{label_str}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()

# ------------------------------------------------------------------------------
# 아래 블록은 dQ/dt 관련 추가 그림(주석 처리)
# ------------------------------------------------------------------------------
"""
# Figure 3: dQ/dt vs Time
plt.figure(figsize=(10, 6))
for idx, file in enumerate(paths):
    (cyc_losses, time_list, qloss_list, cycle_start_indices, dqdt_list,
     calendar_losses, cyc_high_T_losses, cyc_low_T_losses,
     cyc_low_T_high_SOC_losses) = results[file]

    time_array = np.array(time_list) * 3600  # [h] → [s]
    dqdt_array = np.array(dqdt_list) * 100   # [%/hr]
    cutoff_sec = 30
    mask = time_array > cutoff_sec

    label_str = file.split("\\")[-1].replace("cc_cv_2rc_result_", "").replace(".csv", "")
    plt.plot(time_array[mask] / 3600, dqdt_array[mask],
             label=label_str, color=colors[idx % len(colors)])

plt.xlabel("Time (hours)")
plt.ylabel("dQ/dt [%/hr]")
title_str = f"dQdt vs Time (Vmax={vmax}V)"
plt.title(title_str)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, title_str.replace(" ", "_") + ".png"), dpi=300)
plt.close()

# Figure 4: dQ/dt vs Time; 3 cycles
plt.figure(figsize=(16, 6))
for idx, file in enumerate(paths):
    (cyc_losses, time_list, qloss_list, cycle_start_indices, dqdt_list,
     calendar_losses, cyc_high_T_losses, cyc_low_T_losses,
     cyc_low_T_high_SOC_losses) = results[file]

    start_idx = cycle_start_indices[0]
    end_idx = cycle_start_indices[3]

    time_slice_hr = np.array(time_list[start_idx:end_idx])
    time_slice_sec = time_slice_hr * 3600
    dqdt_slice = np.array(dqdt_list[start_idx:end_idx]) * 100

    cutoff_sec = 30
    mask = time_slice_sec > cutoff_sec

    label_str = file.split("\\")[-1].replace("cc_cv_2rc_result_", "").replace(".csv", "")
    plt.plot(time_slice_sec[mask], dqdt_slice[mask],
             label=f"{label_str} (3 Cycles)", color=colors[idx % len(colors)])

plt.xlabel("Time (seconds)")
plt.ylabel("dQ/dt [%/hr]")
title_str = f"dQdt over First 3 Cycles (Vmax={vmax}V)"
plt.title(title_str)
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10000))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, title_str.replace(" ", "_") + "_dQdt_3cycles.png"),
            dpi=300)
plt.close()
"""
