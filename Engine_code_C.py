import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
from scipy.integrate import quad
import matplotlib.pyplot as plt
from Aging_Model_C import k_Cal, k_Cyc_High_T, k_Cyc_Low_T_Current, k_Cyc_Low_T_High_SOC
from tqdm import tqdm

import matplotlib.ticker as ticker  # 눈금 설정을 위한 모듈 추가

# 디버깅코드
print("[k 계수 확인용 출력]")
print("k_Cal(298.15K, SOC=100):", k_Cal(298.15, 100))
print("k_Cyc_High_T(298.15K):", k_Cyc_High_T(298.15))
print("k_Cyc_Low_T_Current(298.15K, 2A):", k_Cyc_Low_T_Current(298.15, 2.0))
print("k_Cyc_Low_T_High_SOC(298.15K, 2A, SOC=100):", k_Cyc_Low_T_High_SOC(298.15, 2.0, 100))

# 데이터 파일 로드
file_paths = {
    4.2: [
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I27.8_V4.20.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I55.6_V4.20.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I111.2_V4.20.csv"
    ],
    4.15: [
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I27.8_V4.15.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I55.6_V4.15.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I111.2_V4.15.csv"
    ],
    4.1: [
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I27.8_V4.10.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I55.6_V4.10.csv",
        r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model\cc_cv_2rc_result_I111.2_V4.10.csv"
    ]
}

T_fixed = 298.15  # Kelvin (25°C)

"""
열화 적분 함수 정의
각 항목별 순간 열화율 dQ/dt를 정의.
time, phi_tot, phi_ch등의 변수에 따라 달라짐.
"""
def integrate_k_cal(t, T, SOC):
    return k_Cal(T, SOC) / (2 * np.sqrt(max(t, 1e-6)))  # dQ/dt

def integrate_k_cycHighT(phi_tot, T):
    return k_Cyc_High_T(T) / (2 * np.sqrt(max(phi_tot, 1e-6)))

def integrate_k_cycLowT(phi_ch, T, I_Ch):
    return k_Cyc_Low_T_Current(T, I_Ch) / (2 * np.sqrt(max(phi_ch, 1e-6)))

def integrate_k_cycLowTHighSOC(T, I_Ch, SOC):
    return k_Cyc_Low_T_High_SOC(T, I_Ch, SOC)

# 손실 계산 함수: 한 사이클 데이터를 입력받아서 dQ/dt와 Q_loss 계산
def calculate_loss(data, initial_time=0, initial_chr_cap=0, initial_cap=0): #initial_: 이전 사이클에서 이어지는 누적값
    # 초기화 변수
    calendar_losses = []    # 시간에 대한 적분(h 단위)
    cyc_high_T_losses = []  # 누적 전하량(abs I 기반)
    cyc_low_T_losses = []   # 충전 전하량(양의 I만)
    cyc_low_T_high_SOC_losses = []  #SOC>82%일 때만, 충전 전하량 기반
    total_losses = []

    chr_cap_list = []   #누적 충전량
    cap_list = []   #전체 전하량
    time_list = []  #시간 축

    chr_cap = initial_chr_cap
    cap = initial_cap
    prev_time = initial_time
    prev_chr_cap = chr_cap
    prev_cap = cap

    dqdt_list = []  # dq/dt 초기화

    for i in range(len(data)):  # 각 timestep에서 SOC, current, time(hr)을 읽음.
        SOC = data.loc[i, 'SOC']
        current = data.loc[i, 'Current(A)']
        time_hr = data.loc[i, 'Time (seconds)'] / 3600  # sec → hr
        dt = time_hr - prev_time    #dt는 현재 timestep간격

        #integrate 함수들을 호출해서 순간 열화율 dQ/dt 계산
        #scipy.integrate.quad로 구간 적분하여 해당 step의 열화량 산출.
        cal_rate = integrate_k_cal(max(time_hr,1e-6), T_fixed, SOC)
        cyc_high_rate = integrate_k_cycHighT(max(cap,1e-6), T_fixed)
        if current > 0:
            cyc_low_rate = integrate_k_cycLowT(max(chr_cap, 1e-6), T_fixed, current)
            if SOC > 82:
                cyc_highSOC_rate = integrate_k_cycLowTHighSOC(T_fixed, current, SOC)
            else:
                cyc_highSOC_rate = 0
        else:
            cyc_low_rate = 0
            cyc_highSOC_rate = 0

        total_rate = cal_rate + cyc_high_rate + cyc_low_rate + cyc_highSOC_rate #해당 step의 총 순간 열화율
        dqdt_list.append(total_rate)

        # 디버깅 코드
        if i % 10000 == 0:
            print(f"[{i}] dt = {dt:.6f} hr, I = {current:.3f} A, dQ = {current * dt:.6e} Ah")

        # 누적량 업데이트
        if current > 0: # 전류가 양수(충전)이면 chr_cap 증가.
            chr_cap += current * dt #chr_cap: 누적 충전량
        cap += abs(current) * dt    #cap: 누적 전하량

        chr_cap_list.append(chr_cap)
        cap_list.append(cap)
        time_list.append(time_hr)

        # 항목별 손실량 적분
        calendar_loss, _ = quad(lambda t: integrate_k_cal(t, T_fixed, SOC), prev_time, time_hr) # 이전 시간부터 현재 시간까지의 구간에서 발생한 캘린더 열화량을 수치적분으로 계산
        cyc_high_T_loss, _ = quad(lambda phi_tot: integrate_k_cycHighT(phi_tot, T_fixed), prev_cap, cap)
        if current > 0:
            cyc_low_T_loss, _ = quad(lambda phi_ch: integrate_k_cycLowT(phi_ch, T_fixed, current), prev_chr_cap, chr_cap)
            if SOC > 82:
                cyc_low_T_high_SOC_loss, _ = quad(lambda phi_ch: integrate_k_cycLowTHighSOC(T_fixed, current, SOC), prev_chr_cap, chr_cap)
            else:
                cyc_low_T_high_SOC_loss = 0
        else:
            cyc_low_T_loss = 0
            cyc_low_T_high_SOC_loss = 0

        # 모델 용량 조정 시, dphi normalize
        scale_highT = np.sqrt(3 / 55.6)  # ≈ 0.48
        scale_lowT = np.sqrt(3 / 55.6)  # ≈ 0.48
        scale_highSOC = 3 / 55.6  # ≈ 0.054
        cyc_high_T_loss *= scale_highT
        cyc_low_T_loss *= scale_lowT
        cyc_low_T_high_SOC_loss *= scale_highSOC

        # 열화 항목 출력
        if i % 10000 == 0:
            print(f"  k_Cal Loss: {calendar_loss:.3e}")
            print(f"  k_Cyc_High_T Loss: {cyc_high_T_loss:.3e}")
            print(f"  k_Cyc_Low_T Loss: {cyc_low_T_loss:.3e}")
            print(f"  k_Cyc_Low_T_High_SOC Loss: {cyc_low_T_high_SOC_loss:.3e}")

        total_loss = calendar_loss + cyc_high_T_loss + cyc_low_T_loss + cyc_low_T_high_SOC_loss
        # 결과 저장, 각 step의 loss를 리스트에 저장
        calendar_losses.append(calendar_loss)
        cyc_high_T_losses.append(cyc_high_T_loss)
        cyc_low_T_losses.append(cyc_low_T_loss)
        cyc_low_T_high_SOC_losses.append(cyc_low_T_high_SOC_loss)
        total_losses.append(total_loss)

        # 구간 정보 업데이트
        prev_time = time_hr
        prev_cap = cap
        prev_chr_cap = chr_cap
    # 누적 Q_loss 생성 및 반환
    final_time = data['Time (seconds)'].iloc[-1] / 3600
    cumulative_total_losses = np.cumsum(total_losses)    #누적 Q loss 생성
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
        time_list,  #initial_time 이후 누적 시간(hr)
        cumulative_total_losses, # 해당 cycle에서 시간별 누적 Q_loss 리스트
        dqdt_list   #순간 열화율
    )


def process_file(file_path):
    #1. 파일 읽기
    data = pd.read_csv(file_path)
    data.rename(columns={
        'Time (s)': 'Time (seconds)',
        'Current (A)': 'Current(A)',
        'Voltage (V)': 'Scaled Voltage(V)'
    }, inplace=True)

    # 2. 사이클 관련 설정
    num_cycles = 100  # cc_cv_2rc_result.csv의 사이클 수
    cycle_length = len(data) // num_cycles  # 사이클당 행 수

    # 3. 초기화
    # 누적 손실 리스트
    cumulative_losses_over_cycles = []  #사이클 단위 누적 Q_loss
    initial_chr_cap = 0
    initial_cap = 0
    initial_time = 0
    all_time_list = []  #전체 시뮬레이션 시간 기록
    all_cumulative_loss_list = []   #전체 누적 Q_loss(시간축)
    all_dqdt_list = []  #전체 dQ/dt 기록

    all_calendar_losses = []
    all_cyc_high_T_losses = []
    all_cyc_low_T_losses = []
    all_cyc_low_T_high_SOC_losses = []

    cycle_start_indices = []  # 각 사이클의 시작 인덱스를 저장

    # 4. 사이클 단위 루프
    # 사이클별 열화량 계산 루프
    for cycle in tqdm(range(num_cycles), desc="Processing Cycles"):
        start_idx = cycle * cycle_length
        end_idx = (cycle + 1) * cycle_length if cycle < num_cycles - 1 else len(data)
        data_cycle = data.iloc[start_idx:end_idx].reset_index(drop=True)    #각 cycle 구간을 data_cycle로 분리

        # 4-1. 열화량 계산, calculate_loss()를 호출해 항목별 손실량과 누적 데이터를 가져옴.
        results = calculate_loss(data_cycle, initial_time, initial_chr_cap, initial_cap)  # initial_time: 이전까지의 총 누적 시간
        (calendar_losses, cyc_high_T_losses, cyc_low_T_losses,
         cyc_low_T_high_SOC_losses, total_losses,
         _1, _2, final_chr_cap, final_cap, final_time,
         time_list, cumulative_total_losses, dqdt_list) = results

        # 4-2. Q_loss vs cycle 저장
        cycle_total_loss = np.sum(total_losses)
        #한 사이클의 total_loss를 합산해서 "사이클별 누적 Q_loss" 생성
        cumulative_losses_over_cycles.append(
            cycle_total_loss if cycle == 0 else cumulative_losses_over_cycles[-1] + cycle_total_loss
        )

        # 4-3. 시간별 Q_loss 이어붙이기
        cycle_start_indices.append(len(all_time_list))  # 사이클 시작점 기록, 각 cycle의 시작 인덱스를 저장 --> 나중에 특정 cycle 구간만 시각화할 때 사용.
        if not all_cumulative_loss_list:
            all_cumulative_loss_list.extend(cumulative_total_losses)    #cumulative_total_losses는 해당 사이클 내 누적 Q_loss
        else:
            last_cum_loss = all_cumulative_loss_list[-1]  # cycle간 이어붙일 때는 이전 cycle의 마지막 누적값을 더해서 연결.
            all_cumulative_loss_list.extend(
                [last_cum_loss + x for x in cumulative_total_losses])  # 전체 실험 시간 동안의 누적 Q_loss 저장
        # 전체 리스트 이어붙이기
        all_time_list.extend(time_list)  # 전체 시점 기준으로 Q_loss가 측정된 시간들을 이어 붙임
        all_dqdt_list.extend(list(dqdt_list))     # 전체 dQ/dt 이어붙이기

        all_calendar_losses.extend(calendar_losses)
        all_cyc_high_T_losses.extend(cyc_high_T_losses)
        all_cyc_low_T_losses.extend(cyc_low_T_losses)
        all_cyc_low_T_high_SOC_losses.extend(cyc_low_T_high_SOC_losses)

        # 4-4. 상태 업데이트
        initial_chr_cap = final_chr_cap
        initial_cap = final_cap
        initial_time = final_time

        # dq/dt 안전장치: 시간 리스트와 dQ/dt 리스트 길이가 맞는지 검사.
        assert len(all_time_list) == len(all_dqdt_list), \
            f"Length mismatch: time_list={len(all_time_list)}, dqdt_list={len(all_dqdt_list)}"
        print(f"DEBUG: time_list len={len(all_time_list)}, dqdt_list len={len(all_dqdt_list)}")

    # 5. 결과 반환
    return (
        cumulative_losses_over_cycles,
        all_time_list,
        all_cumulative_loss_list,
        cycle_start_indices,
        all_dqdt_list,
        # 항목별 누적 리스트 추가
        all_calendar_losses,
        all_cyc_high_T_losses,
        all_cyc_low_T_losses,
        all_cyc_low_T_high_SOC_losses
    )

# 저장 폴더 만들기
save_dir = r"C:\Users\6211s\OneDrive\Desktop\kentech\DFC\Aging_Model_C\Comprehensive Modeling"
os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성


# 시각화

# RGB 컬러 리스트 정의 (0~1 범위)
colors = [
    [0, 0.450980392156863, 0.760784313725490],
    [0.937254901960784, 0.752941176470588, 0],
    [0.803921568627451, 0.325490196078431, 0.298039215686275],
    [0.125490196078431, 0.521568627450980, 0.305882352941177],
    [0.572549019607843, 0.368627450980392, 0.623529411764706],
    [0.882352941176471, 0.529411764705882, 0.152941176470588],
    [0.301960784313725, 0.733333333333333, 0.835294117647059],
    [0.933333333333333, 0.298039215686275, 0.592156862745098],
    [0.494117647058824, 0.380392156862745, 0.282352941176471],
    [0.454901960784314, 0.462745098039216, 0.470588235294118]
]

for vmax, paths in file_paths.items():
    # 모든 파일 한번씩만 처리해서 결과 저장
    results = {}
    for file in paths:
        results[file] = process_file(file)

    # Figure 1: Q_loss vs Cycle
    plt.figure(figsize=(10, 6))
    for idx, file in enumerate(paths):
        (cyc_losses, time_list, qloss_list, #cyc_losses: 사이클별 누적 Q_loss, qloss_list: 시간별 누적 Q_loss
         cycle_start_indices, dqdt_list,
         calendar_losses, cyc_high_T_losses,
         cyc_low_T_losses, cyc_low_T_high_SOC_losses) = results[file]

        # ===== % 단위 변환 =====
        cyc_losses = np.array(cyc_losses) * 100
        qloss_list = np.array(qloss_list) * 100

        label_str = file.split("\\")[-1].replace("cc_cv_2rc_result_", "").replace(".csv", "")
        x_vals = range(1, len(cyc_losses) + 1)
        plt.fill_between(x_vals, cyc_losses, color=colors[idx % len(colors)], alpha=0.3, label=label_str)
        plt.plot(x_vals, cyc_losses, color=colors[idx % len(colors)])  # 윤곽선 유지하고 싶으면 추가

    plt.xlabel("Cycle Number")
    plt.ylabel("Cumulative Q_loss [%]")
    title_str = f"Q_loss vs Cycle (Vmax={vmax}V)"
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, title_str.replace(" ", "_") + ".png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Figure 2: Q_loss vs Time
    plt.figure(figsize=(10, 6))
    for idx, file in enumerate(paths):
        (cyc_losses, time_list, qloss_list,
         cycle_start_indices, dqdt_list,
         calendar_losses, cyc_high_T_losses,
         cyc_low_T_losses, cyc_low_T_high_SOC_losses) = results[file]

        # ===== % 단위 변환 =====
        cyc_losses = np.array(cyc_losses) * 100
        qloss_list = np.array(qloss_list) * 100

        label_str = file.split("\\")[-1].replace("cc_cv_2rc_result_", "").replace(".csv", "")
        plt.fill_between(time_list, qloss_list, color=colors[idx % len(colors)], alpha=0.3, label=label_str)
        plt.plot(time_list, qloss_list, color=colors[idx % len(colors)])

    plt.xlabel("Time [h]")
    plt.ylabel("Cumulative Q_loss [%]")
    title_str = f"Q_loss vs Time (Vmax={vmax}V)"
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, title_str.replace(" ", "_") + ".png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Figure 2-2: Loss Components vs Time (파일마다 따로 저장)
    for idx, file in enumerate(paths):
        (cyc_losses, time_list, qloss_list,
         cycle_start_indices, dqdt_list,
         calendar_losses, cyc_high_T_losses,
         cyc_low_T_losses, cyc_low_T_high_SOC_losses) = results[file]

        label_str = file.split("\\")[-1].replace("cc_cv_2rc_result_", "").replace(".csv", "")

        plt.figure(figsize=(10, 6))  # <-- 루프 안으로 이동

        plt.stackplot(
            time_list,
            np.cumsum(calendar_losses) * 100,
            np.cumsum(cyc_high_T_losses) * 100,
            np.cumsum(cyc_low_T_losses) * 100,
            np.cumsum(cyc_low_T_high_SOC_losses) * 100,
            labels=["Calendar", "Cyc High-T", "Cyc Low-T", "Cyc Low-T High SOC"],
            colors=["blue", "orange", "green", "red"],
            alpha=0.6
        )

        #print로 정확한 수치 확인하기
        calendar_final = np.cumsum(calendar_losses)[-1]
        cyc_high_T_final = np.cumsum(cyc_high_T_losses)[-1]
        cyc_low_T_final = np.cumsum(cyc_low_T_losses)[-1]
        cyc_low_T_high_SOC_final = np.cumsum(cyc_low_T_high_SOC_losses)[-1]
        print(f"\n[기여도 분석] {label_str} @ Vmax={vmax}V")
        print(f"  Calendar: {calendar_final * 100:.3f}%")
        print(f"  Cyc High-T: {cyc_high_T_final * 100:.3f}%")
        print(f"  Cyc Low-T: {cyc_low_T_final * 100:.3f}%")
        print(
            f"  Cyc Low-T High SOC: {cyc_low_T_high_SOC_final * 100:.6f}%")


        plt.xlabel("Time [h]")
        plt.ylabel("Cumulative Component Loss [%]")
        plt.title(f"Loss Components vs Time (Vmax={vmax}V, {label_str})")
        plt.legend()
        plt.grid(True)

        # 파일명에 Vmax + label_str까지 붙여서 구분
        save_path = os.path.join(save_dir, f"Loss_Components_vs_Time_Vmax{vmax}_{label_str}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

"""
    # Figure 3: dQ/dt vs Time
    plt.figure(figsize=(10, 6))
    for idx, file in enumerate(paths):
        # 파일 처리
        (cyc_losses, time_list, qloss_list,
         cycle_start_indices, dqdt_list,
         calendar_losses, cyc_high_T_losses,
         cyc_low_T_losses, cyc_low_T_high_SOC_losses) = results[file]

        # 초반 구간 제외
        time_array = np.array(time_list) * 3600  # [h] → [s]로 변환
        dqdt_array = np.array(dqdt_list) * 100  # [%/hr]
        cutoff_sec = 30  # 여기서 잘라낼 구간(30초), 필요하면 60초로 변경 가능
        mask = time_array > cutoff_sec

        # 파일명 라벨용 추출
        label_str = file.split("\\")[-1].replace("cc_cv_2rc_result_", "").replace(".csv", "")
        plt.plot(time_array[mask] / 3600, dqdt_array[mask],  # 다시 시간[h]로 맞춰서 플롯
                 label=label_str, color=colors[idx % len(colors)])
    plt.xlabel('Time (hours)')
    plt.ylabel('dQ/dt [%/hr]')
    title_str = f'dQdt vs Time (Vmax={vmax}V)'
    plt.title(title_str)
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(save_dir, title_str.replace(" ", "_") + ".png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Figure 4: dQ/dt vs Time; 3cycle만
    plt.figure(figsize=(16, 6))
    for idx, file in enumerate(paths):
        (cyc_losses, time_list, qloss_list,
         cycle_start_indices, dqdt_list,
         calendar_losses, cyc_high_T_losses,
         cyc_low_T_losses, cyc_low_T_high_SOC_losses) = results[file]
        start_idx = cycle_start_indices[0]
        end_idx = cycle_start_indices[3]  # 3번째 사이클 시작 인덱스

        time_slice_hr = np.array(time_list[start_idx:end_idx])
        time_slice_sec = time_slice_hr * 3600
        dqdt_slice = np.array(dqdt_list[start_idx:end_idx]) * 100  # [%/hr] 단위 그대로 활용

        # 초반 구간 제외
        cutoff_sec = 30
        mask = time_slice_sec > cutoff_sec

        label_str = file.split("\\")[-1].replace("cc_cv_2rc_result_", "").replace(".csv", "")
        plt.plot(time_slice_sec[mask], dqdt_slice[mask],
                 label=f'{label_str} (3 Cycles)', color=colors[idx % len(colors)])

    plt.xlabel('Time (seconds)')
    plt.ylabel('dQ/dt [%/hr]')
    title_str = f'dQdt over First 3 Cycles (Vmax={vmax}V)'
    plt.title(title_str)
    plt.grid(True)
    plt.legend()

    # x축 눈금 간격 10000초
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10000))
    plt.tight_layout()
    save_path = os.path.join(save_dir, title_str.replace(" ", "_") + "_dQdt_3cycles.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
"""