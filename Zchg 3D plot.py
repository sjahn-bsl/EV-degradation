import pandas as pd
import plotly.graph_objects as go

# 공통 CSV 파일 경로
csv_path = r"D:\SamsungSTF\Processed_Data\Domain\Zchg_summary_EV6_with_increase.csv"
df = pd.read_csv(csv_path)

# 공통 시간 변환
df["start_time"] = pd.to_datetime(df["start_time"], errors='coerce')

### Plot 1: Z_CHG vs Time vs SOC Initial
df1 = df.dropna(subset=["Z_CHG", "start_time", "soc_initial(%)"])
fig1 = go.Figure(data=[go.Scatter3d(
    x=df1["start_time"],
    y=df1["soc_initial(%)"],
    z=df1["Z_CHG"],
    mode='markers',
    marker=dict(
        size=5,
        color=df1["Z_CHG"],
        colorscale='viridis_r',
        opacity=0.8,
        colorbar=dict(title="Z_CHG")
    ),
    text=df1["filename"] if "filename" in df1.columns else None
)])
fig1.update_layout(
    scene=dict(
        xaxis_title="Start Time",
        yaxis_title="SOC Initial (%)",
        zaxis_title="Z_CHG"
    ),
    title="Z_CHG vs Time vs SOC Initial",
    margin=dict(l=0, r=0, b=0, t=30)
)
fig1.show()

### Plot 2: Z_CHG vs Time vs ini_mod_temp_avg
df2 = df.dropna(subset=["Z_CHG", "start_time", "ini_mod_temp_avg"])
fig2 = go.Figure(data=[go.Scatter3d(
    x=df2["start_time"],
    y=df2["ini_mod_temp_avg"],
    z=df2["Z_CHG"],
    mode='markers',
    marker=dict(
        size=5,
        color=df2["Z_CHG"],
        colorscale='viridis_r',
        opacity=0.8,
        colorbar=dict(title="Z_CHG")
    ),
    text=df2["filename"] if "filename" in df2.columns else None
)])
fig2.update_layout(
    scene=dict(
        xaxis_title="Start Time",
        yaxis_title="Initial Mod Temp (°C)",
        zaxis_title="Z_CHG"
    ),
    title="Z_CHG vs Time vs Initial Mod Temp",
    margin=dict(l=0, r=0, b=0, t=30)
)
fig2.show()

### Plot 3: Z_CHG vs ini_mod_temp_avg vs SOC Initial
df3 = df.dropna(subset=["Z_CHG", "ini_mod_temp_avg", "soc_initial(%)"])
fig3 = go.Figure(data=[go.Scatter3d(
    x=df3["ini_mod_temp_avg"],
    y=df3["soc_initial(%)"],
    z=df3["Z_CHG"],
    mode='markers',
    marker=dict(
        size=5,
        color=df3["Z_CHG"],
        colorscale='viridis_r',
        opacity=0.8,
        colorbar=dict(title="Z_CHG")
    ),
    text=df3["filename"] if "filename" in df3.columns else None
)])
fig3.update_layout(
    scene=dict(
        xaxis_title="Initial Mod Temp (°C)",
        yaxis_title="SOC Initial (%)",
        zaxis_title="Z_CHG"
    ),
    title="Z_CHG vs Initial Mod Temp vs SOC Initial",
    margin=dict(l=0, r=0, b=0, t=30)
)
fig3.show()
