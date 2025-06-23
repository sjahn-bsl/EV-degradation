import pandas as pd
import plotly.graph_objects as go

# 데이터 불러오기
csv_path = r"D:\SamsungSTF\Processed_Data\Domain\Ech_soc_ini.csv"
df = pd.read_csv(csv_path)
df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
df = df.dropna(subset=['Ech_cell_Wh', 'start_time', 'soc_initial(%)', 'ini_mod_temp_avg'])

# Plot 1: Time vs SOC vs Ech
fig1 = go.Figure(data=[go.Scatter3d(
    x=df['start_time'],
    y=df['soc_initial(%)'],
    z=df['Ech_cell_Wh'],
    mode='markers',
    marker=dict(size=4, color=df['Ech_cell_Wh'], colorscale='Viridis_r', opacity=0.8),
)])
fig1.update_layout(
    title='Ech vs Start Time vs SOC',
    scene=dict(
        xaxis_title='Start Time',
        yaxis_title='SOC Initial (%)',
        zaxis_title='Ech_cell_Wh',
        aspectmode='cube'
    )
)
fig1.show()

# Plot 2: SOC vs Temp vs Ech
fig2 = go.Figure(data=[go.Scatter3d(
    x=df['soc_initial(%)'],
    y=df['ini_mod_temp_avg'],
    z=df['Ech_cell_Wh'],
    mode='markers',
    marker=dict(size=4, color=df['Ech_cell_Wh'], colorscale='Viridis_r', opacity=0.8),
)])
fig2.update_layout(
    title='Ech vs SOC vs Initial Temp',
    scene=dict(
        xaxis_title='SOC Initial (%)',
        yaxis_title='Initial Mod Temp (°C)',
        zaxis_title='Ech_cell_Wh',
        aspectmode='cube'
    )
)
fig2.show()

# Plot 3: Time vs Temp vs Ech
fig3 = go.Figure(data=[go.Scatter3d(
    x=df['start_time'],
    y=df['ini_mod_temp_avg'],
    z=df['Ech_cell_Wh'],
    mode='markers',
    marker=dict(size=4, color=df['Ech_cell_Wh'], colorscale='Viridis_r', opacity=0.8),
)])
fig3.update_layout(
    title='Ech vs Start Time vs Initial Temp',
    scene=dict(
        xaxis_title='Start Time',
        yaxis_title='Initial Mod Temp (°C)',
        zaxis_title='Ech_cell_Wh',
        aspectmode='cube'
    )
)
fig3.show()
