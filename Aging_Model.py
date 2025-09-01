import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants and Parameters from Table IV
k_Cal_Ref = 3.69e-4  # h^-0.5
k_Cyc_High_T_Ref = 1.46e-4  # Ah^-0.5
k_Cyc_Low_T_Ref = 4.01e-4  # Ah^-0.5
k_Cyc_Low_T_High_SOC_Ref = 2.03e-6  # Ah^-1
Ea_Cal = 2.06e4  # J/mol
Ea_Cyc_High_T = 3.27e4  # J/mol
Ea_Cyc_Low_T = 5.55e4  # J/mol
Ea_Cyc_Low_T_High_SOC = 2.33e5  # J/mol

alpha = 1.4  # dimensionless, SOC에 따른 열화 민감도를 조절함;  기존값: 0.384
beta_Low_T = 0.65  # h;    기존 값:2.64
beta_Low_T_High_SOC = 0.784  # h; 기존 값: 7.84

SOC_Ref = 82  # 82%
T_Ref = 298.15  # K (25°C)
I_Ch_Ref = 55.6  # A
Ua_Ref = 0.123  # V (Reference potential)
k0 = 0.142  # dimensionless
Rg = 8.314  # J/(mol*K)
F = 96485  # C/mol (Faraday constant)
C0 = 55.6  # Ah, nominal cell capacity

# Equation 9 : Temperature and SOC dependence for Calendar Aging
def k_Cal(T, SOC):
    term1 = np.exp(-Ea_Cal * (1/T - 1/T_Ref) / Rg)
    term2 = np.exp(alpha * F * (Ua_Ref - Ua_SOC(x_a(SOC))) / (Rg * T_Ref))
    return k_Cal_Ref * term1 * (term2 + k0)

#   Ua(SOC) function definition
#   Appendix [A2]에서 SOC를 anode stoichiometry로 환산하는 보간식
def x_a(SOC):
    return 8.5e-3 + 0.01 * SOC * (7.8e-1 - 8.5e-3)
#   Appendix [A1]에서 제시된 anode open circuit potential 식
def Ua_SOC(x_a_value):
    result = (0.6379 + 0.5416 * np.exp(-305.5309 * x_a_value)
              + 0.044 * np.tanh(-(x_a_value - 0.1958) / 0.1088)
              - 0.1978 * np.tanh((x_a_value - 1.0571) / 0.0854)
              - 0.6875 * np.tanh((x_a_value + 0.0117) / 0.0529)
              - 0.0175 * np.tanh((x_a_value - 0.5692) / 0.0875))
    return result

# Equation 15: Temperature dependence for high temperature cycle aging
def k_Cyc_High_T(T):
    return k_Cyc_High_T_Ref * np.exp(-Ea_Cyc_High_T / Rg * (1 / T - 1 / T_Ref))

# Equation 18: Temperature and current rate dependence for low temperature cycle aging
def k_Cyc_Low_T_Current(T, I_Ch):
    temp_term = np.exp(Ea_Cyc_Low_T / Rg * (1 / T - 1 / T_Ref))

    current_term = np.exp(beta_Low_T * (I_Ch - I_Ch_Ref) / C0)
    return k_Cyc_Low_T_Ref * temp_term * current_term

# Equation 21 : Temperature and SOC dependence for low temperature and high SOC cycling Aging
def k_Cyc_Low_T_High_SOC(T, I_Ch, SOC):
    temp_term = np.exp(Ea_Cyc_Low_T_High_SOC / Rg * (1 / T - 1 / T_Ref))
    current_term = np.exp(beta_Low_T_High_SOC * (I_Ch - I_Ch_Ref) / C0)
    sgn_SOC = 1 if SOC >= SOC_Ref else 0  # SOC가 SOC_Ref 이상일 때만 계산 적용
    return k_Cyc_Low_T_High_SOC_Ref * temp_term * current_term * sgn_SOC


# Creating the mesh for Temperature and SOC
temperature_celsius = np.linspace(0, 60, 101)  # Temperature range (°C)
soc_range = np.linspace(0, 100, 101)           # SOC range
T_mesh_celsius, SOC_mesh = np.meshgrid(temperature_celsius, soc_range)

# Creating the mesh for Temperature and Current
c_rate_range = np.linspace(0, 1, 101)  # C-rate: 0C to 1C
current_range = c_rate_range * C0      # A = C-rate × capacity
T_mesh_celsius, Current_mesh = np.meshgrid(temperature_celsius, current_range)
C_rate_mesh = Current_mesh / C0        # For plotting in C-rate units

# Calculate k_Cal values for the mesh
k_Cal_mesh_celsius = np.array([[k_Cal(T+273.15, SOC) for T, SOC in zip(t_row, SOC_mesh_row)]
                               for t_row, SOC_mesh_row in zip(T_mesh_celsius, SOC_mesh)])

# Calculate k_Cyc_High_T values for the mesh
k_Cyc_High_T_mesh = np.array([[k_Cyc_High_T(T+273.15) for T in t_row] for t_row in T_mesh_celsius])

# Calculate k_Cyc_Low_T_Current for the mesh
k_Cyc_Low_T_mesh = np.array([[k_Cyc_Low_T_Current(T+273.15, Current) for T, Current in zip(t_row, Current_mesh_row)]
                             for t_row, Current_mesh_row in zip(T_mesh_celsius, Current_mesh)])
# Calculate k_Cyc_Low_T_High_SOC for the mesh
# k_Cyc_Low_T_High_SOC_mesh = np.array([[k_Cyc_Low_T_High_SOC(T+273.15, Current, SOC) for T, Current, SOC in zip(t_row, Current_mesh_row, SOC_mesh_row)]
#                              for t_row, Current_mesh_row, SOC_mesh_row in zip(T_mesh_celsius, Current_mesh, SOC_mesh)])
# Calculate k_Cyc_Low_T_High_SOC for the mesh
k_Cyc_Low_T_High_SOC_mesh = np.array([[k_Cyc_Low_T_High_SOC(T+273.15, Current, 90) for T, Current in zip(t_row, Current_mesh_row)]
                             for t_row, Current_mesh_row in zip(T_mesh_celsius, Current_mesh)])


# Plotting the 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface
surf = ax.plot_surface(T_mesh_celsius, SOC_mesh, k_Cal_mesh_celsius * 1e3, cmap='viridis')

# Adding labels and title
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('State of Charge (SOC) [%]')
ax.set_zlabel('k_Cal(T, SOC) $[\\times 10^{-3}]$')
ax.set_title('3D Surface Plot of k_Cal(T, SOC)')
ax.set_xlim(60, 0)



# Plotting the 3D surface plot for k_Cyc_Low_T_Current
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
# Plotting the surface
surf2 = ax2.plot_surface(C_rate_mesh, T_mesh_celsius, k_Cyc_Low_T_mesh * 1e3, cmap='viridis')

# Adding labels and title
ax2.set_xlabel('Charge Rate [C]')
ax2.set_ylabel('Temperature (°C)')
ax2.set_zlabel('k_Cyc_Low_T(T, I) $[\\times 10^{-3}]$')
ax2.set_title('3D Surface Plot of k_Cyc_Low_T_Current(T, I)')
ax2.set_xlim(1, 0)
ax2.set_ylim(60, 0)  # Set the x-axis limits from 60 to 0



# Plotting the 3D surface plot
fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')

# Plotting the surface
surf3 = ax3.plot_surface(T_mesh_celsius, C_rate_mesh, k_Cyc_High_T_mesh * 1e4, cmap='viridis')

# Adding labels and title
ax3.set_xlabel('Temperature (°C)')
ax3.set_ylabel('Charge Rate [C]')
ax3.set_zlabel('k_Cyc_High_T(T) $[\\times 10^{-4}]$')
ax3.set_title('3D Surface Plot of k_Cyc_High_T(T)')
ax3.set_xlim(60, 0)


# Plotting the 3D surface plot for k_Cyc_Low_T_Current
fig4 = plt.figure(figsize=(10, 7))
ax4 = fig4.add_subplot(111, projection='3d')

# Plotting the surface
surf4 = ax4.plot_surface(C_rate_mesh, T_mesh_celsius, k_Cyc_Low_T_High_SOC_mesh * 1e3, cmap='viridis')

# Adding labels and title
ax4.set_xlabel('Charge Rate [C]')
ax4.set_ylabel('Temperature (°C)')
ax4.set_zlabel('k_Cyc_Low_T_High_SOC(T, I) $[\\times 10^{-3}]$')
ax4.set_title('3D Surface Plot of k_Cyc_Low_T_High_SOC (T, I, SOC=100)')
ax4.set_ylim(60, 0)  # Set the x-axis limits from 60 to 0
ax4.set_xlim(1, 0)

# Show the plot
plt.show()
