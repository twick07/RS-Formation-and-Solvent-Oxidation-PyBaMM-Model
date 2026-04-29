# %%
# Script to generate Figs. 3 in the paper
# Comparison of the original SC model and the new SC model for half-cell mechanistic validation

# %%
import os
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

# %%
# Define Models to be compared
model_1 = pybamm.lithium_ion.DFN_HALF_CELL_Shrinking_Core_T_Zhuo()
model_2 = pybamm.lithium_ion.DFN_HALF_CELL_Shrinking_Core_Wickramanayake()


# %% 
# Define the experiments to be simulated
C_rate = 0.07
C_rate_cut = 0.015
#C_rate_cut = 0.01


# Define experiment protocol

experiment = pybamm.Experiment(
    [
        (
            f"Charge at {C_rate} C until 3.8 V",
            f"Hold at 3.8 V until {C_rate_cut} C",
            f"Charge at {C_rate} C until 4.0 V",
            f"Hold at 4.0 V until {C_rate_cut} C",
            f"Charge at {C_rate} C until 4.2 V",
            f"Hold at 4.2 V until {C_rate_cut} C",
            f"Charge at {C_rate} C until 4.4 V",
            f"Hold at 4.4 V until {C_rate_cut} C",
            f"Discharge at {C_rate} C until 3.7 V",
        )
    ] * 1,
    period=1
)

experiment = pybamm.Experiment(
    [
        (
            f"Charge at {C_rate} C until 4.0 V",
            f"Hold at 4.0 V until {C_rate_cut} C",
            f"Charge at {C_rate} C until 4.2 V",
            f"Hold at 4.2 V until {C_rate_cut} C",
            f"Charge at {C_rate} C until 4.4 V",
            f"Hold at 4.4 V until {C_rate_cut} C",
            f"Charge at {C_rate} C until 4.6 V",
            f"Hold at 4.6 V until {C_rate_cut} C",
            f"Discharge at {C_rate} C until 3.7 V",
        )
    ] * 1,
    period=1
)

# %%
# Load parameter set
from Parameter_Sets_Wickramanayake2026 import param_half_cell,initial_oxygen_concentration
# Parameters for tuning Zhuo model to align with experimental data
param_half_cell["Initial oxygen concentration in positive shell [mol.m-3]"] = 0
#param_half_cell["Initial lithium concentration in positive core [mol.m-3]"] = 44000 
param_half_cell["Initial lithium concentration in positive core [mol.m-3]"] = 40000 

# %%
# Simulate Model 1 - Original SC Model
var_pts = {
"x_n": 20,  # negative electrode
"x_s": 20,  # separator
"x_p": 20,  # positive electrode
"r_n": 10,  # negative particle
"r_co": 20,  # positive particle active core
"r_sh":10,  # positive particle shell
"r_p": 30,
}

sim = pybamm.Simulation(
    model_1,
    experiment=experiment,
    parameter_values=param_half_cell,
    solver=pybamm.IDAKLUSolver(), #options=solver_options
    var_pts=var_pts,
)

# Solve the simulation
solution = sim.solve()

# %%
# Extract relevant data
time_vector_1 = solution.t
V_cell_1 = solution["Battery voltage [V]"].data
I_app_data_1 = solution["Current [A]"].data
X_ave_Oxygen_boundary_flux_1 = solution["X-averaged Oxygen Mass Flux Outer Boundary [mol.m-2.s-1]"].data
sto_surf_p_1 = solution["Positive particle surface stoichiometry [-]"].data
s_dot_ave_1 = solution["X-averaged Boundary Movement speed [m/s]"].data
Pressure_1 = solution["Oxygen pressure in external tank [Pa]"].data
LAM_pe_1 = solution["Average Loss of active material in positive electrode"].data

# %%
# Simulate Model 2 - New SC Model


experiment = pybamm.Experiment(
    [
        (
            f"Charge at {C_rate} C until 4.0 V",
            f"Hold at 4.0 V until {C_rate_cut} C",
            f"Charge at {C_rate} C until 4.2 V",
            f"Hold at 4.2 V until {C_rate_cut} C",
            f"Charge at {C_rate} C until 4.4 V",
            f"Hold at 4.4 V until {C_rate_cut} C",
            f"Charge at {C_rate} C until 4.6 V",
            f"Hold at 4.6 V until 0.06 C",
            f"Discharge at {C_rate} C until 3.7 V",
        )
    ] * 1,
    period=1
)

param_half_cell["Initial oxygen concentration in positive shell [mol.m-3]"] = initial_oxygen_concentration
sim = pybamm.Simulation(
    model_2,
    experiment=experiment,
    #parameter_values=param_2,
    parameter_values=param_half_cell,
    solver=pybamm.IDAKLUSolver(), #options=solver_options
    var_pts=var_pts,
)

# Solve the simulation
solution = sim.solve() #calc_esoh=False

# %%
# Extract relevant data
time_vector_2 = solution.t
V_cell_2 = solution["Battery voltage [V]"].data
I_app_data_2 = solution["Current [A]"].data
X_ave_Oxygen_boundary_flux_2 = solution["X-averaged Oxygen Mass Flux Outer Boundary [mol.m-2.s-1]"].data
sto_surf_p_2 = solution["Positive particle surface stoichiometry [-]"].data
c_o_surf_2 = solution["Surface Oxygen Concentration [mol.m-3]"].data
s_dot_ave_2 = solution["X-averaged Boundary Movement speed [m/s]"].data
LAM_pe_2 = solution["Average Loss of active material in positive electrode"].data

# %%
################################################################################
# 1. IMPORT REQUIRED DATA
################################################################################
# Import Current Data
current_data = pd.read_csv(os.path.join(_DATA_DIR, 'Current_Bethan_Extracted_EC.csv'))
time_vec = current_data['time'].to_numpy()
current_vec = current_data['current'].to_numpy()

# Import Voltage Data
voltage_data = pd.read_csv(os.path.join(_DATA_DIR, 'Voltage_Bethan_Extracted_EC.csv'))
time_vec_v = voltage_data['time'].to_numpy()
voltage_vec = voltage_data['voltage'].to_numpy()

# Import Gas Evolution Data
o2_data = pd.read_csv(os.path.join(_DATA_DIR, 'O2_Bethan_Extracted_EC.csv'))
time_vec_o2 = o2_data['time'].to_numpy()
o2_vec = o2_data['oxygen'].to_numpy()

co2_data = pd.read_csv(os.path.join(_DATA_DIR, 'CO2_Bethan_Extracted_EC.csv'))
time_vec_co2 = co2_data['time'].to_numpy()
co2_vec = co2_data['CO2'].to_numpy()

co_data = pd.read_csv(os.path.join(_DATA_DIR, 'CO_Bethan_Extracted_EC.csv'))
time_vec_co = co_data['time'].to_numpy()
co_vec = co_data['CO'].to_numpy()

# %%
################################################################################
# 2. PREPROCESS DATA
################################################################################
# Sort all time series data
sort_indices = np.argsort(time_vec)
time_vec_sorted = time_vec[sort_indices]
current_vec_sorted = current_vec[sort_indices]

sort_indices_v = np.argsort(time_vec_v)
time_vec_sorted_v = time_vec_v[sort_indices_v]
voltage_vec_sorted = voltage_vec[sort_indices_v]

sort_indices_o2 = np.argsort(time_vec_o2)
time_vec_sorted_o2 = time_vec_o2[sort_indices_o2]
o2_vec_sorted = o2_vec[sort_indices_o2]

sort_indices_co2 = np.argsort(time_vec_co2)
time_vec_sorted_co2 = time_vec_co2[sort_indices_co2]
co2_vec_sorted = co2_vec[sort_indices_co2]

sort_indices_co = np.argsort(time_vec_co)
time_vec_sorted_co = time_vec_co[sort_indices_co]
co_vec_sorted = co_vec[sort_indices_co]

# Create common time base for interpolation
min_time = min(time_vec_sorted_o2[0], time_vec_sorted_co[0], time_vec_sorted_co2[0])
max_time = max(time_vec_sorted_o2[-1], time_vec_sorted_co[-1], time_vec_sorted_co2[-1])
common_time = np.linspace(min_time, max_time, 1000)

# Interpolate all data to common time base
o2_interp = np.interp(common_time, time_vec_sorted_o2, o2_vec_sorted)
co_interp = np.interp(common_time, time_vec_sorted_co, co_vec_sorted)
co2_interp = np.interp(common_time, time_vec_sorted_co2, co2_vec_sorted)
current_interp = np.interp(common_time, time_vec_sorted, current_vec_sorted)
voltage_interp = np.interp(common_time, time_vec_sorted_v, voltage_vec_sorted)

# Calculate total oxygen equivalents
o2_equiv_total = o2_interp + co_interp + 2 * co2_interp



# %%
# Scale plots for better comparison
# Find index where experimental current first exceeds 9 mA/g
time_vector_hrs = time_vector_1/3600  # Convert to hours
time_vector_hrs_2 = time_vector_2/3600  # Convert to hours
start_index = np.where(current_interp > 9)[0][0]
start_time = common_time[start_index]
o2_equiv_total_shifted = o2_equiv_total[start_index:]
voltage_exp_shifted = voltage_interp[start_index:]

# Adjust time vectors to new zero point
adjusted_exp_time = common_time[start_index:] - start_time
adjusted_exp_current = current_interp[start_index:]

# Adjust model time and ensure it starts from zero
adjusted_model_time = time_vector_hrs - time_vector_hrs[0]
adjusted_model_time_2 = time_vector_hrs_2 - time_vector_hrs_2[0]
#adjusted_model_time_3 = time_vector_hrs_3 - time_vector_hrs_3[0]


# %%
# General plot label size
plt.rcParams.update({
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# %%
# Plot results for Fig. 3 - timeseries of  Gass Flux, LAM_pe,  Voltage, Current
MODEL_TITLES = {
    "model_1": "Original SC-P2D Model",
    "model_2": "New SC-P2D Model",
    "Experimental": "Experimental Data",
}

# Model line colours (requested mapping)
MODEL_LINE_COLOURS = {
    "model_1": "red",
    "model_2": "blue",
    "Experimental": "black",
}

# Keep line styles distinct to help readability in grayscale/print
MODEL_LINESTYLES = {"model_1": "-", "model_2": "-", "Experimental": "--"}

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 8), constrained_layout=True)
# 1) Mass Flux

ax0 = axes[0]
ax0_twin = ax0.twinx()

lines_0 = []
lines_0 += ax0.plot(
    adjusted_model_time,
    X_ave_Oxygen_boundary_flux_1,
    color=MODEL_LINE_COLOURS["model_1"],
    linestyle=MODEL_LINESTYLES["model_1"],
    label=MODEL_TITLES["model_1"],
)
lines_0 += ax0.plot(
    adjusted_model_time_2,
    X_ave_Oxygen_boundary_flux_2,
    color=MODEL_LINE_COLOURS["model_2"],
    linestyle=MODEL_LINESTYLES["model_2"],
    label=MODEL_TITLES["model_2"],
)
lines_0 += ax0_twin.plot(
    adjusted_exp_time,
    o2_equiv_total_shifted*(1e-9),
    color=MODEL_LINE_COLOURS["Experimental"],
    linestyle=MODEL_LINESTYLES["Experimental"],
    label=MODEL_TITLES["Experimental"],
)

ax0.set_ylabel("Model \nGas Flux\n[mol m$^{-2}$ s$^{-1}$]")
ax0_twin.set_ylabel("Experimental \nGas Flux\n[mol m$^{-2}$ s$^{-1}$]")
#ax0.set_title("Gas Flux Comparison")
ax0.grid(True, alpha=0.3)

# Combined legend (model + experimental) so entries align
labels_0 = [ln.get_label() for ln in lines_0]
ax0.legend(lines_0, labels_0, loc="best")

# 2 Boundary Movement Speed
axes[1].plot(adjusted_model_time, np.abs(s_dot_ave_1), color=MODEL_LINE_COLOURS["model_1"], linestyle=MODEL_LINESTYLES["model_1"], label=MODEL_TITLES["model_1"])
axes[1].plot(adjusted_model_time_2, np.abs(s_dot_ave_2), color=MODEL_LINE_COLOURS["model_2"], linestyle=MODEL_LINESTYLES["model_2"], label=MODEL_TITLES["model_2"])
axes[1].set_ylabel("RS Interface \nMovement Speed \n[s$^{-1}$]")
axes[1].grid(True, alpha=0.3)


# 4 Voltage
axes[2].plot(adjusted_model_time,V_cell_1,color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[2].plot(adjusted_model_time_2,V_cell_2,color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"])
axes[2].plot(adjusted_exp_time, voltage_exp_shifted, color=MODEL_LINE_COLOURS["Experimental"], linestyle=MODEL_LINESTYLES["Experimental"], label=MODEL_TITLES["Experimental"])
axes[2].set_ylabel("Voltage \n[V]")
axes[2].grid(True, alpha=0.3)

# Bottom: I_app
axes[3].plot(adjusted_model_time,I_app_data_1,color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[3].plot(adjusted_model_time_2,I_app_data_2,color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"])
axes[3].set_xlabel("Time [h]")
axes[3].set_ylabel("Current \n[A]")
axes[3].grid(True, alpha=0.3)

axes[3].set_xlim(0, 20)  # Limit x-axis to 10 hours for better visibility of early behavior

# Panel labels (top-left). Bottom subplot is (a), top is (d).
panel_labels = {3: "(d)", 2: "(c)", 1: "(b)", 0: "(a)"}
for ax_i, label in panel_labels.items():
    axes[ax_i].text(
        0.02,
        0.98,
        label,
        transform=axes[ax_i].transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )

fig.savefig(
    "Fig3_Mechanistic_Validation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# %%
