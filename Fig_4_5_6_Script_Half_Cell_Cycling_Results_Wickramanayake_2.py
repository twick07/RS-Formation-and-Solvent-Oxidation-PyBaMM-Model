# %%
# Script to generate Figs. 4, and 5 in the paper
# Comparison of the original SC model and the new SC model for half-cell cycling
import numpy as np
import matplotlib.pyplot as plt
import pybamm

# %%
# Define Models to be compared
model_1 = pybamm.lithium_ion.DFN_HALF_CELL_Shrinking_Core_T_Zhuo()
model_2 = pybamm.lithium_ion.DFN_HALF_CELL_Shrinking_Core_Wickramanayake()

# %% 
# Define the experiments to be simulated
N_cycles = 10
experiment = pybamm.Experiment(
    [
        (
            "Charge at 0.5 C until 4.2 V",
            "Hold at 4.2 V until C/50",
            "Rest for 60 minutes",
            "Discharge at 0.5 C until 3.7 V",
            "Hold at 3.7 V until C/50",
            "Rest for 60 minutes",
        )
    ] * N_cycles
,period=1)

# %%
# Import parameter values for the half cell model
from Parameter_Sets_Wickramanayake2026 import param_half_cell

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
solution = sim.solve() #calc_esoh=False

Boundary_location_1 = solution["X-averaged positive particle moving phase boundary location"].data
I_app_1 = solution["Current [A]"].data
V_cell_1 = solution["Battery voltage [V]"].data
Cell_SoC_1 = solution["Cell SoC"].data
s_dot_ave_1 = solution["X-averaged Boundary Movement speed [m/s]"].data
time_vector_1 = solution.t
LAM_pe_1 = solution["Average Loss of active material in positive electrode"].data
X_ave_Oxygen_boundary_flux_1 = solution["X-averaged Oxygen Mass Flux Outer Boundary [mol.m-2.s-1]"].data
R_shell_1 = solution["Positive Shell Resistance [Ohm.m2]"].data
Discharge_capacity_1 = solution["Discharge capacity [A.h]"].data
OCP_p_1 = solution["Positive Electrode OCP [V]"].data
sto_surf_p_1 = solution["Positive Electrode Stoichiometry [-]"].data
Shell_overpotential_1 = solution["Shell Overpotential [V]"].data
# %%
# Simulate Model 2 - New SC Model

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

LAM_pe_2 = solution["Average Loss of active material in positive electrode"].data
Boundary_location_2 = solution["X-averaged positive particle moving phase boundary location"].data
I_app_2 = solution["Current [A]"].data
V_cell_2 = solution["Battery voltage [V]"].data
Cell_SoC_2 = solution["Cell SoC"].data
time_vector_2 = solution.t
s_dot_ave_2 = solution["X-averaged Boundary Movement speed [m/s]"].data
X_ave_Oxygen_boundary_flux_2 = solution["X-averaged Oxygen Mass Flux Outer Boundary [mol.m-2.s-1]"].data
R_shell_2 = solution["Positive Shell Resistance [Ohm.m2]"].data
P_cell_2 = solution["Oxygen pressure in external tank [Pa]"].data
T_cell_2 = solution["Surface Temperature [K]"].data
Discharge_capacity_2 = solution["Discharge capacity [A.h]"].data
OCP_p_2 = solution["Positive Electrode OCP [V]"].data
sto_surf_p_2 = solution["Positive Electrode Stoichiometry [-]"].data
Shell_overpotential_2 = solution["Shell Overpotential [V]"].data

# %%
# General plot label size
plt.rcParams.update({
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# %%
# Plot results for Fig. 4 - timeseries of LAM_pe, Gass Flux, R_shell, Cell SOC, Voltage, Current

# Convert time to hours for readability
t_vector_h_1 = time_vector_1 / 3600
t_vector_h_2 = time_vector_2 / 3600

MODEL_TITLES = {
    "model_1": "Original SC-P2D Model",
    "model_2": "New SC-P2D Model",
}

# Model line colours (requested mapping)
MODEL_LINE_COLOURS = {
    "model_1": "red",
    "model_2": "blue",
}

# Keep line styles distinct to help readability in grayscale/print
MODEL_LINESTYLES = {"model_1": "-", "model_2": "-"}

fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=(10, 8), constrained_layout=True)

# 1 LAM_{pe}
axes[0].plot(t_vector_h_1,100 * LAM_pe_1 - 100 * LAM_pe_1[0],   color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[0].plot(t_vector_h_2,100 * LAM_pe_2 - 100 * LAM_pe_2[0],  color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"])
axes[0].set_ylabel("LAM$_{p}$ [%]")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="lower right")

# 2) Mass Flux
axes[1].plot(t_vector_h_1, -1*s_dot_ave_1, color=MODEL_LINE_COLOURS["model_1"], linestyle=MODEL_LINESTYLES["model_1"], label=MODEL_TITLES["model_1"])
axes[1].plot(t_vector_h_2, -1*s_dot_ave_2, color=MODEL_LINE_COLOURS["model_2"], linestyle=MODEL_LINESTYLES["model_2"], label=MODEL_TITLES["model_2"])
axes[1].set_ylabel("Shell/Core \nInterface \nMovement Speed \n[s$^{-1}$]")
axes[1].grid(True, alpha=0.3)

# 3} Shell Resistance
axes[2].plot(t_vector_h_1, R_shell_1, color=MODEL_LINE_COLOURS["model_1"], linestyle=MODEL_LINESTYLES["model_1"], label=MODEL_TITLES["model_1"])
axes[2].plot(t_vector_h_2, R_shell_2, color=MODEL_LINE_COLOURS["model_2"], linestyle=MODEL_LINESTYLES["model_2"], label=MODEL_TITLES["model_2"])
axes[2].set_ylabel("Rocksalt Layer \nResistance \n$[\Omega.m^2]$")
axes[2].grid(True, alpha=0.3)

# 4) Cell SoC
axes[3].plot(t_vector_h_1,100*Cell_SoC_1,color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[3].plot(t_vector_h_2,100*Cell_SoC_2,color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"])
axes[3].set_ylabel("Cell SoC \n[%]")
axes[3].grid(True, alpha=0.3)

# 4 Voltage
axes[4].plot(t_vector_h_1,V_cell_1,color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[4].plot(t_vector_h_2,V_cell_2,color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"])
axes[4].set_ylabel("Voltage \n[V]")
axes[4].grid(True, alpha=0.3)

# Bottom: I_app
axes[5].plot(t_vector_h_1,I_app_1,color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[5].plot(t_vector_h_2,I_app_2,color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"])
axes[5].set_xlabel("Time [h]")
axes[5].set_ylabel("Current \n[A]")
axes[5].grid(True, alpha=0.3)

# Panel labels (top-right). Bottom subplot is (a), top is (f).
panel_labels = {5: "(f)", 4: "(e)", 3: "(d)", 2: "(c)", 1: "(b)", 0: "(a)"}
for ax_i, label in panel_labels.items():
    axes[ax_i].text(
        0.04,
        0.98,
        label,
        transform=axes[ax_i].transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )

fig.savefig(
    "Fig_4_Half_cell_varable_time_series.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# %%
# Define Function To Calculate indexes for positive current intervals (I_app > 0) - used for calculating throughput capacity in Fig. 7
def positive_current_intervals_idx(time_vector, I_app, tol=0.0):
    """
    Find intervals where applied current is positive (I_app > tol).

    Args:
        time_vector: 1D array (only used for length checking; not returned)
        I_app:       1D array of applied current
        tol:         positivity threshold; positive means I_app > tol

    Returns:
        start_idx: np.ndarray of indices where current becomes positive
        end_idx:   np.ndarray of indices where current stops being positive

    Conventions:
        - start_idx[k] is the first index in the positive segment
        - end_idx[k]   is the last index in the positive segment (inclusive)
        - If the series ends while still positive, end_idx will be len(I_app)-1 for that segment
    """
    t = np.asarray(time_vector)
    I = np.asarray(I_app)

    if t.shape != I.shape:
        raise ValueError("time_vector and I_app must have the same shape")

    n = I.size
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    is_pos = I > tol
    d = np.diff(is_pos.astype(np.int8))

    # False -> True transitions (segment starts)
    start_idx = np.where(d == 1)[0] + 1
    if is_pos[0]:
        start_idx = np.r_[0, start_idx]

    # True -> False transitions (segment ends)
    end_idx = np.where(d == -1)[0]  # index of last True before it turns False
    if is_pos[-1]:
        end_idx = np.r_[end_idx, n - 1]

    return start_idx.astype(int), end_idx.astype(int)


start_idx_1, end_idx_1 = positive_current_intervals_idx(time_vector_1, I_app_1)
start_idx_2, end_idx_2 = positive_current_intervals_idx(time_vector_2, I_app_2)

# Integrate the current over the positive intervals to get throughput capacity
throughput_cap_1 = np.zeros([N_cycles])
throughput_cap_2 = np.zeros([N_cycles])
for k in range(N_cycles):
    throughput_cap_1[k] = np.sum(I_app_1[start_idx_1[k]:end_idx_1[k]])
    throughput_cap_2[k] = np.sum(I_app_2[start_idx_2[k]:end_idx_2[k]])

# %%
# SoH vs Cycles
# calculate capacity in Ah from PE electrode parameters
F_const = 96485.3329  # Faraday constant in C/mol
L_p = param_half_cell["Positive electrode thickness [m]"]
eps_p = param_half_cell["Positive electrode active material volume fraction"]
C_max_pos = param_half_cell["Maximum concentration in positive electrode [mol.m-3]"]
init_c_p = param_half_cell["Initial lithium concentration in positive core [mol.m-3]"]
soc_p_init = init_c_p / C_max_pos
init_cap = (1/3600)*F_const*L_p*eps_p*soc_p_init*C_max_pos*(1-soc_p_init)

#init_cap = (1/3600)*F_const*param_half_cell["Positive electrode active material volume fraction"]*param_half_cell["Positive electrode thickness [m]"]*param_half_cell["Positive electrode area [m2]"]/param_half_cell["Positive electrode maximum concentration [mol.m-3]"]
fig_soh = plt.figure(figsize=(8, 5))
plt.plot(np.linspace(1,N_cycles,N_cycles), 100*throughput_cap_1/throughput_cap_1[0], marker='X', color=MODEL_LINE_COLOURS["model_1"], linestyle=MODEL_LINESTYLES["model_1"], label=MODEL_TITLES["model_1"])
plt.plot(np.linspace(1,N_cycles,N_cycles), 100*throughput_cap_2/throughput_cap_2[0], marker='o', color=MODEL_LINE_COLOURS["model_2"], linestyle=MODEL_LINESTYLES["model_2"], label=MODEL_TITLES["model_2"])
plt.xlabel("Cycles [-]")
plt.ylabel("State of Health [%]")
plt.grid(True, alpha=0.3)
plt.legend()

fig_soh.savefig(
    "Fig_5_half_cell_SoH_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
# %%

# Plot results for Fig 7 - Throuput Capacity vs Cycles
fig_cap = plt.figure(figsize=(8, 5))
plt.plot(np.linspace(1,N_cycles,N_cycles), throughput_cap_1/3600, marker='X', color=MODEL_LINE_COLOURS["model_1"], linestyle=MODEL_LINESTYLES["model_1"], label=MODEL_TITLES["model_1"])
plt.plot(np.linspace(1,N_cycles,N_cycles), throughput_cap_2/3600, marker='o', color=MODEL_LINE_COLOURS["model_2"], linestyle=MODEL_LINESTYLES["model_2"], label=MODEL_TITLES["model_2"])
plt.xlabel("Cycles [-]")
plt.ylabel("Discharge Capacity [Ah]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig_cap.savefig(
    "Fig_5_half_cell_Dishcarge_capacity_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# %%

def current_near_value_intervals_idx(time_vector, I_app, I_target=1.675, rel_tol=0.01, abs_tol=0.0):
    """Find index intervals where the applied current is close to a target value.

    This is intended for identifying constant-current steps such as a 1C segment.

    Args:
        time_vector: 1D array (only used for length checking; not returned)
        I_app:       1D array of applied current
        I_target:    target current (e.g. I_1C) in A
        rel_tol:     relative tolerance (fraction). Default 0.05 corresponds to ±5%.
        abs_tol:     absolute tolerance in A (useful if I_target is close to 0)

    Returns:
        start_idx: np.ndarray of indices where |I_app - I_target| enters the tolerance band
        end_idx:   np.ndarray of indices where it leaves the tolerance band

    Conventions:
        - start_idx[k] is the first index in the in-band segment
        - end_idx[k]   is the last index in the in-band segment (inclusive)
        - If the series ends while still in-band, end_idx will be len(I_app)-1
    """
    t = np.asarray(time_vector)
    I = np.asarray(I_app)

    if t.shape != I.shape:
        raise ValueError("time_vector and I_app must have the same shape")

    n = I.size
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    I_target = float(I_target)
    rel_tol = float(rel_tol)
    abs_tol = float(abs_tol)
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("rel_tol and abs_tol must be non-negative")

    tol = max(abs_tol, rel_tol * abs(I_target))
    in_band = np.abs(I - I_target) <= tol
    d = np.diff(in_band.astype(np.int8))

    start_idx = np.where(d == 1)[0] + 1
    if in_band[0]:
        start_idx = np.r_[0, start_idx]

    end_idx = np.where(d == -1)[0]
    if in_band[-1]:
        end_idx = np.r_[end_idx, n - 1]

    return start_idx.astype(int), end_idx.astype(int)


def oneC_current_intervals_idx(time_vector, I_app, I_1C=1.675, tol_frac=0.05, abs_tol=0.0):
    """Convenience wrapper for finding 1C current segments.

    Args:
        time_vector: 1D array of time
        I_app:       1D array of applied current
        I_1C:        1C current in A (default 1.675)
        tol_frac:    fractional tolerance (default 0.05 corresponds to ±5%)
        abs_tol:     absolute tolerance in A

    Returns:
        (start_idx, end_idx) as in `current_near_value_intervals_idx`.
    """
    return current_near_value_intervals_idx(
        time_vector=time_vector,
        I_app=I_app,
        I_target=I_1C,
        rel_tol=tol_frac,
        abs_tol=abs_tol,
    )


start_idx_1, end_idx_1 = oneC_current_intervals_idx(time_vector_1, I_app_1)
start_idx_2, end_idx_2 = oneC_current_intervals_idx(time_vector_2, I_app_2)

# Integrate the current over the positive intervals to get throughput capacity
throughput_cap_1 = np.zeros([N_cycles])
throughput_cap_2 = np.zeros([N_cycles])
for k in range(N_cycles):
    throughput_cap_1[k] = np.sum(I_app_1[start_idx_1[k]:end_idx_1[k]])
    throughput_cap_2[k] = np.sum(I_app_2[start_idx_2[k]:end_idx_2[k]])

# Plot results for Fig 7 - Throuput Capacity vs Cycles
plt.figure(figsize=(8, 5))
plt.plot(np.linspace(1,N_cycles,N_cycles), 100*throughput_cap_1/throughput_cap_1[0], marker='X', color=MODEL_LINE_COLOURS["model_1"], linestyle=MODEL_LINESTYLES["model_1"], label=MODEL_TITLES["model_1"])
plt.plot(np.linspace(1,N_cycles,N_cycles), 100*throughput_cap_2/throughput_cap_2[0], marker='o', color=MODEL_LINE_COLOURS["model_2"], linestyle=MODEL_LINESTYLES["model_2"], label=MODEL_TITLES["model_2"])
plt.xlabel("Cycles [-]")
plt.ylabel("State of Health [%]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# %%
# Stoichiometric drift plots
def rest_current_intervals_idx(time_vector, I_app, I_rest=0.0, abs_tol=1e-6):
    """Convenience wrapper for finding rest (near-zero current) segments.

    Args:
        time_vector: 1D array of time
        I_app:       1D array of applied current
        I_rest:      rest current target in A (default 0.0)
        abs_tol:     absolute tolerance in A around I_rest (default 1e-6)

    Returns:
        (start_idx, end_idx) as in `current_near_value_intervals_idx`, where
        end_idx is inclusive.
    """
    return current_near_value_intervals_idx(
        time_vector=time_vector,
        I_app=I_app,
        I_target=I_rest,
        rel_tol=0.0,
        abs_tol=abs_tol,
    )


def voltage_hit_index(V_cell, start_idx, V_target=3.7, abs_tol=1e-3, end_idx=None):
    """Find the first index where voltage reaches a target (e.g. 3.0 V).

    The search scans forward from start_idx until the voltage is within abs_tol
    of V_target, or until it crosses the target (depending on whether it starts
    above or below V_target).

    Args:
        V_cell:    1D array-like of voltage values [V]
        start_idx: integer index to start scanning from (inclusive)
        V_target:  target voltage in V (default 3.0)
        abs_tol:   absolute tolerance for a "hit" in V (default 1e-3)
        end_idx:   optional inclusive end index for the scan (default: last)

    Returns:
        idx_hit: integer index of the first sample where the voltage hits/crosses
                 V_target.

    Raises:
        ValueError if no hit occurs within the scan window.
    """
    V = np.asarray(V_cell).astype(float)
    n = V.size
    if n == 0:
        raise ValueError("V_cell is empty")

    start_idx = int(start_idx)
    if start_idx < 0 or start_idx >= n:
        raise ValueError(f"start_idx must be in [0, {n-1}], got {start_idx}")

    if end_idx is None:
        end_idx = n - 1
    end_idx = int(end_idx)
    if end_idx < start_idx or end_idx >= n:
        raise ValueError(f"end_idx must be in [{start_idx}, {n-1}], got {end_idx}")

    V_target = float(V_target)
    abs_tol = float(abs_tol)
    if abs_tol < 0:
        raise ValueError("abs_tol must be non-negative")

    V0 = V[start_idx]
    window = V[start_idx : end_idx + 1]

    # Exact/near hit
    hit_mask = np.abs(window - V_target) <= abs_tol
    if np.any(hit_mask):
        return start_idx + int(np.argmax(hit_mask))

    # Otherwise, detect the first crossing in the expected direction.
    # If starting above target, look for first value <= target.
    # If starting below target, look for first value >= target.
    if V0 > V_target:
        cross_mask = window <= (V_target + abs_tol)
    else:
        cross_mask = window >= (V_target - abs_tol)

    if np.any(cross_mask):
        return start_idx + int(np.argmax(cross_mask))

    raise ValueError(
        f"Voltage did not reach {V_target} V (±{abs_tol} V) between indices {start_idx} and {end_idx}"
    )

# Extract the time points for the first and last cycle
start_idx_1, end_idx_1 = rest_current_intervals_idx(time_vector_1, I_app_1)
start_idx_2, end_idx_2 = rest_current_intervals_idx(time_vector_2, I_app_2)
#start_idx_1, end_idx_1 = positive_current_intervals_idx(time_vector_1, I_app_1)
#start_idx_2, end_idx_2 = positive_current_intervals_idx(time_vector_2, I_app_2)
# find the start points of the rest phase:
idx_first_start_1 = end_idx_1[0]
idx_first_end_1 = voltage_hit_index(V_cell_1, idx_first_start_1, V_target=3.7, abs_tol=1e-3)

idx_last_start_1 = end_idx_1[-2]
idx_last_end_1 = voltage_hit_index(V_cell_1, idx_last_start_1, V_target=3.7, abs_tol=1e-3)

idx_first_start_2 = end_idx_2[0]
idx_first_end_2 = voltage_hit_index(V_cell_2, idx_first_start_2, V_target=3.7, abs_tol=1e-3)

idx_last_start_2 = end_idx_2[-2]
idx_last_end_2 = voltage_hit_index(V_cell_2, idx_last_start_2, V_target=3.7, abs_tol=1e-3)

# %%

# plot the cell voltage vs Cell SoC for the first and last cycle for both models
time_shift =0
plt.figure(figsize=(8, 5))
plt.plot(Cell_SoC_1[idx_first_start_1-time_shift:idx_first_end_1], V_cell_1[idx_first_start_1-time_shift:idx_first_end_1], linestyle=MODEL_LINESTYLES["model_1"], color=MODEL_LINE_COLOURS["model_1"], label=f"{MODEL_TITLES['model_1']} - Cycle 1")
plt.plot(Cell_SoC_1[idx_last_start_1-time_shift:idx_last_end_1], V_cell_1[idx_last_start_1-time_shift:idx_last_end_1], linestyle=MODEL_LINESTYLES["model_1"], color=MODEL_LINE_COLOURS["model_1"], label=f"{MODEL_TITLES['model_1']} - Cycle {N_cycles} Cell Voltage")
plt.plot(Cell_SoC_2[idx_first_start_2-time_shift:idx_first_end_2], V_cell_2[idx_first_start_2-time_shift:idx_first_end_2], linestyle=MODEL_LINESTYLES["model_2"], color=MODEL_LINE_COLOURS["model_2"], label=f"{MODEL_TITLES['model_2']} - Cycle 1 Cell Voltage")
plt.plot(Cell_SoC_2[idx_last_start_2-time_shift:idx_last_end_2], V_cell_2[idx_last_start_2-time_shift:idx_last_end_2], linestyle=MODEL_LINESTYLES["model_2"], color=MODEL_LINE_COLOURS["model_2"], label=f"{MODEL_TITLES['model_2']} - Cycle {N_cycles} Cell Voltage")     
plt.xlabel("Cell SoC [%]")
plt.ylabel("Cell Voltage [V]")
plt.gca().invert_xaxis()
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()



# %%
# Plot Voltage vs SoC for the original SC model for the first and last cycle

from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

fig_vsoc, ax_vsoc = plt.subplots(figsize=(8, 5))
fig_vsoc.subplots_adjust(top=0.82)

# Linestyle encodes cycle
ls_cycle_1 = "-"
ls_cycle_N = "--"

# Colour encodes signal
signal_colours = {
    "NMC811 OCP$_p$": "blue",
    "Cell voltage": "green",
    "Graphite OCP$_n$": "red",
    "Shell overpotential": "black",
}

# Plot lines (suppress per-line legend entries; we build custom legends below)
ax_vsoc.plot(
    100 * Cell_SoC_1[idx_first_start_1:idx_first_end_1],
    V_cell_1[idx_first_start_1:idx_first_end_1],
    linestyle=ls_cycle_1,
    color=signal_colours["Cell voltage"],
    label="_nolegend_",
)
ax_vsoc.plot(
    100 * Cell_SoC_1[idx_last_start_1:idx_last_end_1],
    V_cell_1[idx_last_start_1:idx_last_end_1],
    linestyle=ls_cycle_N,
    color=signal_colours["Cell voltage"],
    label="_nolegend_",
)

ax_vsoc.plot(
    100 * Cell_SoC_1[idx_first_start_1:idx_first_end_1],
    OCP_p_1[-1, idx_first_start_1:idx_first_end_1],
    linestyle=ls_cycle_1,
    color=signal_colours["NMC811 OCP$_p$"],
    label="_nolegend_",
)
ax_vsoc.plot(
    100 * Cell_SoC_1[idx_last_start_1:idx_last_end_1],
    OCP_p_1[-1, idx_last_start_1:idx_last_end_1],
    linestyle=ls_cycle_N,
    color=signal_colours["NMC811 OCP$_p$"],
    label="_nolegend_",
)

ax_vsoc.plot(
    100 * Cell_SoC_1[idx_first_start_1:idx_first_end_1],
    Shell_overpotential_1[-1, idx_first_start_1:idx_first_end_1],
    linestyle=ls_cycle_1,
    color=signal_colours["Shell overpotential"],
    label="_nolegend_",
)
ax_vsoc.plot(
    100 * Cell_SoC_1[idx_last_start_1:idx_last_end_1],
    Shell_overpotential_1[-1, idx_last_start_1:idx_last_end_1],
    linestyle=ls_cycle_N,
    color=signal_colours["Shell overpotential"],
    label="_nolegend_",
)

ax_vsoc.set_xlabel("Cell SoC [%]")
ax_vsoc.set_ylabel("Voltage [V]")
ax_vsoc.invert_xaxis()
ax_vsoc.grid(True, alpha=0.3)

# ---- Right y-axis ticks at requested Shell overpotential points + grid lines ----
shell_tick_points = np.array(
    [
        Shell_overpotential_1[-1, idx_first_end_1],
        Shell_overpotential_1[-1, idx_last_end_1],
    ],
    dtype=float,
)

ax_vsoc_right = ax_vsoc.twinx()
ax_vsoc_right.set_ylim(ax_vsoc.get_ylim())
ax_vsoc_right.set_yticks(shell_tick_points)
ax_vsoc_right.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
#ax_vsoc_right.set_ylabel("Shell overpotential markers [V]")

# Draw horizontal grid lines at these specific points
for yv in shell_tick_points:
    ax_vsoc.axhline(
        y=yv,
        color=signal_colours["Shell overpotential"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        zorder=0,
    )

# Custom x-ticks at key SoC points (rounded to 3 d.p.)
soc_tick_positions = np.round(
    [
        100 * Cell_SoC_1[idx_first_start_1],
        100 * Cell_SoC_1[idx_first_end_1],
        100 * Cell_SoC_1[idx_last_end_1],
        100 * Cell_SoC_1[idx_last_start_1],
    ],
    3,
)
ax_vsoc.set_xticks(soc_tick_positions[[0, 1, 2]])
ax_vsoc.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_vsoc.minorticks_off()

# Put the final-cycle start SoC tick on a second x-axis at the top to avoid overlap
ax_vsoc_top = ax_vsoc.twiny()
ax_vsoc_top.set_xlim(ax_vsoc.get_xlim())
ax_vsoc_top.set_xticks([soc_tick_positions[3]])
ax_vsoc_top.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_vsoc_top.minorticks_off()
ax_vsoc_top.set_xlabel("")

# Legend (ii): colour mapping (centre right)
colour_handles = [
    Line2D([0], [0], color=signal_colours["NMC811 OCP$_p$"], linestyle="-", linewidth=2.0, label="NMC811 OCP$_p$"),
    Line2D([0], [0], color=signal_colours["Cell voltage"], linestyle="-", linewidth=2.0, label="Cell voltage"),
    # Line2D([0], [0], color=signal_colours["Graphite OCP$_n$"], linestyle="-", linewidth=2.0, label="Graphite OCP$_n$"),
    Line2D([0], [0], color=signal_colours["Shell overpotential"], linestyle="-", linewidth=2.0, label="Shell overpotential"),
]
leg_colours = ax_vsoc.legend(handles=colour_handles, loc="center right", frameon=True)
ax_vsoc.add_artist(leg_colours)

# Legend (i): cycle mapping (outside top-left and top-right)
first_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1")
last_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}")

leg_cycle_1 = ax_vsoc.legend(
    handles=[first_cycle_handle],
    loc="lower left",
    bbox_to_anchor=(0.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)
ax_vsoc.add_artist(leg_cycle_1)

ax_vsoc.legend(
    handles=[last_cycle_handle],
    loc="lower right",
    bbox_to_anchor=(1.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)

plt.show()

plt.figure(figsize=(8, 5))

# Export figure to .png
fig_vsoc.savefig(
    "Fig_6a_Half_cell_Voltage_vs_SoC_comparison_OG.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
# Stoichiometry vs Cell SoC for old SC model
from matplotlib.lines import Line2D

fig_sto, ax_sto_p = plt.subplots(figsize=(8, 5))
#ax_sto_n = ax_sto_p.twinx()

electrode_colours = {"NMC811 cathode": "blue", "Graphite anode": "red"}

# Colour encodes electrode; linestyle encodes cycle
ls_cycle_1 = "-"
ls_cycle_N = "--"

ax_sto_p.plot(
    100*Cell_SoC_1[idx_first_start_1:idx_first_end_1],
    100*sto_surf_p_1[-1, idx_first_start_1:idx_first_end_1],
    linestyle=ls_cycle_1,
    color=electrode_colours["NMC811 cathode"],
    linewidth=2.0,
    label="_nolegend_",
)
ax_sto_p.plot(
    100*Cell_SoC_1[idx_last_start_1:idx_last_end_1],
    100*sto_surf_p_1[-1, idx_last_start_1:idx_last_end_1],
    linestyle=ls_cycle_N,
    color=electrode_colours["NMC811 cathode"],
    linewidth=2.0,
    label="_nolegend_",
)



ax_sto_p.set_xlabel("Cell SoC [%]")
ax_sto_p.set_ylabel("NMC811 cathode stoichiometry [%]")
#ax_sto_n.set_ylabel("Graphite anode stoichiometry [%]")

ax_sto_p.invert_xaxis()
ax_sto_p.grid(True, alpha=0.3)

soc_tick_positions = np.round(
    [
        100*Cell_SoC_1[idx_first_start_1],
        100*Cell_SoC_1[idx_first_end_1],
        100*Cell_SoC_1[idx_last_end_1],
        100*Cell_SoC_1[idx_last_start_1],
    ],
    3,
)
ax_sto_p.set_xticks(soc_tick_positions[[0, 1, 2]])
ax_sto_p.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_sto_p.minorticks_off()

# Put the final-cycle start SoC tick on a second x-axis at the top to avoid overlap
ax_sto_p_top = ax_sto_p.twiny()
ax_sto_p_top.set_xlim(ax_sto_p.get_xlim())
ax_sto_p_top.set_xticks([soc_tick_positions[3]])
ax_sto_p_top.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_sto_p_top.minorticks_off()
ax_sto_p_top.set_xlabel("")

# Custom y-ticks at key stoichiometry points
sto_p_tick_positions = np.array(
    [
        100 * sto_surf_p_1[-1, idx_first_start_1],
        100 * sto_surf_p_1[-1, idx_first_end_1],
        100 * sto_surf_p_1[-1, idx_last_start_1],
        100* sto_surf_p_1[-1, idx_last_end_1]
        
    ],
    dtype=float,
)


ax_sto_p.set_yticks(sto_p_tick_positions)
ax_sto_p.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
#ax_sto_n.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

# Two compact legends: electrode (colour) and cycle (linestyle)
electrode_handles = [
    Line2D([0], [0], color=electrode_colours["NMC811 cathode"], linestyle="-", linewidth=2.0, label="NMC811 cathode"),
    #Line2D([0], [0], color=electrode_colours["Graphite anode"], linestyle="-", linewidth=2.0, label="Graphite anode"),
]
cycle_handles = [
    Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1"),
    Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}"),
]

leg1 = ax_sto_p.legend(handles=electrode_handles, loc="right", frameon=True)
ax_sto_p.add_artist(leg1)
#ax_sto_p.legend(handles=cycle_handles, loc="upper center", frameon=True)

# Legend (i): cycle mapping (outside top-left and top-right)
first_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1")
last_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}")

leg_cycle_1 = ax_sto_p.legend(
    handles=[first_cycle_handle],
    loc="lower left",
    bbox_to_anchor=(0.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)
ax_sto_p.add_artist(leg_cycle_1)

ax_sto_p.legend(
    handles=[last_cycle_handle],
    loc="lower right",
    bbox_to_anchor=(1.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)

plt.show()


plt.show()

# %%
# Stoichiometry vs Cell Voltage for original SC model
idx_first_CV_end_1 = start_idx_1[1]-1
idx_last_CV_end_1 = start_idx_1[-1]-1
from matplotlib.lines import Line2D

fig_sto, ax_sto_p = plt.subplots(figsize=(8, 5))
#ax_sto_n = ax_sto_p.twinx()

electrode_colours = {"NMC811 cathode": "blue", "Graphite anode": "red"}

# Colour encodes electrode; linestyle encodes cycle
ls_cycle_1 = "-"
ls_cycle_N = "--"

ax_sto_p.plot(
    #100*Cell_SoC_1[idx_first_start_1:idx_first_CV_end_1],
    V_cell_1[idx_first_start_1:idx_first_end_1],
    100*sto_surf_p_1[-1, idx_first_start_1:idx_first_end_1],
    linestyle=ls_cycle_1,
    color=electrode_colours["NMC811 cathode"],
    linewidth=2.0,
    label="_nolegend_",
)
ax_sto_p.plot(
    #100*Cell_SoC_1[idx_last_start_1:idx_last_CV_end_1],
    V_cell_1[idx_last_start_1:idx_last_end_1],
    100*sto_surf_p_1[-1, idx_last_start_1:idx_last_end_1],
    linestyle=ls_cycle_N,
    color=electrode_colours["NMC811 cathode"],
    linewidth=2.0,
    label="_nolegend_",
)




#ax_sto_p.set_xlabel("Cell SoC [%]")
ax_sto_p.set_xlabel("Cell Voltage [V]")
ax_sto_p.set_ylabel("NMC811 cathode stoichiometry [%]")
#ax_sto_n.set_ylabel("Graphite anode stoichiometry [%]")

ax_sto_p.invert_xaxis()
ax_sto_p.grid(True, alpha=0.3)

'''
soc_tick_positions = np.round(
    [
        100*Cell_SoC_1[idx_first_start_1],
        100*Cell_SoC_1[idx_first_CV_end_1],
        100*Cell_SoC_1[idx_last_CV_end_1],
        100*Cell_SoC_1[idx_last_start_1],
    ],
    3,
)
'''
voltage_tick_positions = np.round(
    [
        V_cell_1[idx_first_start_1],
        V_cell_1[idx_first_end_1],
        V_cell_1[idx_last_end_1],
        V_cell_1[idx_last_start_1],
    ],
    2,
)
#ax_sto_p.set_xticks(soc_tick_positions)
ax_sto_p.set_xticks(voltage_tick_positions)
ax_sto_p.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_sto_p.minorticks_off()

# Custom y-ticks at key stoichiometry points
sto_p_tick_positions = np.array(
    [
        100 * sto_surf_p_1[-1, idx_first_start_1],
        100 * sto_surf_p_1[-1, idx_first_end_1],
        100 * sto_surf_p_1[-1, idx_last_start_1],
        100 * sto_surf_p_1[-1, idx_last_end_1],
    ],
    dtype=float,
)


ax_sto_p.set_yticks(sto_p_tick_positions)
#ax_sto_n.set_yticks(sto_n_tick_positions)
ax_sto_p.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
#ax_sto_n.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# Two compact legends: electrode (colour) and cycle (linestyle)
electrode_handles = [
    #Line2D([0], [0], color=electrode_colours["NMC811 cathode"], linestyle="-", linewidth=2.0, label="NMC811 cathode"),
    #Line2D([0], [0], color=electrode_colours["Graphite anode"], linestyle="-", linewidth=2.0, label="Graphite anode"),
]
cycle_handles = [
    Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1"),
    Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}"),
]

leg1 = ax_sto_p.legend(handles=electrode_handles, loc="right", frameon=True)
ax_sto_p.add_artist(leg1)
#ax_sto_p.legend(handles=cycle_handles, loc="upper center", frameon=True)

# Legend (i): cycle mapping (outside top-left and top-right)
first_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1")
last_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}")

leg_cycle_1 = ax_sto_p.legend(
    handles=[first_cycle_handle],
    loc="lower left",
    bbox_to_anchor=(0.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)
ax_sto_p.add_artist(leg_cycle_1)

ax_sto_p.legend(
    handles=[last_cycle_handle],
    loc="lower right",
    bbox_to_anchor=(1.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)

plt.show()


plt.show()

# Export figure to .png
fig_sto.savefig(
    "Fig_6b_Half_cell_Stoic_vs_voltage_OG.png",
    dpi=300,
    bbox_inches="tight",
)



# %%
# Plot Voltage vs SoC for the  new SC model for the first and last cycle

from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

fig_vsoc, ax_vsoc = plt.subplots(figsize=(8, 5))
fig_vsoc.subplots_adjust(top=0.82)

# Linestyle encodes cycle
ls_cycle_1 = "-"
ls_cycle_N = "--"

# Colour encodes signal
signal_colours = {
    "NMC811 OCP$_p$": "blue",
    "Cell voltage": "green",
    "Graphite OCP$_n$": "red",
    "Shell overpotential": "black",
}

# Plot lines (suppress per-line legend entries; we build custom legends below)
ax_vsoc.plot(
    100 * Cell_SoC_2[idx_first_start_2:idx_first_end_2],
    V_cell_2[idx_first_start_2:idx_first_end_2],
    linestyle=ls_cycle_1,
    color=signal_colours["Cell voltage"],
    label="_nolegend_",
)
ax_vsoc.plot(
    100 * Cell_SoC_2[idx_last_start_2:idx_last_end_2],
    V_cell_2[idx_last_start_2:idx_last_end_2],
    linestyle=ls_cycle_N,
    color=signal_colours["Cell voltage"],
    label="_nolegend_",
)

ax_vsoc.plot(
    100 * Cell_SoC_2[idx_first_start_2:idx_first_end_2],
    OCP_p_2[-1, idx_first_start_2:idx_first_end_2],
    linestyle=ls_cycle_1,
    color=signal_colours["NMC811 OCP$_p$"],
    label="_nolegend_",
)
ax_vsoc.plot(
    100 * Cell_SoC_2[idx_last_start_2:idx_last_end_2],
    OCP_p_2[-1, idx_last_start_2:idx_last_end_2],
    linestyle=ls_cycle_N,
    color=signal_colours["NMC811 OCP$_p$"],
    label="_nolegend_",
)

ax_vsoc.plot(
    100 * Cell_SoC_2[idx_first_start_2:idx_first_end_2],
    Shell_overpotential_2[-1, idx_first_start_2:idx_first_end_2],
    linestyle=ls_cycle_1,
    color=signal_colours["Shell overpotential"],
    label="_nolegend_",
)
ax_vsoc.plot(
    100 * Cell_SoC_2[idx_last_start_2:idx_last_end_2],
    Shell_overpotential_2[-1, idx_last_start_2:idx_last_end_2],
    linestyle=ls_cycle_N,
    color=signal_colours["Shell overpotential"],
    label="_nolegend_",
)

ax_vsoc.set_xlabel("Cell SoC [%]")
ax_vsoc.set_ylabel("Voltage [V]")
ax_vsoc.invert_xaxis()
ax_vsoc.grid(True, alpha=0.3)

# ---- Right y-axis ticks at requested Shell overpotential points + grid lines ----
shell_tick_points = np.array(
    [
        Shell_overpotential_2[-1, idx_last_end_2],
        Shell_overpotential_2[-1, idx_first_end_2],
    ],
    dtype=float,
)

ax_vsoc_right = ax_vsoc.twinx()
ax_vsoc_right.set_ylim(ax_vsoc.get_ylim())
ax_vsoc_right.set_yticks(shell_tick_points)
ax_vsoc_right.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
#ax_vsoc_right.set_ylabel("Shell overpotential markers [V]")

# Draw horizontal grid lines at these specific points
for yv in shell_tick_points:
    ax_vsoc.axhline(
        y=yv,
        color=signal_colours["Shell overpotential"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        zorder=0,
    )

# Custom x-ticks at key SoC points (rounded to 3 d.p.)
soc_tick_positions = np.round(
    [
        100 * Cell_SoC_2[idx_first_start_2],
        100 * Cell_SoC_2[idx_first_end_2],
        100 * Cell_SoC_2[idx_last_end_2],
        100 * Cell_SoC_2[idx_last_start_2],
    ],
    3,
)
ax_vsoc.set_xticks(soc_tick_positions[[0, 1, 2]])
ax_vsoc.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_vsoc.minorticks_off()

# Put the final-cycle start SoC tick on a second x-axis at the top to avoid overlap
ax_vsoc_top = ax_vsoc.twiny()
ax_vsoc_top.set_xlim(ax_vsoc.get_xlim())
ax_vsoc_top.set_xticks([soc_tick_positions[3]])
ax_vsoc_top.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_vsoc_top.minorticks_off()
ax_vsoc_top.set_xlabel("")

# Legend (ii): colour mapping (centre right)
colour_handles = [
    Line2D([0], [0], color=signal_colours["NMC811 OCP$_p$"], linestyle="-", linewidth=2.0, label="NMC811 OCP$_p$"),
    Line2D([0], [0], color=signal_colours["Cell voltage"], linestyle="-", linewidth=2.0, label="Cell voltage"),
    # Line2D([0], [0], color=signal_colours["Graphite OCP$_n$"], linestyle="-", linewidth=2.0, label="Graphite OCP$_n$"),
    Line2D([0], [0], color=signal_colours["Shell overpotential"], linestyle="-", linewidth=2.0, label="Shell overpotential"),
]
leg_colours = ax_vsoc.legend(handles=colour_handles, loc="center right", frameon=True)
ax_vsoc.add_artist(leg_colours)

# Legend (i): cycle mapping (outside top-left and top-right)
first_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1")
last_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}")

leg_cycle_1 = ax_vsoc.legend(
    handles=[first_cycle_handle],
    loc="lower left",
    bbox_to_anchor=(0.6, 1.02),
    frameon=True,
    borderaxespad=0.0,
)
ax_vsoc.add_artist(leg_cycle_1)

ax_vsoc.legend(
    handles=[last_cycle_handle],
    loc="lower right",
    bbox_to_anchor=(1.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)

plt.show()

plt.figure(figsize=(8, 5))

# Export figure to .png
fig_vsoc.savefig(
    "Fig_6c_Half_cell_Voltage_vs_SoC_comparison_New.png",
    dpi=300,
    bbox_inches="tight",
)
# %%
# Stoichiometry vs Cell SoC for new SC model
from matplotlib.lines import Line2D

fig_sto, ax_sto_p = plt.subplots(figsize=(8, 5))
#ax_sto_n = ax_sto_p.twinx()

electrode_colours = {"NMC811 cathode": "blue", "Graphite anode": "red"}

# Colour encodes electrode; linestyle encodes cycle
ls_cycle_1 = "-"
ls_cycle_N = "--"

ax_sto_p.plot(
    100*Cell_SoC_2[idx_first_start_2:idx_first_end_2],
    100*sto_surf_p_2[-1, idx_first_start_2:idx_first_end_2],
    linestyle=ls_cycle_1,
    color=electrode_colours["NMC811 cathode"],
    linewidth=2.0,
    label="_nolegend_",
)
ax_sto_p.plot(
    100*Cell_SoC_2[idx_last_start_2:idx_last_end_2],
    100*sto_surf_p_2[-1, idx_last_start_2:idx_last_end_2],
    linestyle=ls_cycle_N,
    color=electrode_colours["NMC811 cathode"],
    linewidth=2.0,
    label="_nolegend_",
)




ax_sto_p.set_xlabel("Cell SoC [%]")
ax_sto_p.set_ylabel("NMC811 cathode stoichiometry [%]")
#ax_sto_n.set_ylabel("Graphite anode stoichiometry [%]")

ax_sto_p.invert_xaxis()
ax_sto_p.grid(True, alpha=0.3)

soc_tick_positions = np.round(
    [
        100*Cell_SoC_2[idx_first_start_2],
        100*Cell_SoC_2[idx_first_end_2],
        100*Cell_SoC_2[idx_last_end_2],
        100*Cell_SoC_2[idx_last_start_2],
    ],
    3,
)
ax_sto_p.set_xticks(soc_tick_positions[[0, 1, 2]])
ax_sto_p.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_sto_p.minorticks_off()

# Put the final-cycle start SoC tick on a second x-axis at the top to avoid overlap
ax_sto_p_top = ax_sto_p.twiny()
ax_sto_p_top.set_xlim(ax_sto_p.get_xlim())
ax_sto_p_top.set_xticks([soc_tick_positions[3]])
ax_sto_p_top.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_sto_p_top.minorticks_off()
ax_sto_p_top.set_xlabel("")

# Custom y-ticks at key stoichiometry points
sto_p_tick_positions = np.array(
    [
        100 * sto_surf_p_2[-1, idx_first_start_2],
        100 * sto_surf_p_2[-1, idx_first_end_2],
        100 * sto_surf_p_2[-1, idx_last_start_2],
        100 * sto_surf_p_2[-1, idx_last_end_2],
    ],
    dtype=float,
)


ax_sto_p.set_yticks(sto_p_tick_positions)
#ax_sto_n.set_yticks(sto_n_tick_positions)
ax_sto_p.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
#ax_sto_n.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# Two compact legends: electrode (colour) and cycle (linestyle)
electrode_handles = [
    Line2D([0], [0], color=electrode_colours["NMC811 cathode"], linestyle="-", linewidth=2.0, label="NMC811 cathode"),
    Line2D([0], [0], color=electrode_colours["Graphite anode"], linestyle="-", linewidth=2.0, label="Graphite anode"),
]
cycle_handles = [
    Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1"),
    Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}"),
]

leg1 = ax_sto_p.legend(handles=electrode_handles, loc="right", frameon=True)
ax_sto_p.add_artist(leg1)
#ax_sto_p.legend(handles=cycle_handles, loc="upper center", frameon=True)

# Legend (i): cycle mapping (outside top-left and top-right)
first_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1")
last_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}")

leg_cycle_1 = ax_sto_p.legend(
    handles=[first_cycle_handle],
    loc="lower left",
    bbox_to_anchor=(0.6, 1.02),
    frameon=True,
    borderaxespad=0.0,
)
ax_sto_p.add_artist(leg_cycle_1)

ax_sto_p.legend(
    handles=[last_cycle_handle],
    loc="lower right",
    bbox_to_anchor=(1.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)

plt.show()


plt.show()


# %%
# Stoichiometry vs Cell Voltage for new SC model
idx_first_CV_end_2 = start_idx_2[1]-1
idx_last_CV_end_2 = start_idx_2[-1]-1
from matplotlib.lines import Line2D

fig_sto, ax_sto_p = plt.subplots(figsize=(8, 5))
#ax_sto_n = ax_sto_p.twinx()

electrode_colours = {"NMC811 cathode": "blue", "Graphite anode": "red"}

# Colour encodes electrode; linestyle encodes cycle
ls_cycle_1 = "-"
ls_cycle_N = "--"

ax_sto_p.plot(
    #100*Cell_SoC_2[idx_first_start_2:idx_first_CV_end_2],
    V_cell_2[idx_first_start_2:idx_first_end_2],
    100*sto_surf_p_2[-1, idx_first_start_2:idx_first_end_2],
    linestyle=ls_cycle_1,
    color=electrode_colours["NMC811 cathode"],
    linewidth=2.0,
    label="_nolegend_",
)
ax_sto_p.plot(
    #100*Cell_SoC_2[idx_last_start_2:idx_last_CV_end_2],
    V_cell_2[idx_last_start_2:idx_last_end_2],
    100*sto_surf_p_2[-1, idx_last_start_2:idx_last_end_2],
    linestyle=ls_cycle_N,
    color=electrode_colours["NMC811 cathode"],
    linewidth=2.0,
    label="_nolegend_",
)



#ax_sto_p.set_xlabel("Cell SoC [%]")
ax_sto_p.set_xlabel("Cell Voltage [V]")
ax_sto_p.set_ylabel("NMC811 cathode stoichiometry [%]")


ax_sto_p.invert_xaxis()
ax_sto_p.grid(True, alpha=0.3)


voltage_tick_positions = np.round(
    [
        V_cell_2[idx_first_start_2],
        V_cell_2[idx_first_CV_end_2],
        V_cell_2[idx_last_CV_end_2],
        V_cell_2[idx_last_start_2],
    ],
    2,
)
#ax_sto_p.set_xticks(soc_tick_positions)
ax_sto_p.set_xticks(voltage_tick_positions)
ax_sto_p.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_sto_p.minorticks_off()

# Custom y-ticks at key stoichiometry points
sto_p_tick_positions = np.array(
    [
        100 * sto_surf_p_2[-1, idx_first_start_2],
        100 * sto_surf_p_2[-1, idx_first_end_2],
        100 * sto_surf_p_2[-1, idx_last_start_2],
        100 * sto_surf_p_2[-1, idx_last_end_2],
    ],
    dtype=float,
)

ax_sto_p.set_yticks(sto_p_tick_positions)
ax_sto_p.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
#ax_sto_n.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# Two compact legends: electrode (colour) and cycle (linestyle)
electrode_handles = [
    #Line2D([0], [0], color=electrode_colours["NMC811 cathode"], linestyle="-", linewidth=2.0, label="NMC811 cathode"),
    #Line2D([0], [0], color=electrode_colours["Graphite anode"], linestyle="-", linewidth=2.0, label="Graphite anode"),
]
cycle_handles = [
    Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1"),
    Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}"),
]

leg1 = ax_sto_p.legend(handles=electrode_handles, loc="right", frameon=True)
ax_sto_p.add_artist(leg1)
ax_sto_p.legend(handles=cycle_handles, loc="upper center", frameon=True)

# Legend (i): cycle mapping (outside top-left and top-right)
first_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_1, linewidth=2.0, label="Cycle 1")
last_cycle_handle = Line2D([0], [0], color="black", linestyle=ls_cycle_N, linewidth=2.0, label=f"Cycle {N_cycles}")

leg_cycle_1 = ax_sto_p.legend(
    handles=[first_cycle_handle],
    loc="lower left",
    bbox_to_anchor=(0.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)
ax_sto_p.add_artist(leg_cycle_1)

ax_sto_p.legend(
    handles=[last_cycle_handle],
    loc="lower right",
    bbox_to_anchor=(1.0, 1.02),
    frameon=True,
    borderaxespad=0.0,
)

plt.show()


plt.show()

# Export figure to .png
fig_sto.savefig(
    "Fig_6d_Half_cell_Stoic_vs_voltage_New.png",
    dpi=300,
    bbox_inches="tight",
)


# %%
