# %%
# Script to generate Figs. 6, 7, and 9 in the paper
# Comparison of the original SC model and the new SC model for full-cell cycling
import numpy as np
import matplotlib.pyplot as plt
import pybamm

# %%
# Define Model
model = pybamm.lithium_ion.DFN_FULL_CELL_Shrinking_Core_Wickramanayake()


# %%
# Load parameter values
from Parameter_Sets_Wickramanayake2026 import param_full_cell
Heat_Transfer_Coefficient_Test = [100,50,10]

# Set Different  heat transfer coefficients
param_full_cell['Total heat transfer coefficient [W.m-2.K-1]'] = Heat_Transfer_Coefficient_Test[0]
# %% 
# Define the experiments to be simulated
N_cycles = 15
experiment = pybamm.Experiment(
    [
        (
            "Charge at 0.5 C until 4.2 V",
            "Hold at 4.2 V until C/50",
            "Rest for 60 minutes",
            "Discharge at 0.5 C until 3.0 V",
            "Hold at 3.0 V until C/50",
            "Rest for 60 minutes",
        )
    ] * N_cycles
,period=60)


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
    model,
    experiment=experiment,
    parameter_values=param_full_cell,
    solver=pybamm.IDAKLUSolver(), #options=solver_options
    var_pts=var_pts,
)


# %%
# Solve the simulation for case 1
solution = sim.solve() #calc_esoh=False

I_app_1 = solution["Current [A]"].data
V_cell_1 = solution["Battery voltage [V]"].data
X_ave_Oxygen_boundary_flux_1 = solution["X-averaged Oxygen Mass Flux Outer Boundary [mol.m-2.s-1]"].data
time_vector_1 = solution.t
P_cell_1 = solution["Oxygen pressure in external tank [Pa]"].data
T_cell_1 = solution["Surface Temperature [K]"].data
LAM_pe_1 = solution["Average Loss of active material in positive electrode"].data


# Set Different  heat transfer coefficients
param_full_cell['Total heat transfer coefficient [W.m-2.K-1]'] = Heat_Transfer_Coefficient_Test[1]
# %% 
# Define the experiments to be simulated


sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=param_full_cell,
    solver=pybamm.IDAKLUSolver(), #options=solver_options
    var_pts=var_pts,
)


# %%
# Solve the simulation for case 1
solution = sim.solve() #calc_esoh=False

I_app_2 = solution["Current [A]"].data
V_cell_2 = solution["Battery voltage [V]"].data
X_ave_Oxygen_boundary_flux_2 = solution["X-averaged Oxygen Mass Flux Outer Boundary [mol.m-2.s-1]"].data
time_vector_2 = solution.t
P_cell_2 = solution["Oxygen pressure in external tank [Pa]"].data
T_cell_2 = solution["Surface Temperature [K]"].data
LAM_pe_2 = solution["Average Loss of active material in positive electrode"].data

# Set Different  heat transfer coefficients
param_full_cell['Total heat transfer coefficient [W.m-2.K-1]'] = Heat_Transfer_Coefficient_Test[2]
# %% 
# Define the experiments to be simulated

sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=param_full_cell,
    solver=pybamm.IDAKLUSolver(), #options=solver_options
    var_pts=var_pts,
)


# %%
# Solve the simulation for case 1
solution = sim.solve() #calc_esoh=False
I_app_3 = solution["Current [A]"].data
V_cell_3 = solution["Battery voltage [V]"].data
X_ave_Oxygen_boundary_flux_3 = solution["X-averaged Oxygen Mass Flux Outer Boundary [mol.m-2.s-1]"].data
time_vector_3 = solution.t
P_cell_3 = solution["Oxygen pressure in external tank [Pa]"].data
T_cell_3 = solution["Surface Temperature [K]"].data
LAM_pe_3 = solution["Average Loss of active material in positive electrode"].data


# %%
# General plot label size
plt.rcParams.update({
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# %%
# Plot results - timeseries of pressure change, surface temperature, gas flux, voltage, current
T_abs_zero = 273.15

case_data = [
    {
        "t": time_vector_1,
        "P": P_cell_1,
        "T": T_cell_1,
        "flux": X_ave_Oxygen_boundary_flux_1,
        "V": V_cell_1,
        "I": I_app_1,
        "LAM_pe": LAM_pe_1,
        "h": Heat_Transfer_Coefficient_Test[0],
    },
    {
        "t": time_vector_2,
        "P": P_cell_2,
        "T": T_cell_2,
        "flux": X_ave_Oxygen_boundary_flux_2,
        "V": V_cell_2,
        "I": I_app_2,
        "LAM_pe": LAM_pe_2,
        "h": Heat_Transfer_Coefficient_Test[1],
    },
    {
        "t": time_vector_3,
        "P": P_cell_3,
        "T": T_cell_3,
        "flux": X_ave_Oxygen_boundary_flux_3,
        "V": V_cell_3,
        "I": I_app_3,
        "LAM_pe": LAM_pe_3,
        "h": Heat_Transfer_Coefficient_Test[2],
    },
]

case_styles = [
    {"color": "tab:blue", "linestyle": "-"},
    {"color": "tab:green", "linestyle": "--"},
    {"color": "tab:red", "linestyle": "--"},
]

fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=(10, 9), constrained_layout=True)

for i, (case, sty) in enumerate(zip(case_data, case_styles)):
    t_h = np.asarray(case["t"]) / 3600
    label = f"h = {case['h']}"

    # 1) Cell Pressure
    axes[0].plot(
        t_h,
        (case["P"] - case["P"][0]) / (1e6),
        color=sty["color"],
        linestyle=sty["linestyle"],
        label=label,
    )

    # 2) Cell Temperature
    axes[1].plot(
        t_h,
        case["T"] - T_abs_zero,
        color=sty["color"],
        linestyle=sty["linestyle"],
    )

    # 3) Mass Flux
    axes[2].plot(
        t_h,
        case["flux"],
        color=sty["color"],
        linestyle=sty["linestyle"],
    )

    # 4) LAM_pe (above cell voltage)
    axes[3].plot(
        t_h,
        100*case["LAM_pe"],
        color=sty["color"],
        linestyle=sty["linestyle"],
    )

    # 5) Cell Voltage
    axes[4].plot(
        t_h,
        case["V"],
        color=sty["color"],
        linestyle=sty["linestyle"],
    )

    # 6) Applied Current
    axes[5].plot(
        t_h,
        case["I"],
        color=sty["color"],
        linestyle=sty["linestyle"],
    )

axes[0].set_ylabel("$\\Delta$ Pressure \n[MPa]")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="lower right")

# Venting threshold line on pressure subplot
venting_threshold_mpa = 3.5
axes[0].axhline(venting_threshold_mpa, color="k", linestyle=":", linewidth=1.5)
axes[0].annotate(
    "Venting Threshold Pressure",
    xy=(0.3, venting_threshold_mpa),
    xycoords=("axes fraction", "data"),
    xytext=(0, -4),
    textcoords="offset points",
    ha="right",
    va="top",
    fontsize=10,
)

axes[1].set_ylabel("Surface \nTemperature \n[°C]")
axes[1].grid(True, alpha=0.3)

axes[2].set_ylabel("Oxidation \nGas Flux\n[mol m$^{-2}$ s$^{-1}$]")
axes[2].grid(True, alpha=0.3)

axes[3].set_ylabel("LAM$_{p}$\n[%]")
axes[3].grid(True, alpha=0.3)

axes[4].set_ylabel("Cell Voltage \n[V]")
axes[4].grid(True, alpha=0.3)

axes[5].set_ylabel("Current \n[A]")
axes[5].set_xlabel("Time [h]")
axes[5].grid(True, alpha=0.3)

# Panel labels (bottom is (a), top is (f))
panel_labels = {5: "(f)", 4: "(e)", 3: "(d)", 2: "(c)", 1: "(b)", 0: "(a)"}
for ax_i, label in panel_labels.items():
    axes[ax_i].text(
        0.02,
        0.95,
        label,
        transform=axes[ax_i].transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )

plt.show()


# Export figure to .png
fig.savefig(
    "Fig_10_Internal_Pressure_Change_Plot_Wickramanayake.png",
    dpi=300,
    bbox_inches="tight",
)
# %%
