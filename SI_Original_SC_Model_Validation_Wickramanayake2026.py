# %%
# Script to generate Figures in Section S4 of the Supplementary Information
# Script compares the original SC model from Zhuo et al, with a custom implementation
import os
import numpy as np
import matplotlib.pyplot as plt
import pybamm

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

# %%
# Define Models to be compared
model_1 = pybamm.lithium_ion.DFN({"PE degradation": "phase transition"})
model_2 = pybamm.lithium_ion.DFN_FULL_CELL_Shrinking_Core_T_ave()

# %% 
# Define the experiments to be simulated
experiment = pybamm.Experiment(
    [
        (
            "Charge at 0.5 C until 4.2 V",
            "Hold at 4.2 V until C/50",
            "Rest for 60 minutes",
            "Discharge at 0.5 C until 3.0 V",
            "Rest for 60 minutes",
        )
    ] * 10
,period=1)
# %%
# Define Model Parameters
# Note, these model parameters are taken from Zhuo et al. 2023.



# Import Required CSV Data
sic_18650_ocp_Zhuo2023_data = pybamm.parameters.process_1D_data("sic_18650_ocp_Zhuo2023.csv", path=_DATA_DIR)
sic_18650_dUdT_Zhuo2023_data = pybamm.parameters.process_1D_data("sic_18650_dUdT_Zhuo2023_2.csv", path=_DATA_DIR)
nmc811_18650_ocp_Zhuo2023_data = pybamm.parameters.process_1D_data("nmc811_18650_ocp_Zhuo2023.csv", path=_DATA_DIR)
nmc811_18650_dUdT_Zhuo2023_data = pybamm.parameters.process_1D_data("nmc811_18650_dUdT_Zhuo2023_2.csv", path=_DATA_DIR)

# Define Required Parameters
def sic_18650_ocp_Zhuo2023(sto):
    name, (x, y) = sic_18650_ocp_Zhuo2023_data
    return pybamm.Interpolant(x, y, sto, name=name)


def sic_18650_dUdT_Zhuo2023(sto):
    name, (x, y) = sic_18650_dUdT_Zhuo2023_data
    return pybamm.Interpolant(x, y, sto, name=name)


def nmc811_18650_ocp_Zhuo2023(sto):
    name, (x, y) = nmc811_18650_ocp_Zhuo2023_data
    return pybamm.Interpolant(x, y, sto, name=name)


def nmc811_18650_dUdT_Zhuo2023(sto):
    name, (x, y) = nmc811_18650_dUdT_Zhuo2023_data
    return pybamm.Interpolant(x, y, sto, name=name)


def sic_18650_electrolyte_exchange_current_density_Zhuo2023(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
        [1] Sturm, J., et al. "Modeling and Simulation of Inhomogeneities in a
    18650 Nickel-Rich, Silicon-Graphite Lithium-Ion Cell during Fast Charging".
    Journal of Power Sources, vol. 412, Feb. 2019, pp. 204–23.
    doi: 10.1016/j.jpowsour.2018.11.043.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    m_ref = 1.0e-11 * pybamm.constants.F  # [m2.5.mol-0.5.s-1]
    E_r = 29931.48
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def nmc811_18650_electrolyte_exchange_current_density_Zhuo2023(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
        [1] Sturm, J., et al. "Modeling and Simulation of Inhomogeneities in a
    18650 Nickel-Rich, Silicon-Graphite Lithium-Ion Cell during Fast Charging".
    Journal of Power Sources, vol. 412, Feb. 2019, pp. 204–23.
    doi: 10.1016/j.jpowsour.2018.11.043.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 3.2e-11 * pybamm.constants.F  # [m2.5.mol-0.5.s-1]
    E_r = 29931.48
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def sic_18650_diffusivity_Zhuo2023(c_s, T):
    """
    SiC diffusivity as a function of temperature.

    References
    ----------
        [1] Ghosh, Abir, et al. "A Shrinking-Core Model for the Degradation of
    High-Nickel Cathodes (NMC811) in Li-Ion Batteries: Passivation Layer Growth
    and Oxygen Evolution." Journal of The Electrochemical Society,
    168(2) (2021): 020509.

    Parameters
    ----------
    c_s: :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 1.0e-14

    aEne = 9977.16

    arrhenius = np.exp(aEne / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def nmc811_18650_diffusivity_Zhuo2023(c_s, T):
    """
    NMC811 diffusivity as a function of temperature.

    References
    ----------
        [1] Ghosh, Abir, et al. "A Shrinking-Core Model for the Degradation of
    High-Nickel Cathodes (NMC811) in Li-Ion Batteries: Passivation Layer Growth
    and Oxygen Evolution." Journal of The Electrochemical Society,
    168(2) (2021): 020509.

    Parameters
    ----------
    c_s: :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 1.0e-14

    aEne = 9977.16

    arrhenius = np.exp(aEne / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius

def nmc811_18650_diffusivity_Zhuo2023_2(c_s, c_max_pos, T):
    """
    NMC811 diffusivity as a function of temperature.

    References
    ----------
        [1] Ghosh, Abir, et al. "A Shrinking-Core Model for the Degradation of
    High-Nickel Cathodes (NMC811) in Li-Ion Batteries: Passivation Layer Growth
    and Oxygen Evolution." Journal of The Electrochemical Society,
    168(2) (2021): 020509.

    Parameters
    ----------
    c_s: :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    
    D_ref = 1.0e-14

    aEne = 9977.16

    arrhenius = np.exp(aEne / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def electrolyte_diffusivity_Capiglia1999(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration. The original data
    is from [1]. The fit from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 5.34e-10 * np.exp(-0.65 * c_e / 1000)
    E_D_e = 37040
    arrhenius = np.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def electrolyte_conductivity_Capiglia1999(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration. The original
    data is from [1]. The fit is from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid conductivity
    """

    sigma_e = (
        0.0911
        + 1.9101 * (c_e / 1000)
        - 1.052 * (c_e / 1000) ** 2
        + 0.1554 * (c_e / 1000) ** 3
    )

    E_k_e = 34700
    arrhenius = np.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius


def initial_oxygen_concentration(r_sh_nd, x):
    """
     Initial oxygen concentration profile.

     References
     ----------
     .. [1] Ghosh, Abir, et al. "A Shrinking-Core Model for the Degradation of
     High-Nickel Cathodes (NMC811) in Li-Ion Batteries: Passivation Layer Growth
     and Oxygen Evolution." Journal of The Electrochemical Society,
     168(2) (2021): 020509.
     .. [2] Mingzhao Zhuo, Gregory Offer, Monica Marinescu, "Degradation model of
    high-nickel positive electrodes: Effects of loss of active material and
    cyclable lithium on capacity fade", Journal of Power Sources},
    556 (2023): 232461. doi: 10.1016/j.jpowsour.2022.232461.

     Parameters
     ----------
     r_sh_nd: :class:`pybamm.SpatialVariable`
         r_sh / R_typ, Dimensionless spatial variable (cartesian) in changed 1D domain,
         varying from 0 to 1

     # R_typ: :class:`pybamm.FunctionParameter`
     #     The typical particle radius in the middle of the electrode

     x: :class:`pybamm.SpatialVariable`
         through-cell distance (x) [m]

     Returns
     -------
     :class:`pybamm.Symbol`
         Initial concentration
    """

    c_o_ini_ref = 15219.321

    return c_o_ini_ref * ((1 - r_sh_nd) ** 2)



param_dict = {
    "chemistry": "lithium_ion",
    # sei
    "Ratio of lithium moles to SEI moles": 2.0,
    "Inner SEI reaction proportion": 0.5,
    "Inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
    "Outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
    "SEI reaction exchange current density [A.m-2]": 1.5e-07,
    "SEI resistivity [Ohm.m]": 2.0e5,
    "Outer SEI solvent diffusivity [m2.s-1]": 2.5e-22,
    "Bulk solvent concentration [mol.m-3]": 2636.0,
    "Inner SEI open-circuit potential [V]": 0.1,
    "Outer SEI open-circuit potential [V]": 0.8,
    "Inner SEI electron conductivity [S.m-1]": 8.95e-14,
    "Inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
    "Lithium interstitial reference concentration [mol.m-3]": 15.0,
    "Initial inner SEI thickness [m]": 2.5e-09,
    "Initial outer SEI thickness [m]": 2.5e-09,
    "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
    "EC diffusivity [m2.s-1]": 2e-18,
    "SEI kinetic rate constant [m.s-1]": 1e-12,
    "SEI open-circuit potential [V]": 0.4,
    "SEI growth activation energy [J.mol-1]": 0.0,
    "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
    "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
    # PE degradation caused by phase transition
    "Positive shell oxygen diffusivity [m2.s-1]": 1e-17,
    "Forward chemical reaction coefficient [m.s-1]": 0.8544e-11,
    "Reverse chemical reaction coefficient [m4.mol-1.s-1]": 1.732e-16,
    "Trapped lithium concentration in the shell [mol.m-3]": 20000,  # for Fig 7 in Zhuo2023
    "Minimum concentration in positive electrode when fully discharged [mol.m-3]": 10953.48,  # 0.222 * c_max_p
    "Minimum concentration in negative electrode when fully discharged [mol.m-3]": 68.514,  # 0.002 * c_max_n
    "Threshold lithium concentration for phase transition [mol.m-3]": 15000,  # 0.3 * c_max_p
    "Positive electrode shell resistivity [Ohm.m]": 1e6,  # Safari2009
    "Constant oxygen concentration in the core [mol.m-3]": 152193.21,
    "Initial lithium concentration in positive core [mol.m-3]": 46478.28,  # 0.942 * c_max_p
    "Initial oxygen concentration in positive shell [mol.m-3]": initial_oxygen_concentration,
    "Initial core-shell phase boundary location": 0.9868421,  # 3.75e-6 / 3.8e-6
    # cell
    "Negative electrode thickness [m]": 86.7e-6,
    "Separator thickness [m]": 12e-6,
    "Positive electrode thickness [m]": 66.2e-6,
    "Electrode height [m]": 5.8e-2,
    "Electrode width [m]": 1.23,
    "Cell cooling surface area [m2]": 0.01,
    "Cell volume [m3]": 2.42e-05,
    "Negative current collector conductivity [S.m-1]": 58411000.0,
    "Positive current collector conductivity [S.m-1]": 36914000.0,
    "Negative current collector density [kg.m-3]": 8960.0,
    "Positive current collector density [kg.m-3]": 2700.0,
    "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
    "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
    #"Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
    "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
    "Nominal cell capacity [A.h]": 3.35,
    "Current function [A]": 3.35,
    "Contact resistance [Ohm]": 0,
    # negative electrode
    "Negative electrode conductivity [S.m-1]": 100.0,
    "Maximum concentration in negative electrode [mol.m-3]": 34257.0,
    "Negative electrode diffusivity [m2.s-1]": sic_18650_diffusivity_Zhuo2023,
    "Negative electrode OCP [V]": sic_18650_ocp_Zhuo2023,
    "Negative electrode porosity": 0.216,
    "Negative electrode active material volume fraction": 0.694,
    "Negative particle radius [m]": 6.1e-06,
    "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Negative electrode Bruggeman coefficient (electrode)": 1.5,
    "Negative electrode charge transfer coefficient": 0.5,
    "Negative electrode double-layer capacity [F.m-2]": 0.2,
    "Negative electrode exchange-current density [A.m-2]": sic_18650_electrolyte_exchange_current_density_Zhuo2023,
    "Negative electrode density [kg.m-3]": 1657.0,
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 918.8,
    "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
    "Negative electrode OCP entropic change [V.K-1]": sic_18650_dUdT_Zhuo2023,
    # positive electrode
    "Positive electrode conductivity [S.m-1]": 0.17,
    "Maximum concentration in positive electrode [mol.m-3]": 49340.0,
    "Positive electrode diffusivity [m2.s-1]": nmc811_18650_diffusivity_Zhuo2023,
    "Positive electrode OCP [V]": nmc811_18650_ocp_Zhuo2023,
    "Positive electrode porosity": 0.171,
    "Positive electrode active material volume fraction": 0.745,
    "Positive particle radius [m]": 3.8e-6,
    "Positive electrode Bruggeman coefficient (electrolyte)": 1.85,
    "Positive electrode Bruggeman coefficient (electrode)": 1.5,
    "Positive electrode charge transfer coefficient": 0.5,
    "Positive electrode double-layer capacity [F.m-2]": 0.2,
    "Positive electrode exchange-current density [A.m-2]": nmc811_18650_electrolyte_exchange_current_density_Zhuo2023,
    "Positive electrode density [kg.m-3]": 3262.0,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
    "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
    "Positive electrode OCP entropic change [V.K-1]": nmc811_18650_dUdT_Zhuo2023,
    # separator
    "Separator porosity": 0.45,
    "Separator Bruggeman coefficient (electrolyte)": 1.5,
    "Separator density [kg.m-3]": 397.0,
    "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
    "Separator thermal conductivity [W.m-1.K-1]": 0.16,
    # electrolyte
    "Initial concentration in electrolyte [mol.m-3]": 1000.0,
    "Cation transference number": 0.4,
    "Thermodynamic factor": 1.0,
    "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Capiglia1999,
    "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Capiglia1999,
    # experiment
    "Reference temperature [K]": 298.15,
    "Ambient temperature [K]": 298.15,
    "Number of electrodes connected in parallel to make a cell": 1.0,
    "Number of cells connected in series to make a battery": 1.0,
    "Lower voltage cut-off [V]": 2.8,
    "Upper voltage cut-off [V]": 4.2,
    "Open-circuit voltage at 0% SOC [V]": 2.8,
    "Open-circuit voltage at 100% SOC [V]": 4.2,
    "Initial concentration in negative electrode [mol.m-3]": 68.514,
    "Initial concentration in positive electrode [mol.m-3]": 46478.28,  # 0.942 * c_max_p
    # thermal
    "Initial temperature [K]": 298.15,
    "Total heat transfer coefficient [W.m-2.K-1]": 100,
    "Positive current collector thickness [m]": 1.6e-05,
    "Negative current collector thickness [m]": 1.2e-05,
    # citations
    "citations": ["Zhuo2023"],
}
param = pybamm.ParameterValues(param_dict)

# %%
# Simulate Model 1 - Original Zhuo et al. SC Model
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
    parameter_values=param,
    solver=pybamm.IDAKLUSolver(), #options=solver_options
    var_pts=var_pts,
)

# Solve the simulation
solution = sim.solve() #calc_esoh=False

# Extract data into numpy arrays



LAM_pe_1 = solution["X-averaged loss of active material due to PE phase transition"].data
Boundary_location_1 = solution["X-averaged positive particle moving phase boundary location"].data
I_app_1 = solution["Current [A]"].data
V_cell_1 = solution["Terminal voltage [V]"].data
Cell_SoC_1 = solution["Cell SoC"].data
s_dot_ave_1 = solution["X-averaged time derivative of positive particle moving phase boundary location [s-1]"].data
time_vector_1 = solution.t

# %%
# Simulate Model 2 - Custom implementation of Zhuo et al. SC Model
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
    model_2,
    experiment=experiment,
    parameter_values=param,
    solver=pybamm.IDAKLUSolver(), #options=solver_options
    var_pts=var_pts,
)

# Solve the simulation
solution = sim.solve() #calc_esoh=False

# Extract data into numpy arrays
LAM_pe_2 = solution["Average Loss of active material in positive electrode"].data
Boundary_location_2 = solution["X-averaged positive particle moving phase boundary location"].data
I_app_2 = solution["Current [A]"].data
V_cell_2 = solution["Battery voltage [V]"].data
Cell_SoC_2 = solution["Cell SoC"].data
s_dot_ave_2 = solution["X-averaged Boundary Movement speed [m/s]"].data
time_vector_2 = solution.t

# %%
# General plot label size
plt.rcParams.update({
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# %%
# Plot the results - time series of key variables for both models, to compare the two implementations
# Convert time to hours for readability
t_vector_h_1 = time_vector_1 / 3600
t_vector_h_2 = time_vector_2 / 3600

MODEL_TITLES = {
    "model_1": "Zhuo et al. Original Implementation",
    "model_2": "Zhuo et al. Custom Implementation",
}

# Model line colours (requested mapping)
MODEL_LINE_COLOURS = {
    "model_1": "red",
    "model_2": "blue",
}

# Keep line styles distinct to help readability in grayscale/print
MODEL_LINESTYLES = {"model_1": "-", "model_2": ":"}

MODEL_MARKERS = {"model_1": "none", "model_2": "*"}

fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(10, 8), constrained_layout=True)

# 1 LAM_{pe}
axes[0].plot(t_vector_h_1,100 * LAM_pe_1 - 100 * LAM_pe_1[0],   color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[0].plot(t_vector_h_2,100 * LAM_pe_2 - 100 * LAM_pe_2[0],  color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"],marker=MODEL_MARKERS["model_2"], markevery=0.05)
axes[0].set_ylabel("LAM$_{p}$ [%]")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="lower right")

# 2) Mass Flux
axes[1].plot(t_vector_h_1, -1*s_dot_ave_1, color=MODEL_LINE_COLOURS["model_1"], linestyle=MODEL_LINESTYLES["model_1"], label=MODEL_TITLES["model_1"])
axes[1].plot(t_vector_h_2, -1*s_dot_ave_2, color=MODEL_LINE_COLOURS["model_2"], linestyle=MODEL_LINESTYLES["model_2"], label=MODEL_TITLES["model_2"],marker=MODEL_MARKERS["model_2"], markevery=0.05)
axes[1].set_ylabel("Shell/Core \nInterface \nMovement Speed \n[s$^{-1}$]")
axes[1].grid(True, alpha=0.3)


# 4) Cell SoC
axes[2].plot(t_vector_h_1,100*Cell_SoC_1,color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[2].plot(t_vector_h_2,100*Cell_SoC_2,color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"],marker=MODEL_MARKERS["model_2"], markevery=0.05)
axes[2].set_ylabel("Cell SoC \n[%]")
axes[2].grid(True, alpha=0.3)

# 4 Voltage
axes[3].plot(t_vector_h_1,V_cell_1,color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[3].plot(t_vector_h_2,V_cell_2,color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"],marker=MODEL_MARKERS["model_2"], markevery=0.05)
axes[3].set_ylabel("Voltage \n[V]")
axes[3].grid(True, alpha=0.3)

# Bottom: I_app
axes[4].plot(t_vector_h_1,I_app_1,color=MODEL_LINE_COLOURS["model_1"],linestyle=MODEL_LINESTYLES["model_1"],label=MODEL_TITLES["model_1"])
axes[4].plot(t_vector_h_2,I_app_2,color=MODEL_LINE_COLOURS["model_2"],linestyle=MODEL_LINESTYLES["model_2"],label=MODEL_TITLES["model_2"],marker=MODEL_MARKERS["model_2"], markevery=0.05)
axes[4].set_xlabel("Time [h]")
axes[4].set_ylabel("Current \n[A]")
axes[4].grid(True, alpha=0.3)

# Panel labels (top-right). Bottom subplot is (a), top is (e).
panel_labels = {4: "(e)", 3: "(d)", 2: "(c)", 1: "(b)", 0: "(a)"}
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
    "Fig_S4_Zhuo_models_time_series_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# %%
# Function to calculate RMSE values
import numpy as np

def align_and_compute_error(t1: np.ndarray, y1: np.ndarray,
                             t2: np.ndarray, y2: np.ndarray) -> float:
    """
    Aligns two time-series signals and computes the mean absolute percentage error,
    ignoring time points where the reference signal is zero (NaN or Inf entries).

    Steps:
        1. Identifies which time vector ends earlier and uses it as the reference.
        2. Interpolates the longer signal at the shorter signal's time points.
        3. Computes the elementwise absolute percentage error vector.
        4. Removes NaN and Inf entries (caused by zero-valued reference points).
        5. Returns the mean of the remaining valid entries.

    Parameters:
        t1 : time vector for signal 1
        y1 : signal 1 values
        t2 : time vector for signal 2
        y2 : signal 2 values

    Returns:
        error_percent : scalar mean absolute percentage error (NaNs and Infs excluded)
    """
    # Step 1: Select the shorter time vector as reference
    if t1[-1] <= t2[-1]:
        t_ref, y_ref = t1, y1
        t_long, y_long = t2, y2
    else:
        t_ref, y_ref = t2, y2
        t_long, y_long = t1, y1

    # Step 2: Interpolate the longer signal at the reference time points
    y_long_interp = np.interp(t_ref, t_long, y_long)

    # Steps 3-5: Compute elementwise error vector, strip NaNs and Infs, then average
    error_percent = _mape(y_ref, y_long_interp)

    return error_percent


def _mape(y_reference: np.ndarray, y_compare: np.ndarray) -> float:
    """
    Computes the Mean Absolute Percentage Error (MAPE) between two aligned signals,
    ignoring time points that produce NaN (0/0) or Inf (non-zero/0) entries.

        error_vector[i] = |y_compare[i] - y_reference[i]| / |y_reference[i]|
        error_percent   = mean( error_vector[i] for all i where error_vector[i] is finite )

    Parameters:
        y_reference : reference signal (denominator), zero entries produce NaN/Inf and are excluded
        y_compare   : signal to compare against the reference

    Returns:
        error_percent : scalar MAPE value with NaN and Inf entries excluded from the mean
    """
    # Step 3: Compute the full elementwise error vector (zero denominator -> NaN or Inf)
    error_vector = np.abs(y_compare - y_reference) / np.abs(y_reference)

    # Step 4: Remove NaN and Inf entries using np.isfinite (keeps only finite values)
    error_vector_clean = error_vector[np.isfinite(error_vector)]

    # Step 5: Mean of remaining valid entries
    error_percent = np.mean(error_vector_clean)

    return error_percent
# %%
# Compute error between the two models for LAM_pe and Boundary Location
error_LAM_pe = align_and_compute_error(t_vector_h_1, LAM_pe_1, t_vector_h_2, LAM_pe_2)
error_boundary_movement_speed = align_and_compute_error(t_vector_h_1, s_dot_ave_1, t_vector_h_2, s_dot_ave_2)
error_SoC = align_and_compute_error(t_vector_h_1, Cell_SoC_1, t_vector_h_2, Cell_SoC_2)
error_voltage = align_and_compute_error(t_vector_h_1, V_cell_1, t_vector_h_2, V_cell_2)
error_current = align_and_compute_error(t_vector_h_1, I_app_1, t_vector_h_2, I_app_2)   

print(f"Mean Absolute Percentage Error between the two models:")
print(f"LAM_pe: {error_LAM_pe:.5f}%")
print(f"Boundary Movement Speed: {error_boundary_movement_speed:.5f}%")
print(f"State of Charge (SoC): {error_SoC:.5f}%")
print(f"Voltage: {error_voltage:.5f}%")
print(f"Current: {error_current:.5f}%")
# %%
