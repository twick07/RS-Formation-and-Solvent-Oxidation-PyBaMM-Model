# %%
# Script to generate Figs. 1 and 2 in the paper
# Illustrating the operation of the new shrinking core half-cell model

# %%
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
# Define Model
model = pybamm.lithium_ion.DFN_HALF_CELL_Shrinking_Core_Wickramanayake()


# %% 
# Define the experiments to be simulated
# Define experiment protocol
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
    ] * 2,
    period=1
)

# %%
# Import parameter set
from Parameter_Sets_Wickramanayake2026 import param_half_cell, nmc811_18650_diffusivity_Wickramanayake2026


# %%
# Simulate Model - New SC Model
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
    #parameter_values=param_2,
    parameter_values=param_half_cell,
    solver=pybamm.IDAKLUSolver(), #options=solver_options
    var_pts=var_pts,
)

# Solve the simulation
solution = sim.solve() #calc_esoh=False

# Extract relevant data
time_vector_3 = solution.t
V_cell_3 = solution["Battery voltage [V]"].data
I_app_data_3 = solution["Current [A]"].data
X_ave_Oxygen_boundary_flux_3 = solution["X-averaged Oxygen Mass Flux Outer Boundary [mol.m-2.s-1]"].data
sto_surf_p_3 = solution["Positive particle surface stoichiometry [-]"].data
c_s_surf_3 = solution["X-Average positive core surface lithium concentration [mol.m-3]"].data
c_o_surf_3 = solution["Surface Oxygen Concentration [mol.m-3]"].data
s_dot_ave_3 = solution["X-averaged positive particle moving phase boundary location"].data
LAM_pe_3 = solution["Average Loss of active material in positive electrode"].data
Ox_r_average = solution["R-average postive shell oxygen concentration [mol.m-3]"].data
X_ave_Oxygen_boundary_flux_inner_3 = solution["X-averaged Oxygen Mass Flux Inner Boundary [mol.m-2.s-1]"].data
c_o_surf_distribution = solution["Surface Oxygen Concentration Distribution [mol.m-3]"].data
c_o_shell = solution["Positive shell oxygen concentration [mol.m-3]"].data
c_o_ave_shell = solution["Average Oxygen Concentration [mol.m-3]"].data
c_o_r_avg = solution["X-averaged oxygen concentration in shell [mol.m-3]"].data
c_c_r_avg = solution["X-averaged positive core lithium concentration [mol.m-3]"].data

# %%
# General plot label size
plt.rcParams.update({
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
# %%
# Plot Fig 1: Model Operation
# Stacked plots (top-to-bottom): Mass flux, LAM_pe, s_dot_ave, Voltage, current
RS_conc_thresh = param_half_cell["Threshold lithium concentration for phase transition [mol.m-3]"]
t_hr = time_vector_3 / 3600

fig, axs = plt.subplots(
    5,
    1,
    sharex=True,
    figsize=(10, 12),
    constrained_layout=True,
)

x = 3000
axs[0].plot(t_hr, X_ave_Oxygen_boundary_flux_3, label="Solvent Oxidation Flux")
axs[0].plot(t_hr, X_ave_Oxygen_boundary_flux_inner_3, label="RS Formation Flux")
axs[0].set_ylabel("Mass flux\n[mol m$^{-2}$ s$^{-1}$]")
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc="upper right",)


axs[1].plot(t_hr, c_o_ave_shell)
axs[1].set_ylabel("Average \nShell Oxygen \nConcentration\n[mol m$^{-3}$]")
axs[1].grid(True, alpha=0.3)

axs[2].plot(t_hr, c_s_surf_3)
axs[2].set_ylabel("Surface \nCore Lithium \nConcentration\n[mol m$^{-3}$]")
axs[2].axhline(
    RS_conc_thresh,
    color="red",
    linestyle="--",
    linewidth=1.0,
    alpha=0.6,
    zorder=0,
    label="RS Formation Threshold",
)
axs[2].legend()
axs[2].grid(True, alpha=0.3)


axs[3].plot(t_hr, V_cell_3)
axs[3].set_ylabel("Voltage\n[V]")
axs[3].grid(True, alpha=0.3)

axs[4].plot(t_hr, I_app_data_3)
axs[4].set_ylabel("Current\n[A]")
axs[4].set_xlabel("Time [h]")
axs[4].grid(True, alpha=0.3)

# --- Choose sampled time points (EDIT THESE) ---
# These time points (in hours) are used for:
#  - stacked-plot markers/labels (t1..tN)
#  - the two line-profile plots below
#  - the heatmaps below
#
# Set to a list/array of ANY length (e.g. 4, 5, 8...), or set to None to use a
# default evenly-spaced sampling.
#manual_time_points_hr = [0.5, 1.8, 7.5, 10.4]
manual_time_points_hr = [0.5, 3.5, 6.5, 10.9, 15]

if manual_time_points_hr is None:
    n_time_markers = 4
    marker_idxs = np.linspace(0, len(t_hr) - 1, n_time_markers, dtype=int)
else:
    manual_time_points_hr = np.asarray(manual_time_points_hr, dtype=float)
    n_time_markers = int(manual_time_points_hr.size)
    if n_time_markers < 1:
        raise ValueError("manual_time_points_hr must contain at least one time point")

    if np.any(manual_time_points_hr < float(t_hr[0])) or np.any(
        manual_time_points_hr > float(t_hr[-1])
    ):
        raise ValueError(
            f"Manual time points must be within [{float(t_hr[0])}, {float(t_hr[-1])}] hours"
        )

    marker_idxs = np.array(
        [int(np.argmin(np.abs(t_hr - t))) for t in manual_time_points_hr], dtype=int
    )
    if len(set(marker_idxs.tolist())) != len(marker_idxs):
        raise ValueError(
            "Two or more manual time points mapped to the same index; choose more separated times."
        )

t_mark = t_hr[marker_idxs]

# Capture the original plotted lines (before we add marker-only lines)
base_lines_by_ax = [list(ax.get_lines()) for ax in axs]

def _add_time_markers(ax, line):
    y = np.asarray(line.get_ydata())
    if y.shape[0] != len(t_hr):
        return
    ax.plot(
        t_mark,
        y[marker_idxs],
        linestyle="None",
        marker="x",
        color=line.get_color(),
        markersize=7,
        markeredgewidth=1.8,
        zorder=5,
    )

for ax_i in (1, 2):
    ax = axs[ax_i]
    for line in base_lines_by_ax[ax_i]:
        _add_time_markers(ax, line)

# Label each marker on each line in each subplot
def _label_time_markers(ax, line, line_index: int):
    y_full = np.asarray(line.get_ydata())
    if y_full.shape[0] != len(t_hr):
        return
    y = y_full[marker_idxs]
    y_offset = 8 if (line_index % 2 == 0) else -14
    va = "bottom" if y_offset > 0 else "top"
    for i, (x, y_i) in enumerate(zip(t_mark, y), start=1):
        ax.annotate(
            f"t{i}",
            xy=(x, y_i),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            color=line.get_color(),
        )

for ax_i in (1, 2):
    ax = axs[ax_i]
    base_lines = base_lines_by_ax[ax_i]
    for line_index, line in enumerate(base_lines):
        _label_time_markers(ax, line, line_index)

for ax in axs:
    ax.grid(True, alpha=0.3)

# Panel labels (top-left). Bottom subplot is (a), top is (e).
panel_labels = {4: "(e)", 3: "(d)", 2: "(c)", 1: "(b)", 0: "(a)"}
for ax_i, label in panel_labels.items():
    axs[ax_i].text(
        0.02,
        0.98,
        label,
        transform=axs[ax_i].transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )

# Export stacked subplot figure
fig.savefig(
    "Fig_1_model_operation.png",
    dpi=300,
    bbox_inches="tight",
)



# %%
# plot Fig 2: c_o and c_c r_ave distributions at time points

def _as_x_time(arr: np.ndarray, x_points: int, t_points: int) -> np.ndarray:
    """Return array shaped (x, time) for plotting profiles vs x at chosen times."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")

    # Prefer the user-specified interpretation: first dimension corresponds to x (e.g., x_p = 20)
    if arr.shape[0] == x_points:
        return arr
    if arr.shape[1] == x_points:
        return arr.T

    # Fallback: if one dimension matches time length, treat the other as x
    if arr.shape[0] == t_points:
        return arr.T
    if arr.shape[1] == t_points:
        return arr

    raise ValueError(
        f"Unable to interpret array with shape {arr.shape} as (x,time) with x={x_points}, time={t_points}"
    )


# --- Plot 1: R-averaged oxygen concentration profiles (t1-t6) ---
x_points = int(var_pts.get("x_p", 20))
t_points = len(time_vector_3)

c_o_x_t = _as_x_time(np.asarray(c_o_r_avg), x_points=x_points, t_points=t_points)
x_axis = np.linspace(0.0, 1.0, c_o_x_t.shape[0])

fig_oxygen, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
for k, idx in enumerate(marker_idxs, start=1):
    (line,) = ax.plot(
        x_axis,
        c_o_x_t[:, idx],
    )
    y_end = float(c_o_x_t[-1, idx])
    ax.annotate(
        f"t{k}",
        xy=(x_axis[-1], y_end),
        xytext=(4, 4),
        textcoords="offset points",
        color=line.get_color(),
        ha="left",
        va="bottom",
    )
ax.set_xlabel("Normalised Shell Radius [-]")
ax.set_ylabel("Averaged Shell \nOxygen Concentration [mol m$^{-3}$]")
ax.grid(True, alpha=0.3)
fig_oxygen.savefig("Fig2b_radial_oxygen.png", dpi=300, bbox_inches="tight")


# --- Plot 2: R-averaged core lithium concentration profiles (t1-t6) ---
c_c_x_t = _as_x_time(np.asarray(c_c_r_avg), x_points=x_points, t_points=t_points)
x_axis = np.linspace(0.0, 1.0, c_c_x_t.shape[0])

fig_lithium, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
for k, idx in enumerate(marker_idxs, start=1):
    (line,) = ax.plot(
        x_axis,
        c_c_x_t[:, idx],
    )
    y_end = float(c_c_x_t[-1, idx])
    ax.annotate(
        f"t{k}",
        xy=(x_axis[-1], y_end),
        xytext=(4, 4),
        textcoords="offset points",
        color=line.get_color(),
        ha="left",
        va="bottom",
    )
ax.set_xlabel("Normalised Core Radius [-]")
ax.set_ylabel("Averaged Core \nLithium Concentration [mol m$^{-3}$]")
ax.grid(True, alpha=0.3)

# RS formation threshold
ax.axhline(
    RS_conc_thresh,
    color="red",
    linestyle="--",
    linewidth=1.0,
    alpha=0.8,
    zorder=0,
)
ax.annotate(
    "RS Formation Threshold",
    xy=(x_axis[1] if len(x_axis) > 1 else x_axis[0], RS_conc_thresh),
    xytext=(0, 6),
    textcoords="offset points",
    color="red",
    ha="left",
    va="bottom",
)
fig_lithium.savefig("Fig2a_radial_lithium.png", dpi=300, bbox_inches="tight")

# %%

# --- c_o_shell heatmaps (r_sh vs x_p) at sampled time points ---
idx_sampled = marker_idxs
import matplotlib.colors as mcolors

tN = len(time_vector_3)
c_shell = np.asarray(c_o_shell)

# Try to recover spatial coordinates for (x_p, r_sh) if available
x_pts = None
r_pts = None
try:
    shell_var = solution["Positive shell oxygen concentration [mol.m-3]"]
    spatial_pts = getattr(shell_var, "spatial_pts", None)
    if isinstance(spatial_pts, dict):
        for k, v in spatial_pts.items():
            k_lower = str(k).lower()
            if ("r" in k_lower) and ("sh" in k_lower):
                r_pts = np.asarray(v)
            elif "x" in k_lower:
                x_pts = np.asarray(v)
    elif isinstance(spatial_pts, (list, tuple)) and len(spatial_pts) == 2:
        a0 = np.asarray(spatial_pts[0])
        a1 = np.asarray(spatial_pts[1])
        # Heuristic: x_p typically has more points than r_sh
        if a0.size >= a1.size:
            x_pts, r_pts = a0, a1
        else:
            x_pts, r_pts = a1, a0
except Exception:
    pass

# Bring c_shell into shape (time, r, x)
def _as_time_r_x(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        # Move time axis to front
        if arr.shape[0] == tN:
            out = arr
        elif arr.shape[1] == tN:
            out = np.moveaxis(arr, 1, 0)
        elif arr.shape[2] == tN:
            out = np.moveaxis(arr, 2, 0)
        else:
            raise ValueError(
                f"c_o_shell has shape {arr.shape}, but no axis matches time length {tN}"
            )

        # Ensure order is (time, r, x)
        if r_pts is not None and x_pts is not None:
            if out.shape[1] == x_pts.size and out.shape[2] == r_pts.size:
                out = np.transpose(out, (0, 2, 1))
        return out

    if arr.ndim == 2:
        # Flattened spatial dimension
        if arr.shape[0] == tN:
            flat = arr
        elif arr.shape[1] == tN:
            flat = arr.T
        else:
            raise ValueError(
                f"c_o_shell has shape {arr.shape}, but neither axis matches time length {tN}"
            )

        n_spatial = flat.shape[1]
        if r_pts is not None and x_pts is not None and (r_pts.size * x_pts.size == n_spatial):
            return flat.reshape(tN, r_pts.size, x_pts.size)

        # Fallback: use var_pts if it matches
        nr_guess = int(var_pts.get("r_sh", 0) or 0)
        nx_guess = int(var_pts.get("x_p", 0) or 0)
        if nr_guess > 0 and nx_guess > 0 and (nr_guess * nx_guess == n_spatial):
            return flat.reshape(tN, nr_guess, nx_guess)

        raise ValueError(
            "Unable to reshape c_o_shell into (time, r_sh, x_p). "
            f"Got shape {arr.shape}; spatial size={n_spatial}."
        )

    raise ValueError(f"Expected c_o_shell to be 2D or 3D, got shape {arr.shape}")

c_time_r_x = _as_time_r_x(c_shell)

if x_pts is None:
    x_pts = np.linspace(0.0, 1.0, c_time_r_x.shape[2])
if r_pts is None:
    r_pts = np.linspace(0.0, 1.0, c_time_r_x.shape[1])

# Use a common color scale across all subplots
vmin = np.nanmin(c_time_r_x[idx_sampled, :, :])
vmax = np.nanmax(c_time_r_x[idx_sampled, :, :])
vcenter = np.nanmedian(c_time_r_x[idx_sampled, :, :])
if not (vmin < vcenter < vmax):
    vcenter = 0.5 * (vmin + vmax)

norm = None
if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

fig, axs = plt.subplots(
    int(np.ceil(len(idx_sampled) / 2)),
    2,
    figsize=(12, 14),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)
axs = np.atleast_1d(axs).ravel()

n_panels = len(idx_sampled)
axs_used = axs[:n_panels]
for ax in axs[n_panels:]:
    ax.axis("off")

extent = [float(np.min(x_pts)), float(np.max(x_pts)), 0.0, 1.0]

ims = []
for k, idx in enumerate(idx_sampled):
    ax = axs_used[k]
    data2d = c_time_r_x[idx, :, :]
    im = ax.imshow(
        data2d,
        aspect="auto",
        cmap="jet",
        norm=norm,
        extent=extent,
        origin="lower",
        interpolation="nearest",
    )
    ims.append(im)

    ax.set_title(f"t{k + 1}")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(0.0, 1.0)

    # Keep simple normalized ticks if coordinates are normalized
    ax.set_xticks(np.linspace(extent[0], extent[1], 5))
    ax.set_yticks(np.linspace(0.0, 1.0, 5))

# Axis labels
ncols = 2
nrows = int(np.ceil(n_panels / ncols))
for i, ax in enumerate(axs_used):
    row = i // ncols
    col = i % ncols
    if col == 0:
        ax.set_ylabel("Normalised Shell Radius [-]")
    if row == (nrows - 1):
        ax.set_xlabel("Normalised PE Thickness [-]")

cbar = fig.colorbar(ims[0], ax=axs_used, shrink=0.9, pad=0.02)
cbar.set_label("Shell oxygen concentration [mol m$^{-3}$]")

plt.show()
# %%
# Plot Fig 1: Model Operation with no points
# Stacked plots (top-to-bottom): Mass flux, LAM_pe, s_dot_ave, Voltage, current
RS_conc_thresh = param_half_cell["Threshold lithium concentration for phase transition [mol.m-3]"]
t_hr = time_vector_3 / 3600

fig, axs = plt.subplots(
    5,
    1,
    sharex=True,
    figsize=(10, 12),
    constrained_layout=True,
)

x = 3000
axs[0].plot(t_hr, X_ave_Oxygen_boundary_flux_3, label="Solvent Oxidation Flux")
axs[0].plot(t_hr, X_ave_Oxygen_boundary_flux_inner_3, label="RS Formation Flux")
axs[0].set_ylabel("Mass flux\n[mol m$^{-2}$ s$^{-1}$]")
axs[0].grid(True, alpha=0.3)
#axs[0].set_title("Mass flux")
axs[0].legend()


axs[1].plot(t_hr, c_o_ave_shell)
axs[1].set_ylabel("Average \nShell Oxygen \nConcentration\n[mol m$^{-3}$]")
axs[1].grid(True, alpha=0.3)

axs[2].plot(t_hr, c_s_surf_3)
axs[2].set_ylabel("Surface \nCore Lithium \nConcentration\n[mol m$^{-3}$]")
axs[2].axhline(
    RS_conc_thresh,
    color="red",
    linestyle="--",
    linewidth=1.0,
    alpha=0.6,
    zorder=0,
    label="RS Formation Threshold",
)
axs[2].legend()
axs[2].grid(True, alpha=0.3)


axs[3].plot(t_hr, V_cell_3)
axs[3].set_ylabel("Voltage\n[V]")
axs[3].grid(True, alpha=0.3)

axs[4].plot(t_hr, I_app_data_3)
axs[4].set_ylabel("Current\n[A]")
axs[4].set_xlabel("Time [h]")
axs[4].grid(True, alpha=0.3)

# %%
# --- GIF generation (plac
# ed at bottom of script, as requested) ---
# Oxygen concentration heat map at 10 equally spaced time points
from matplotlib.animation import FuncAnimation

gif_frames = 10


def _equally_spaced_time_indices_hr(t_hr_vec: np.ndarray, n: int) -> np.ndarray:
    t_hr_vec = np.asarray(t_hr_vec, dtype=float)
    if t_hr_vec.ndim != 1 or t_hr_vec.size < 2:
        raise ValueError("t_hr must be a 1D array with at least two entries")
    if n < 2:
        raise ValueError("gif_frames must be >= 2")

    targets = np.linspace(float(t_hr_vec[0]), float(t_hr_vec[-1]), n)
    idx = np.searchsorted(t_hr_vec, targets, side="left")
    idx = np.clip(idx, 0, t_hr_vec.size - 1).astype(int)

    for i in range(1, idx.size):
        if idx[i] <= idx[i - 1]:
            idx[i] = min(idx[i - 1] + 1, t_hr_vec.size - 1)

    if np.unique(idx).size != n:
        idx = np.linspace(0, t_hr_vec.size - 1, n, dtype=int)
    return idx


def _as_time_r_x_for_gif(
    arr: np.ndarray,
    tN: int,
    r_pts: np.ndarray | None,
    x_pts: np.ndarray | None,
    var_pts: dict,
) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[0] == tN:
            out = arr
        elif arr.shape[1] == tN:
            out = np.moveaxis(arr, 1, 0)
        elif arr.shape[2] == tN:
            out = np.moveaxis(arr, 2, 0)
        else:
            raise ValueError(
                f"c_o_shell has shape {arr.shape}, but no axis matches time length {tN}"
            )

        if r_pts is not None and x_pts is not None:
            if out.shape[1] == x_pts.size and out.shape[2] == r_pts.size:
                out = np.transpose(out, (0, 2, 1))
        return out

    if arr.ndim == 2:
        if arr.shape[0] == tN:
            flat = arr
        elif arr.shape[1] == tN:
            flat = arr.T
        else:
            raise ValueError(
                f"c_o_shell has shape {arr.shape}, but neither axis matches time length {tN}"
            )

        n_spatial = flat.shape[1]
        if r_pts is not None and x_pts is not None and (r_pts.size * x_pts.size == n_spatial):
            return flat.reshape(tN, r_pts.size, x_pts.size)

        nr_guess = int(var_pts.get("r_sh", 0) or 0)
        nx_guess = int(var_pts.get("x_p", 0) or 0)
        if nr_guess > 0 and nx_guess > 0 and (nr_guess * nx_guess == n_spatial):
            return flat.reshape(tN, nr_guess, nx_guess)

        raise ValueError(
            "Unable to reshape c_o_shell into (time, r_sh, x_p). "
            f"Got shape {arr.shape}; spatial size={n_spatial}."
        )

    raise ValueError(f"Expected c_o_shell to be 2D or 3D, got shape {arr.shape}")


try:
    # Pull the raw oxygen field again (keeps the GIF block self-contained)
    t_hr_all = np.asarray(solution.t, dtype=float) / 3600.0
    tN_all = int(t_hr_all.size)
    c_shell_raw = np.asarray(solution["Positive shell oxygen concentration [mol.m-3]"].data)

    # Try to recover spatial coordinates for (x_p, r_sh)
    x_pts_gif = None
    r_pts_gif = None
    try:
        shell_var = solution["Positive shell oxygen concentration [mol.m-3]"]
        spatial_pts = getattr(shell_var, "spatial_pts", None)
        if isinstance(spatial_pts, dict):
            for k, v in spatial_pts.items():
                k_lower = str(k).lower()
                if ("r" in k_lower) and ("sh" in k_lower):
                    r_pts_gif = np.asarray(v)
                elif "x" in k_lower:
                    x_pts_gif = np.asarray(v)
        elif isinstance(spatial_pts, (list, tuple)) and len(spatial_pts) == 2:
            a0 = np.asarray(spatial_pts[0])
            a1 = np.asarray(spatial_pts[1])
            if a0.size >= a1.size:
                x_pts_gif, r_pts_gif = a0, a1
            else:
                x_pts_gif, r_pts_gif = a1, a0
    except Exception:
        pass

    c_time_r_x_gif = _as_time_r_x_for_gif(
        c_shell_raw, tN=tN_all, r_pts=r_pts_gif, x_pts=x_pts_gif, var_pts=var_pts
    )

    if x_pts_gif is None:
        x_pts_gif = np.linspace(0.0, 1.0, c_time_r_x_gif.shape[2])
    if r_pts_gif is None:
        r_pts_gif = np.linspace(0.0, 1.0, c_time_r_x_gif.shape[1])

    idx_gif = _equally_spaced_time_indices_hr(t_hr_all, gif_frames)

    vmin_gif = np.nanmin(c_time_r_x_gif[idx_gif, :, :])
    vmax_gif = np.nanmax(c_time_r_x_gif[idx_gif, :, :])
    if not (np.isfinite(vmin_gif) and np.isfinite(vmax_gif)):
        raise ValueError("Non-finite values encountered in oxygen concentration data")

    extent_gif = [float(np.min(x_pts_gif)), float(np.max(x_pts_gif)), 0.0, 1.0]
    fig_gif, ax_gif = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    im_gif = ax_gif.imshow(
        c_time_r_x_gif[idx_gif[0], :, :],
        aspect="auto",
        cmap="jet",
        vmin=vmin_gif,
        vmax=vmax_gif,
        extent=extent_gif,
        origin="lower",
        interpolation="nearest",
    )

    ax_gif.set_xlabel("Normalised PE Thickness [-]")
    ax_gif.set_ylabel("Normalised Shell Radius [-]")
    title_txt = ax_gif.set_title(f"t = {float(t_hr_all[idx_gif[0]]):.3f} hr")

    cbar_gif = fig_gif.colorbar(im_gif, ax=ax_gif, pad=0.02)
    cbar_gif.set_label("Shell oxygen concentration [mol m$^{-3}$]")

    def _update_gif(frame_k: int):
        idx = int(idx_gif[frame_k])
        im_gif.set_data(c_time_r_x_gif[idx, :, :])
        title_txt.set_text(f"t = {float(t_hr_all[idx]):.3f} hr")
        return (im_gif, title_txt)

    ani = FuncAnimation(
        fig_gif,
        _update_gif,
        frames=len(idx_gif),
        interval=700,
        blit=False,
        repeat=True,
    )

    gif_name = "c_o_shell_heatmap_10frames.gif"
    try:
        ani.save(gif_name, writer="pillow", dpi=150)
        print(f"Saved GIF: {gif_name}")
    except Exception as e:
        print(
            "Failed to save GIF with Pillow. If needed, install pillow (pip install pillow).\n"
            f"Error: {e}"
        )

    plt.show()
except Exception as e:
    print(f"GIF generation skipped due to error: {e}")



# --- GIF: stacked plot with dashed time marker (10 frames) ---
# Uses the same 10 equally spaced time points as the concentration heatmap GIF.
try:
    t_hr_stack = np.asarray(solution.t, dtype=float) / 3600.0
    idx_gif_stack = _equally_spaced_time_indices_hr(t_hr_stack, gif_frames)

    # Reuse the stacked-plot figure created above ("Model Operation with no points")
    x0 = float(t_hr_stack[idx_gif_stack[0]])
    vlines = [
        ax.axvline(
            x0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            zorder=10,
        )
        for ax in np.atleast_1d(axs).ravel()
    ]

    def _update_stacked(frame_k: int):
        x = float(t_hr_stack[int(idx_gif_stack[frame_k])])
        for ln in vlines:
            ln.set_xdata([x, x])
        return tuple(vlines)

    ani_stacked = FuncAnimation(
        fig,
        _update_stacked,
        frames=len(idx_gif_stack),
        interval=700,
        blit=False,
        repeat=True,
    )

    gif_name_stacked = "Fig1_model_operation_stacked_10frames.gif"
    try:
        ani_stacked.save(gif_name_stacked, writer="pillow", dpi=150)
        print(f"Saved GIF: {gif_name_stacked}")
    except Exception as e:
        print(
            "Failed to save stacked-plot GIF with Pillow. If needed, install pillow (pip install pillow).\n"
            f"Error: {e}"
        )
except Exception as e:
    print(f"Stacked-plot GIF generation skipped due to error: {e}")

# %%
