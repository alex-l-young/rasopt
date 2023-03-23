# Evolution of the fit metric between the optimal and guess models.

# Library imports.
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# rasopt imports.
from rasopt.Utilities.utils import inun_fit, extract_depths, inun_error, inun_sensitivity

# Load in the depth data.
# gt_plan_fp = r"C:\Users\ay434\Documents\10_min_Runs\Secchia_flood_2014_10min_Orig_GT_LC_Cluster\Secchia_Panaro.p23.hdf"
# cal_plan_fp = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Mannings_Sensitivity\Secchia_Panaro.p23_camp0.0575.hdf"
# guess_plan_fp = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Clustered_GT\Secchia_Panaro.p23_camp0.07.hdf"

file_dir = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Roughness_Output"
gt_fname ="Secchia_Panaro.p23_GT.hdf"
sim_fnames = [
    "Secchia_Panaro.p23_c1s5.hdf",
    "Secchia_Panaro.p23_c2s5.hdf",
    "Secchia_Panaro.p23_c3s5.hdf",
    "Secchia_Panaro.p23_c4s5.hdf",
    "Secchia_Panaro.p23_c5s5.hdf",
]

save_label = 's5'
# labels = [
#     "$N_{class}=1, N_{sensor}=Max$",
#     "$N_{class}=2, N_{sensor}=Max$",
#     "$N_{class}=3, N_{sensor}=Max$",
# "$N_{class}=4, N_{sensor}=Max$",
# "$N_{class}=5, N_{sensor}=Max$",
# ]

labels = []
for name in sim_fnames:
    n_class = re.search(r'c(\d)', name).group(1)
    n_sens = re.search(r's(.*)\.', name).group(1)
    labels.append(f"$N_{{class}}={n_class}, N_{{sensor}}={n_sens}$")

gt_fp = os.path.join(file_dir, gt_fname)
sim_fps = [os.path.join(file_dir, fname) for fname in sim_fnames]

# Time step duration (hrs).
dt = 1/6

# HDF Paths.
# Path to cell coordinates.
cell_coord_path = '/Geometry/2D Flow Areas/Secchia_Panaro/Cells Center Coordinate'

# Path to water depths.
depth_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Secchia_Panaro/Depth'

# Depths from all models.
gt_depths = extract_depths(gt_fp, depth_path, cell_coord_path)

# Number of time steps.
Nt = gt_depths.shape[1] - 2

# Statistics.
sens = np.zeros((Nt, len(sim_fps)))
t1 = np.zeros((Nt, len(sim_fps)))
t2 = np.zeros((Nt, len(sim_fps)))
fit = np.zeros((Nt, len(sim_fps)))

for i, sim_fp in enumerate(sim_fps):
    print('Processing', labels[i])
    # Simulation depths.
    sim_depths = extract_depths(sim_fp, depth_path, cell_coord_path)
    for t in range(Nt):
        ts_name = f'Time_{t}'

        # Depths for a particular time step.
        gt_ts = gt_depths.loc[:, ts_name]
        sim_ts = sim_depths.loc[:, ts_name]

        # Fit metric.
        sens[t, i] = inun_fit(gt_ts, sim_ts, depth_cut=0.01)
        t1[t, i] = inun_error(gt_ts, sim_ts, 1, depth_cut=0.01)
        t2[t, i] = inun_error(gt_ts, sim_ts, 2, depth_cut=0.01)
        fit[t, i] = inun_fit(gt_ts, sim_ts, depth_cut=0.01)

# Print the mean values over all time steps.
for i, label in enumerate(labels):
    print(f'{label} Sensitivity: {np.mean(sens[:,i])}')
    print(f'{label} Fit: {np.mean(fit[:, i])}')
    print(f'{label} Type I Error: {np.mean(t1[:, i])}')
    print(f'{label} Type II Error: {np.mean(t2[:, i])}')

# Plot the fit metrics over the duration of the flood.
time_ax = np.arange(Nt) * dt

fig, ax = plt.subplots(2,2,figsize=(10,8), sharex='col')
for i, sim_fp in enumerate(sim_fps):
    ax[0,0].plot(time_ax, sens[:, i], label=labels[i])

    ax[1,0].plot(time_ax, t1[:, i], label=labels[i])

    ax[0,1].plot(time_ax, fit[:, i], label=labels[i])

    ax[1,1].plot(time_ax, t2[:, i], label=labels[i])


# Format the axes.
ax[0,0].set_ylabel('Sensitivity', fontsize=16)
ax[0,0].tick_params(axis='x', labelsize=14)
ax[0,0].tick_params(axis='y', labelsize=14)
ax[0,0].legend(fontsize=14)

ax[1, 0].set_xlabel('Time After Beach (hr)', fontsize=16)
ax[1, 0].set_ylabel('Type I Error', fontsize=16)
ax[1, 0].tick_params(axis='x', labelsize=14)
ax[1, 0].tick_params(axis='y', labelsize=14)

ax[0, 1].set_ylabel('Fit', fontsize=16)
ax[0, 1].tick_params(axis='x', labelsize=14)
ax[0, 1].tick_params(axis='y', labelsize=14)

ax[1, 1].set_xlabel('Time After Beach (hr)', fontsize=16)
ax[1, 1].set_ylabel('Type II Error', fontsize=16)
ax[1, 1].tick_params(axis='x', labelsize=14)
ax[1, 1].tick_params(axis='y', labelsize=14)

fig.tight_layout()
fig.savefig(r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Figures\Roughness_Case_Study\inun_stats_{}.png".format(save_label))

plt.show()


