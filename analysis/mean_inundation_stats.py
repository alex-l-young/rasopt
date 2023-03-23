# Evolution of the fit metric between the optimal and guess models.

# Library imports.
import numpy as np
import pandas as pd
import os
import re

# rasopt imports.
from rasopt.Utilities.utils import inun_fit, extract_depths, inun_error, inun_sensitivity

# Load in the depth data.
# gt_plan_fp = r"C:\Users\ay434\Documents\10_min_Runs\Secchia_flood_2014_10min_Orig_GT_LC_Cluster\Secchia_Panaro.p23.hdf"
# cal_plan_fp = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Mannings_Sensitivity\Secchia_Panaro.p23_camp0.0575.hdf"
# guess_plan_fp = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Clustered_GT\Secchia_Panaro.p23_camp0.07.hdf"

gt_file_dir = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Roughness_Output"
gt_fname ="Secchia_Panaro.p23_GT.hdf"
sim_file_dir = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Mannings_Sensitivity"
sim_fnames = [
    "Secchia_Panaro.p23_veg1_0.03.hdf",
    "Secchia_Panaro.p23_veg1_0.05.hdf",
    "Secchia_Panaro.p23_veg1_0.07.hdf",
    "Secchia_Panaro.p23_veg1_0.09.hdf",
    "Secchia_Panaro.p23_veg1_0.11.hdf",
]

save_id = 'empirical'

labels = []
for name in sim_fnames:
    try:
        n_class = re.search(r'c(\d)', name).group(1)
        n_sens = re.search(r's(.*)\.', name).group(1)
        labels.append(f"$N_{{class}}={n_class}, N_{{sensor}}={n_sens}$")
    except:
        labels.append(f"n={os.path.splitext(name)[0].split('_')[-1]}")

gt_fp = os.path.join(gt_file_dir, gt_fname)
sim_fps = [os.path.join(sim_file_dir, fname) for fname in sim_fnames]

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
data = {'Class': [],
        'Sensitivity': [],
        'Fit': [],
        'Type_I': [],
        'Type_II': []}
for i, label in enumerate(labels):
    try:
        n_class = re.search(r'c(\d)', sim_fnames[i]).group(1)
    except:
        n_class = eval(label.split("=")[-1])
    mean_sens = np.mean(sens[:,i], where=~np.isnan(sens[:,i]))
    mean_fit = np.mean(fit[:, i], where=~np.isnan(fit[:,i]))
    mean_t1 = np.mean(t1[:, i], where=~np.isnan(t1[:,i]))
    mean_t2 = np.mean(t2[:, i], where=~np.isnan(t2[:,i]))
    print(f'{label} Sensitivity: {mean_sens}')
    print(f'{label} Fit: {mean_fit}')
    print(f'{label} Type I Error: {mean_t1}')
    print(f'{label} Type II Error: {mean_t2}')

    data['Class'].append(n_class)
    data['Sensitivity'].append(mean_sens)
    data['Fit'].append(mean_fit)
    data['Type_I'].append(mean_t1)
    data['Type_II'].append(mean_t2)

df = pd.DataFrame(data)
# df.to_csv(os.path.join(sim_file_dir, f'inun_stats_{save_id}.csv'), index=False)

