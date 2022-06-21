# Evolution of the fit metric between the optimal and guess models.

# Library imports.
import numpy as np
import matplotlib.pyplot as plt

# rasopt imports.
from rasopt.Utilities.utils import inun_fit, extract_depths, inun_error, inun_sensitivity

# Load in the depth data.
# gt_plan_fp = r"C:\Users\ay434\Documents\10_min_Runs\Secchia_flood_2014_10min_Orig_GT_LC_Cluster\Secchia_Panaro.p23.hdf"
# cal_plan_fp = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Mannings_Sensitivity\Secchia_Panaro.p23_camp0.0575.hdf"
# guess_plan_fp = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Clustered_GT\Secchia_Panaro.p23_camp0.07.hdf"

gt_plan_fp =r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\CDF_Single_Veg\Secchia_Panaro.p23_Orig_GT.hdf"
cal_plan_fp = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\CDF_Single_Veg\Secchia_Panaro.p23_camp0.0588.hdf"
guess_plan_fp = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Clustered_GT\Secchia_Panaro.p23_camp0.07.hdf"


# Time step duration (hrs).
dt = 1/6

# Labels.
n_cal = 0.0588
n_guess = 0.07

# HDF Paths.
# Path to cell coordinates.
cell_coord_path = '/Geometry/2D Flow Areas/Secchia_Panaro/Cells Center Coordinate'

# Path to water depths.
depth_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Secchia_Panaro/Depth'

# Depths from all models.
gt_depths = extract_depths(gt_plan_fp, depth_path, cell_coord_path)
cal_depths = extract_depths(cal_plan_fp, depth_path, cell_coord_path)
guess_depths = extract_depths(guess_plan_fp, depth_path, cell_coord_path)

# Number of time steps.
Nt = gt_depths.shape[1] - 2

# Fit metrics.
cal_fit = []
guess_fit = []
cal_error1 = []
guess_error1 = []
cal_error2 = []
guess_error2 = []
cal_sens = []
guess_sens = []
for t in range(Nt):
    ts_name = f'Time_{t}'

    # Depths for a particular time step.
    gt_ts = gt_depths.loc[:, ts_name]
    cal_ts = cal_depths.loc[:, ts_name]
    guess_ts = guess_depths.loc[:, ts_name]

    # Fit metric.
    cal_fit.append(inun_fit(gt_ts, cal_ts, depth_cut=0.01))
    guess_fit.append(inun_fit(gt_ts, guess_ts, depth_cut=0.01))
    cal_error1.append(inun_error(gt_ts, cal_ts, 1, depth_cut=0.01))
    guess_error1.append(inun_error(gt_ts, guess_ts, 1, depth_cut=0.01))
    cal_error2.append(inun_error(gt_ts, cal_ts, 2, depth_cut=0.01))
    guess_error2.append(inun_error(gt_ts, guess_ts, 2, depth_cut=0.01))
    cal_sens.append(inun_sensitivity(gt_ts, cal_ts, depth_cut=0.01))
    guess_sens.append(inun_sensitivity(gt_ts, guess_ts, depth_cut=0.01))

# Plot the fit metrics over the duration of the flood.
time_ax = np.arange(Nt) * dt

fig, ax = plt.subplots(2,2,figsize=(10,8), sharex='col')
ax[0,0].plot(time_ax, cal_sens, label='$n^*=0.0588$')
ax[0,0].plot(time_ax, guess_sens, label='$n_p=0.07$')
# ax[0,0].set_xlabel('Time After Beach (hr)', fontsize=16)
ax[0,0].set_ylabel('Sensitivity', fontsize=16)
ax[0,0].tick_params(axis='x', labelsize=14)
ax[0,0].tick_params(axis='y', labelsize=14)
ax[0,0].legend(fontsize=14)

ax[1,0].plot(time_ax, cal_error1, label='$\hat{n}=0.0588$')
ax[1,0].plot(time_ax, guess_error1, label='$n=0.07$')
ax[1,0].set_xlabel('Time After Beach (hr)', fontsize=16)
ax[1,0].set_ylabel('Type I Error', fontsize=16)
ax[1,0].tick_params(axis='x', labelsize=14)
ax[1,0].tick_params(axis='y', labelsize=14)
# ax[1,0].legend(fontsize=14)

ax[0,1].plot(time_ax, cal_fit, label='$\hat{n}=0.0588$')
ax[0,1].plot(time_ax, guess_fit, label='$n=0.07$')
# ax[0,1].set_xlabel('Time After Beach (hr)', fontsize=16)
ax[0,1].set_ylabel('Fit', fontsize=16)
ax[0,1].tick_params(axis='x', labelsize=14)
ax[0,1].tick_params(axis='y', labelsize=14)
# ax[0,1].legend(fontsize=14)

ax[1,1].plot(time_ax, cal_error2, label='$\hat{n}=0.0588$')
ax[1,1].plot(time_ax, guess_error2, label='$n=0.07$')
ax[1,1].set_xlabel('Time After Beach (hr)', fontsize=16)
ax[1,1].set_ylabel('Type II Error', fontsize=16)
ax[1,1].tick_params(axis='x', labelsize=14)
ax[1,1].tick_params(axis='y', labelsize=14)
# ax[1,1].legend(fontsize=14)
fig.tight_layout()
fig.savefig(r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Figures\Case_Study_1\inun_metrics.png")

plt.show()


