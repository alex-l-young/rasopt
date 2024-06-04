# Compare the skill of the model forecast using the depth data at the resolution of the DEM.

# Library imports.
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import os
import rasterio
import geojson

# Local imports.
from rasopt.Utilities import utils, land_cover_utils

# Load in the files.
# Directory to files.
file_dir = r"C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Roughness_Output"

# Ground truth.
gt_fname = "Secchia_Panaro.p23_GT.hdf"
gt_fp = os.path.join(file_dir, gt_fname)

# Calibrated model.
cal_fname = "Secchia_Panaro.p23_c4s6.hdf"
cal_fp = os.path.join(file_dir, cal_fname)

# HDF paths.
# Cell coordinate path.
cell_coord_path = 'Geometry/2D Flow Areas/Secchia_Panaro/Cells Center Coordinate'

# Water depths path.
depth_path = ('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/'
              '2D Flow Areas/Secchia_Panaro/Depth')

# Manning's n calibration table path.
cal_table_path = "Geometry/Land Cover (Manning's n)/Calibration Table"

# Cell facepoint index path.
cell_facepoint_idx_path = 'Geometry/2D Flow Areas/Secchia_Panaro/Cells FacePoint Indexes'

# Facepoint coordinate path.
facepoint_coord_path = 'Geometry/2D Flow Areas/Secchia_Panaro/FacePoints Coordinate'

# Extract ground truth depths.
gt_depths = utils.extract_depths(gt_fp, depth_path, cell_coord_path)

# Extract calibrated depths.
cal_depths = utils.extract_depths(cal_fp, depth_path, cell_coord_path)

# Number of cells and time steps.
ncell, ncol = gt_depths.shape
Nt = ncol - 2

# Count the number of timesteps where the inundation prediction and truth did not match.
inun_cut = 0.1
gt_and_notcal = np.zeros((1, ncell))
notgt_and_cal = np.zeros((1, ncell))
gt_and_cal = np.zeros((1, ncell))
for i in range(2, ncol):
    # Depths from a single time step.
    Dgt = gt_depths.iloc[:, i].to_numpy()
    Dcal = cal_depths.iloc[:, i].to_numpy()

    # Set depths above inun_cut to 1 and below inun_cut to 0.
    Dgt[Dgt >= inun_cut] = 1
    Dgt[Dgt < inun_cut] = 0
    gt_bool = np.array(Dgt, dtype=bool)
    Dcal[Dcal >= inun_cut] = 1
    Dcal[Dcal < inun_cut] = 0
    cal_bool = np.array(Dcal, dtype=bool)

    # Compute logical values.
    gt_and_cal += np.logical_and(gt_bool, cal_bool).astype(float)
    gt_and_notcal += np.logical_and(gt_bool, ~cal_bool).astype(float)
    notgt_and_cal += np.logical_and(~gt_bool, cal_bool).astype(float)


# Create depth geojson for the logical arrays.
gt_and_cal_gjson = utils.depth_geojson(gt_fp, cell_facepoint_idx_path, facepoint_coord_path,
                                     depth_path, cell_coord_path, depths=gt_and_cal)

# Save geojson.
gt_and_cal_savefp = r"C:\Users\ay434\Desktop\gt_and_cal.geojson"
with open(gt_and_cal_savefp, 'w') as dst:
    geojson.dump(gt_and_cal_gjson, dst)