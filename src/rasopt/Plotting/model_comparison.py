# %%

# Compare the skill of the model forecast using the depth data at the resolution of the DEM.

# Library imports.
import matplotlib.pyplot as plt
import numpy as np
import os

# Local imports.
from rasopt.Utilities import utils, land_cover_utils

# %%

# Plan files to compare.
gt_plan_fp = r"Data\Secchia_Panaro.p23_Orig.hdf"
sim_best_plan_fp = r"Data\Secchia_Panaro.p23_15min_Best.hdf"

# Path to the DEM file.
dem_fname = r"DTM_1m.tif"
dem_fp = os.path.join(os.path.dirname(gt_plan_fp), dem_fname)

# Comparison timestep.
timestep = 120
dt = 0.25  # Hours.

# Downsample factor.
resample = 0.05

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

# %%

# Depth rasters at the resolution of the DEM raster.
gt_depths, gt_depth_path = land_cover_utils.terrain_water_depth(dem_fp, gt_plan_fp, cell_facepoint_idx_path,
                                                                facepoint_coord_path,
                                                                depth_path, cell_coord_path, timestep,
                                                                out_fname_prefix='GT', nodata=-999.0,
                                                                cleanup_rasters=False, resample=resample)
sim_depths, sim_depth_path = land_cover_utils.terrain_water_depth(dem_fp, sim_best_plan_fp, cell_facepoint_idx_path,
                                                                  facepoint_coord_path,
                                                                  depth_path, cell_coord_path, timestep,
                                                                  out_fname_prefix='', nodata=-999.0,
                                                                  cleanup_rasters=False, resample=resample)

# Compute the sensitivity.
sensitivity = utils.inun_sensitivity(gt_depths, sim_depths, depth_cut=0.01)

# Type I and II Error.
typeI = utils.inun_error(gt_depths, sim_depths, 1, depth_cut=0.01)
typeII = utils.inun_error(gt_depths, sim_depths, 2, depth_cut=0.01)

# Print results.
print('Sensitivity', sensitivity)
print('Type I', typeI)
print('Type II', typeII)

# %%

# Get the total number of timesteps.
depths = utils.extract_depths(gt_plan_fp, depth_path, cell_coord_path)
Nt = depths.shape[1] - 2
print(Nt)

# %%

# Critical depth comparison.
depth_cut = 0.1

# Time to critical depth arrays.
gt_tc = np.zeros_like(gt_depths)
sim_tc = np.zeros_like(gt_depths)

timesteps = list(range(Nt))
for t in range(300, 305):
    print('Processing Timestep:', t)
    timestep = t
    # Get the depths for that timestep.
    gt_depths, gt_depth_path = land_cover_utils.terrain_water_depth(dem_fp, gt_plan_fp, cell_facepoint_idx_path,
                                                                    facepoint_coord_path,
                                                                    depth_path, cell_coord_path, timestep,
                                                                    out_fname_prefix='GT', nodata=-999.0,
                                                                    cleanup_rasters=True,
                                                                    resample=resample)
    sim_depths, sim_depth_path = land_cover_utils.terrain_water_depth(dem_fp, sim_best_plan_fp, cell_facepoint_idx_path,
                                                                      facepoint_coord_path,
                                                                      depth_path, cell_coord_path, timestep,
                                                                      out_fname_prefix='', nodata=-999.0,
                                                                      cleanup_rasters=True,
                                                                      resample=resample)
    # Cells that exceed cutoff depth.
    gt_exceed = gt_depths > depth_cut
    sim_exceed = sim_depths > depth_cut

    # Set gt_tc or sim_tc cells to the timestep when the first surpass the depth cut.
    gt_tc_mask = gt_tc > 0  # Already assigned cells.
    sim_tc_mask = sim_tc > 0  # Already assigned cells.
    gt_tc[gt_exceed & ~gt_tc_mask] = t
    sim_tc[sim_exceed & ~sim_tc_mask] = t

    print('Completed Timstep:', t, '\n')

# %%

# Difference in time to critical depth.
delta_tc = (gt_tc - sim_tc) * dt
print(np.max(delta_tc[delta_tc > 0]))

print(np.sum(delta_tc))

fig, ax = plt.subplots()
ax.hist(delta_tc[delta_tc != 0])

delta_tc_plot = delta_tc.copy()
delta_tc_plot[delta_tc_plot == 0] = np.nan
fig, ax = plt.subplots(figsize=(20, 20))
im = ax.imshow(delta_tc_plot)
plt.colorbar(im)

# %%


# %%


