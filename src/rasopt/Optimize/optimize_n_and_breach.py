# ====================================================================================================
# Single Bayesian Optimization of HEC-RAS Parameters.
# Alex Young
# June 2021
# ====================================================================================================

# Library imports.
from skopt.space import Real
from ax import RangeParameter, ParameterType
import os, sys
import pandas as pd

# Local imports.
from rasopt.Optimize import optimize1D2D

## Configuration file structure for Bayesian Optimization of the HEC-RAS model.

### 2D MODEL CONFIGURATIONS
### ===============================
# <<< Directories and file names. >>>
# ------------------------------------

# Ground truth directory.
gt_ras_path_2d = r"C:\Users\ay434\Documents\10_min_Runs\Secchia_flood_2014_10min_Orig_GT"

# Simulation directory.
sim_ras_path_2d = r"C:\Users\ay434\Documents\10_min_Runs\Secchia_flood_2014_10min_Sim"

# Results output directory.
res_output_dir_2d = "Output_Files"

# Project file name.
prj_fname_2d = "Secchia_Panaro.prj"

# Boundary condition file name.
bdy_fname_2d = "Secchia_Panaro.b23"

# Unstead flow file name.
flow_fname_2d = "Secchia_Panaro.u01"

# Plan HDF file name.
plan_hdf_fname_2d = "Secchia_Panaro.p23.hdf"

# Plan file name.
plan_fname_2d = "Secchia_Panaro.p23"

# Geometry HDF file name.
geom_fname_2d = "Secchia_Panaro.g23.hdf"

# DEM File path.
dem_fname = dem_fp = r"DTM_1m.tif"

# <<< HDF Paths >>>
# ------------------

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

# <<< Model Configurations >>>
# -----------------------------

# Comparison Type "Binary" or "Sensor".
comparison_type = 'Sensor'

# SENSOR CALIBRATION
# Number of hours after start when sensors are placed.
start_ts = 1

# BINARY CALIBRATION

# Time steps at which to do binary comparison.
binary_timesteps = [120]

# Binary comparison depth cutoff.
depth_cutoff = .1 # meters.

# Raster cell dimensions in units of the coordinate system.
cell_width_X = 100
cell_width_Y = 100

# Uncertainty spatial correlation radius.
radius = 10

# Depth error. The range of possible depth errors for a spurious cell in the uncertainty raster.
depth_error = 0.1

# No data value.
nodata = -999.0

# Number of uncertainty seeds.
n_seeds = 5

# Uncertainty type. "Binary" or "Depth".
uncertainty_type = 'Binary'

# Maximum error probability for flipping cells to a spurious value.
max_probability = 1

# Number of hours to run model.
n_ts = 10

# Mapping interval in minutes.
map_interval = 10

# <<< Optimization Configurations >>>
# -----------------------------------

# Number of initializations.
nstarts = 3

# Number of evaluations.
nevals = 20

# Parameter names that will be optimized. Must match the names in the model.
param_names_2d = ['campagna']

# Parameter bounds. put "_g" after land roughness parameters.
range_param_1 = RangeParameter(name='campagna_n_g', parameter_type=ParameterType.FLOAT, lower=.001, upper=.3)
# range_param_2 = RangeParameter(name='centri_n_g', parameter_type=ParameterType.FLOAT, lower=.001, upper=.5)
param_bounds_2d = [range_param_1]

# Loss function. ['MSE', 'RMSE', 'NNSE']
loss_func = 'RMSE'

# Sensor locations.
# locs = [(4952299.22, 655041.01), (4950859.92, 655273.81), (4952264.88, 655921.65), (4951767.55, 655117.21)]
        # (4953264.49, 657720.27)]

loc_df = pd.read_csv('sensor_locations.csv', names=['Latitude', 'Longitude'])
locs = list(zip(loc_df.Latitude, loc_df.Longitude))


### ===============================

### 1D MODEL CONFIGURATIONS
### ===============================
run_1D = False

# <<< Directories and file names. >>>
# -----------------------------------

# Simulation directory.
sim_ras_path_1d = r"C:\Users\ay434\Documents\Secchia_River_1D_2014_Sim"

# Results output directory.
res_output_dir_1d = "Output_Files"

# Project file name.
prj_fname_1d = "levee_breach_14.prj"

# Boundary condition file name.
bdy_fname_1d = "levee_breach_14.b01"

# Unstead flow file name.
flow_fname_1d = "levee_breach_14.u01"

# Plan HDF file name.
plan_hdf_fname_1d = "levee_breach_14.p01.hdf"

# Plan file name.
plan_fname_1d = "levee_breach_14.p01"

# Geometry HDF file name.
geom_fname_1d = "levee_breach_14.g01.hdf"

# <<< HDF Paths >>>
# -----------------

# Path to breach hydrograph.
hydrograph_path = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Lateral Structures/Secchia cassa_confine_er 41200.00/Structure Variables"

# <<< Model Configurations >>>
# ----------------------------

# Start Datetime. Formatted as "21DEC2013,2400"
start_datetime_1d = "21DEC2013,2400"

# End Datetime. Formatted as "21DEC2013,2400"
end_datetime_1d = "03FEB2014,2350"

# Output interval for breach hydrograph in minutes.
output_interval = 10

# Breach start.
breach_start = "19JAN2014 0240"

# <<< Optimization Configurations >>>
# -----------------------------------

# # Parameter names. ['weir_coefficient', 'bottom_width', 'formation_time']
# param_names_1d = ['weir_coefficient', 'bottom_width', 'formation_time']
#
# # Parameter bounds. Put "_b" at the end of the name to specify breach parameter.
# param_bounds_1d = [Real(1, 4, name='weir_coefficient_b'), Real(30, 100, name='bottom_width_b'), Real(2, 20, name='formation_time_b')]

# Parameter names. ['weir_coefficient', 'bottom_width', 'formation_time']
param_names_1d = []

# Parameter bounds. Put "_b" at the end of the name to specify breach parameter.
param_bounds_1d = []

# =======================================================================================================================================

# =======================================================================================================================================
# Configure the 1D and 2D models.
configure_class = optimize1D2D.HecRasModel(sim_ras_path_2d, bdy_fname_2d, plan_fname_2d, n_ts, map_interval,
                 sim_ras_path_1d, bdy_fname_1d, plan_fname_1d, start_datetime_1d, end_datetime_1d, output_interval)
configure_class.model_1d_static_setup()
configure_class.model_2d_static_setup()

if comparison_type == 'Binary':
    configure_class.satellite_comparison_setup(binary_timesteps, gt_ras_path_2d, dem_fname, cell_facepoint_idx_path, facepoint_coord_path,
                                       depth_path, cell_coord_path, cell_width_X, cell_width_Y, nodata, n_seeds, radius,
                                       uncertainty_type, depth_error=depth_error, max_probability=max_probability)
    gt_sat_rasters = configure_class.gt_sat_rasters
else:
    gt_sat_rasters = None

# Run the optimization.
optim = optimize1D2D.RASOpt(gt_ras_path_2d, sim_ras_path_2d, res_output_dir_2d, prj_fname_2d, bdy_fname_2d, flow_fname_2d,
                 plan_fname_2d, geom_fname_2d, dem_fname, cell_coord_path, depth_path, cell_facepoint_idx_path, facepoint_coord_path, cal_table_path, start_ts, nstarts, nevals,
                 map_interval, param_names_2d, param_bounds_2d, loss_func, locs, sim_ras_path_1d, res_output_dir_2d,
                 prj_fname_1d, bdy_fname_1d, flow_fname_1d, plan_fname_1d, geom_fname_1d, hydrograph_path,
                 param_names_1d, param_bounds_1d, breach_start, comparison_type=comparison_type, depth_cut=depth_cutoff,
                 binary_timesteps=binary_timesteps, run_1D=run_1D, gt_sat_rasters=gt_sat_rasters)
x_best, experiment = optim.optimize_parameters()
print('BEST X', x_best)

# Output the results.
output = optimize1D2D.Output(gt_ras_path_2d, sim_ras_path_2d, res_output_dir_2d, x_best)
output.save_botorch_experiment(experiment, '_'.join(param_names_2d + param_names_1d))

