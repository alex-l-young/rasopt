# Run the optimization.

# Library imports.
import yaml
from skopt.space import Real
from ax import RangeParameter, ParameterType
import os, sys
import pandas as pd
import argparse

# Local imports.
from rasopt.Optimize import optimize1D2D


def main(cfg):
    ### 2D MODEL CONFIGURATIONS
    ### ===============================
    # <<< Directories and file names. >>>
    # ------------------------------------

    # Ground truth directory.
    gt_ras_path_2d = cfg['dir_2d']['gt_ras_path_2d']

    # Simulation directory.
    sim_ras_path_2d = cfg['dir_2d']['sim_ras_path_2d']

    # Results output directory.
    res_output_dir_2d = cfg['dir_2d']['res_output_dir_2d']

    # Project file name.
    prj_fname_2d = cfg['dir_2d']['prj_fname_2d']

    # Boundary condition file name.
    bdy_fname_2d = cfg['dir_2d']['bdy_fname_2d']

    # Unstead flow file name.
    flow_fname_2d = cfg['dir_2d']['flow_fname_2d']

    # Plan HDF file name.
    plan_hdf_fname_2d = cfg['dir_2d']['plan_hdf_fname_2d']

    # Plan file name.
    plan_fname_2d = cfg['dir_2d']['plan_fname_2d']

    # Geometry HDF file name.
    geom_fname_2d = cfg['dir_2d']['geom_fname_2d']

    # DEM File path.
    dem_fname = cfg['dir_2d']['dem_fname']

    # <<< HDF Paths >>>
    # ------------------

    # Cell coordinate path.
    cell_coord_path = cfg['dir_2d']['cell_coord_path']

    # Water depths path.
    depth_path = cfg['dir_2d']['depth_path']

    # Manning's n calibration table path.
    cal_table_path = cfg['dir_2d']['cal_table_path']

    # Cell facepoint index path.
    cell_facepoint_idx_path = cfg['dir_2d']['cell_facepoint_idx_path']

    # Facepoint coordinate path.
    facepoint_coord_path = cfg['dir_2d']['facepoint_coord_path']

    # <<< Model Configurations >>>
    # -----------------------------

    # Comparison Type "Binary" or "Sensor".
    comparison_type = cfg['model_2d']['comparison_type']

    # SENSOR CALIBRATION
    # Number of hours after start when sensors are placed.
    start_ts = cfg['model_2d']['start_ts']

    # BINARY CALIBRATION

    # Time steps at which to do binary comparison.
    binary_timesteps = cfg['model_2d']['binary_timesteps']

    # Binary comparison depth cutoff.
    depth_cutoff = cfg['model_2d']['depth_cutoff'] # meters.

    # Raster cell dimensions in units of the coordinate system.
    cell_width_X = cfg['model_2d']['cell_width_X']
    cell_width_Y = cfg['model_2d']['cell_width_Y']

    # # Uncertainty spatial correlation radius.
    # radius = cfg['model_2d']['radius']
    #
    # # Depth error. The range of possible depth errors for a spurious cell in the uncertainty raster.
    # depth_error = cfg['model_2d']['depth_error']

    # No data value.
    nodata = cfg['model_2d']['nodata']

    # # Number of uncertainty seeds.
    # n_seeds = cfg['model_2d']['n_seeds']
    #
    # # Uncertainty type. "Binary" or "Depth".
    # uncertainty_type = cfg['model_2d']['uncertainty_type']
    #
    # # Maximum error probability for flipping cells to a spurious value.
    # max_probability = cfg['model_2d']['max_probability']

    # <<< Optimization Configurations >>>
    # -----------------------------------

    # Number of initializations.
    nstarts = cfg['optim']['nstarts']

    # Number of evaluations.
    nevals = cfg['optim']['nevals']

    # Number of hours to run model.
    n_ts = cfg['optim']['n_ts']

    # Mapping interval in minutes.
    map_interval = cfg['optim']['map_interval']

    # Parameter names that will be optimized. Must match the names in the model.
    param_names_2d = cfg['param_2d']['param_names_2d']

    # Parameter bounds. put "_g" after land roughness parameters.
    range_param_names = cfg['param_2d']['range_param_names_2d']
    param_lower_2d = cfg['param_2d']['param_lower_2d']
    param_upper_2d = cfg['param_2d']['param_upper_2d']
    param_bounds_2d = []
    for i, name in enumerate(range_param_names):
        low = param_lower_2d[i]
        up = param_upper_2d[i]
        param_bounds_2d.append(RangeParameter(name=name, parameter_type=ParameterType.FLOAT, lower=low, upper=up))
    # range_param_1 = RangeParameter(name='campagna_n_g', parameter_type=ParameterType.FLOAT, lower=.001, upper=.3)
    # range_param_2 = RangeParameter(name='centri_n_g', parameter_type=ParameterType.FLOAT, lower=.001, upper=.5)
    # param_bounds_2d = [range_param_1]

    # Loss function. ['MSE', 'RMSE', 'NNSE']
    loss_func = cfg['model_2d']['loss_func']

    # Sensor locations.
    # locs = [(4952299.22, 655041.01), (4950859.92, 655273.81), (4952264.88, 655921.65), (4951767.55, 655117.21)]
            # (4953264.49, 657720.27)]

    sensor_loc_fname = cfg['model_2d']['sensor_locations']
    if comparison_type == 'Sensor':
        loc_df = pd.read_csv(sensor_loc_fname, names=['Latitude', 'Longitude'])
        locs = list(zip(loc_df.Latitude, loc_df.Longitude))
    elif comparison_type == 'Sensor_Max':
        loc_df = pd.read_csv(sensor_loc_fname, names=['cell_index'])
        locs = list(loc_df.cell_index)


    ### ===============================

    ### 1D MODEL CONFIGURATIONS
    ### ===============================
    run_1D = cfg['flag_1d']['run_1D']

    # <<< Directories and file names. >>>
    # -----------------------------------

    # Simulation directory.
    sim_ras_path_1d = cfg['dir_1d']['sim_ras_path_1d']

    # Results output directory.
    res_output_dir_1d = cfg['dir_1d']['res_output_dir_1d']

    # Project file name.
    prj_fname_1d = cfg['dir_1d']['prj_fname_1d']

    # Boundary condition file name.
    bdy_fname_1d = cfg['dir_1d']['bdy_fname_1d']

    # Unstead flow file name.
    flow_fname_1d = cfg['dir_1d']['flow_fname_1d']

    # Plan HDF file name.
    plan_hdf_fname_1d = cfg['dir_1d']['plan_hdf_fname_1d']

    # Plan file name.
    plan_fname_1d = cfg['dir_1d']['plan_fname_1d']

    # Geometry HDF file name.
    geom_fname_1d = cfg['dir_1d']['geom_fname_1d']

    # <<< HDF Paths >>>
    # -----------------

    # Path to breach hydrograph.
    hydrograph_path = cfg['dir_1d']['hydrograph_path']

    # <<< Model Configurations >>>
    # ----------------------------

    # Start Datetime. Formatted as "21DEC2013,2400"
    start_datetime_1d = cfg['model_1d']['start_datetime_1d']

    # End Datetime. Formatted as "21DEC2013,2400"
    end_datetime_1d = cfg['model_1d']['end_datetime_1d']

    # Output interval for breach hydrograph in minutes.
    output_interval = cfg['model_1d']['output_interval']

    # Breach start.
    breach_start = cfg['model_1d']['breach_start']

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
    experiment = optim.optimize_parameters()

    # Output the results.
    output = optimize1D2D.Output(gt_ras_path_2d, sim_ras_path_2d, res_output_dir_2d)
    output.save_botorch_experiment(experiment, '_'.join(param_names_2d + param_names_1d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_1D2D_optimize',
        description='Runs the HEC RAS Bayesian optimization procedure.')
    parser.add_argument('--config', help='Configuration file path')
    args = parser.parse_args()

    # Load in configuration file.
    with open(args.config, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    print(cfg)

    # Run the main function with configurations.
    main(cfg)