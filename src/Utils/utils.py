# ==========================================================================================
# Utility functions for parameter scripts.
# Alex Young
# April 2021
# ==========================================================================================

# Library imports.
import os
import re
import psutil
import subprocess
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import rasterio
import pickle
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.space import Real
from scipy.stats.contingency import margins
from ax import RangeParameter, ParameterType
from geojson import Feature, Polygon, FeatureCollection
import osgeo.ogr as ogr
import geojson
try:
    import rascontrol
except:
    print('Could not import rascontrol library.')

try:
    import win32com.client
except:
    print('win32com.client could not be imported.')


# Local imports.
import land_cover_utils

# ==========================================================================================
# UTILITY FUNCTIONS
def get_set_dataset(fp, h5_path, get_ds=True, new_ds=None):
    """
    Either retrieves or replaces (gets or sets) the dataset from the hdf5 file given the path where the dataset lives.
    :param fp: Hdf5 file path.
    :param h5_path: Path to dataset within the hdf5 file. E.g., "Geometry/2D Flow/Cell Elevation"
    :param get_ds: Returns dataset in file when True, replaces dataset with new_ds when false.
    :param new_ds: New dataset to replace current one in file. Must be included if get_ds=False.
    :return: The dataset as a numpy array if getting, otherwise returns True.
    """
    # Open the file.
    f = h5py.File(fp, 'r+')

    # Get the dataset.
    ds = f[h5_path]

    if get_ds is True:
        dataset = ds[:]
        f.close()
        return dataset

    elif get_ds is False and new_ds is not None:
        ds[...] = new_ds
        f.close()

        # Check if the data was set correctly.
        f = h5py.File(fp, 'r+')
        try:
            set_correct = np.allclose(f[h5_path][:], new_ds)
        except Exception as e:
            print(e)
            print('Most likely trying to compare non-numerics. Assuming correct...')
            set_correct = True
            print('FILE DATASET', f[h5_path][:])
            print('NEW DATASET', new_ds)
        f.close()
        return set_correct

    else:
        return None


def extract_depths(fp, depth_path, coord_path):
    """
    Extracts depths from a HEC-RAS geometry file and returns data frame with columns for latitude, longitude, and
    depths at each time step.
    :param fp: File path to extract depth from.
    :param depth_path: Path to depth data within the file.
    :param coord_path: Path to coordinate data within the file.
    :return: Data frame with columns | lat | lon | Time_0 | Time_1 | ... | Time_n |. Time columns are the depth at time
        steps 0 to n.
    """
    # Get depth data.
    depths = get_set_dataset(fp, depth_path)

    # Get cell coordinates.
    coords = get_set_dataset(fp, coord_path)

    # Transpose depth data to column.
    depth_cols = np.transpose(depths)

    # Concatenate the coordinate and depth data frames.
    depth_loc_ar = np.concatenate((coords, depth_cols), axis=1)

    # Create data frame.
    cols = ['lon', 'lat'] + [f'Time_{i}' for i in range(depth_cols.shape[1])]
    depth_loc_df = pd.DataFrame(depth_loc_ar, columns=cols)

    return depth_loc_df


def closest_row(df, loc, nrows=1):
    """
    Finds the row that is closest to the location specified by Euclidian distance.
    :param df: Data frame where first two rows are "lat" and "lon" respectively.
    :param loc: Ideal location that will be used as closest point. Should be a tuple (lat, lon).
    :return: Numpy array of the row without the lat and lon columns.
    """
    # Distance function.
    calc_dist = lambda lat, lon: ((lat - loc[0])**2 + (lon - loc[1])**2)**0.5

    # Add a column for the distance to each row's location.
    df['dist'] = np.vectorize(calc_dist)(df['lat'], df['lon'])

    # Sort by distance.
    df_sort = df.sort_values(by=['dist'])
    df_sort.reset_index(inplace=True, drop=True)

    # Select the top n rows.
    close_rows = df_sort.iloc[0:nrows, :]

    # Drop the location and distance columns. Create numpy array.
    close_rows = close_rows.drop(columns=['lat', 'lon', 'dist'])
    close_rows_ar = close_rows.to_numpy()

    # If only one row is requested, just return that row, rather than as a 2D array.
    if nrows == 1:
        close_rows_ar = close_rows_ar[0]

    return close_rows_ar


def location_from_cell_id(fp, cell_id, cell_coord_path):
    """
    Returns the lat and lon given a cell ID.
    :param fp: Path to file containing cell location information.
    :param cell_id: Cell ID. Should be an integer value.
    :param cell_coord_path: Path to the cell coordinates in the file.
    :return: lat, lon.
    """
    # Get coordinates.
    coords = get_set_dataset(fp, cell_coord_path)

    # Cell list.
    cell_idxs = np.arange(0, coords.shape[0])  # ASSUMES THAT THE CELLS ARE IN ORDER.
    cell_idxs = cell_idxs[:, None]

    # Concatenate.
    coords_cells = np.concatenate((coords, cell_idxs), axis=1)

    # Select row with cell_id.
    lat = coords_cells[coords_cells[:, 2] == cell_id, 1]
    lon = coords_cells[coords_cells[:, 2] == cell_id, 0]

    return lat[0], lon[0]


def plot_point_time_series(gt_fp, sim_fp, locations, cell_coord_path, depth_path, iter_num, output_path=None):
    """
    Plots the simulated vs observed time series at the points being used to optimize the parameter set.
    :param gt_fp: Path to ground truth pXX file.
    :param sim_fp: Path to simulated pXX file.
    :param locations: Locations to plot as a list of tuples [(lat, lon)]
    :param cell_coord_path: Path to cell coordinates within the pXX file.
    :param depth_path: Path to depth information within the pXX file.
    :param iter_num: Iteration number for labeling figures.
    :param output_path: Specify the output path if desired.
    :return: Saves the figures with the iteration number.
    """
    # Extract ground truth depths.
    gt_dep_df = extract_depths(gt_fp, depth_path, cell_coord_path)

    # Extract simulated depths.
    sim_dep_df = extract_depths(sim_fp, depth_path, cell_coord_path)

    for loc in locations:
        # Select closest row from simulated data set.
        sim_row = closest_row(sim_dep_df, loc)

        # Select closest row from measured data set.
        gt_row = closest_row(gt_dep_df, loc)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(gt_row, 'b', label='Ground Truth')
        ax.plot(sim_row, 'r', label='Optimized')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Depth')
        ax.set_title('Point comparison')
        ax.legend()

        if output_path is None:
            plt.savefig(f'Point_Timeseries_{loc[0]}_{loc[1]}_{iter_num}.png')
        else:
            plt.savefig(os.path.join(output_path, f'Point_Timeseries_{loc[0]}_{loc[1]}_{iter_num}.png'))

    return


def controller_hec_run(proj_path, controller_version):
    """
    Runs the HEC-RAS model with the HEC-RAS Controller.
    :param proj_path: Path to project to run.
    :param controller_version: Version of HEC-RAS to run. e.g., "RAS507.HECRASController"
    :return: None.
    """
    # Instantiate the HECRAS controller.
    hec = win32com.client.Dispatch(controller_version)

    # Project.
    RAS_project = os.path.join(proj_path)
    hec.Project_Open(RAS_project)

    # Handle versions differently.
    if '507' in controller_version:
        NMsg, TabMsg, block = None, None, True
        cur_plan_out = hec.Compute_CurrentPlan(NMsg, TabMsg, block)
        print(cur_plan_out)

        # Close HEC-RAS and quit.
        hec.QuitRas()
        del hec

    elif '41' in controller_version:
        NMsg, TabMsg = None, None
        cur_plan_out = hec.Compute_CurrentPlan(NMsg, TabMsg)
        print(cur_plan_out)

        del hec


def RMSE(m, o):
    "Computes RMSE from modeled and observed values."
    return np.sqrt(np.sum(np.square(m - o)) / len(m))

def MSE(m, o):
    "Computes MSE from modeled and observed values."
    return np.sum(np.square(m - o)) / len(m)

def NNSE(m, o):
    "Computes the Normalized Nash-Sutcliffe Efficiency from modeled and observed values."
    NSE = 1 - (np.sum(np.square(m - o)) / np.sum(np.square(o - np.mean(o))))
    normNSE = 1 / (2 - NSE)
    return normNSE

def KGE(m, o):
    "Computes Kling-Gupta Efficiency from the modeled and observed values."
    return

def squared_resid_like(m, o, T):
    "Squared variance of the residual likelihood calculation. m - modeled, o - observed"
    resid = np.array(o) - np.array(m)
    resid_var = np.var(resid)
    print('RESIDUAL VARIANCE', resid_var)

    L = (resid_var) ** -T
    return L


def inun_fit(gt_depths, sim_depths, depth_cut=None):
    """
    Calculates the inundation fit between the ground truth depths and sim depths above a certain cutoff depth.
    Fit is defined as sum(Area_sim & Area_gt) / sum(Area_sim | Area_gt) is a metric that describes the amount of
    overlap between the two inundated area. A fit of 0 is no overlap and a fit of 1 is complete overlap.
    Fit only compares the depths in a binary manner, the numerical depth at each cell is not taken into account.
    :param gt_depths: Ground truth depths at a single time step. [Numpy array]
    :param sim_depths: Simulation depths at a single time step. [Numpy array]
    :param depth_cut: Depth over which a cell is considered "inundated". [float in meters].
    :return: Fit value.
    """
    if depth_cut is None:
        gt_inun = gt_depths
        sim_inun = sim_depths
    else:
        # Booleans where inundation has occurred.
        gt_inun = gt_depths > depth_cut
        sim_inun = sim_depths > depth_cut

    # If both are all zeros, return 1.
    if np.sum(gt_inun) == 0 and np.sum(sim_inun) == 0:
        return 1

    # Compute fit.
    fit = np.sum(gt_inun & sim_inun) / np.sum(gt_inun | sim_inun)

    return fit


def inun_sensitivity(gt_depths, sim_depths, depth_cut=None):
    """
    Compute the sensitity of the model {True Pos. / (True Pos. + False Neg.)}
    :param gt_depths: Ground truth depths at a single time step. [Numpy array]
    :param sim_depths: Simulation depths at a single time step. [Numpy array]
    :param depth_cut: Depth over which a cell is considered inundated. [float in meters]
    :return: Sensitivity of the model at that time step.
    """
    # Depth cutoff.
    if depth_cut is None:
        gt_inun = gt_depths
        sim_inun = sim_depths
    else:
        # Booleans where inundation has occurred.
        gt_inun = gt_depths > depth_cut
        sim_inun = sim_depths > depth_cut

    # Number of true positives.
    TP = np.sum(gt_inun & sim_inun)

    # Number of false negatives.
    FN = np.sum(gt_inun & ~sim_inun)

    # Sensitivity.
    sens = TP / (TP + FN)
    return sens


def inun_error(gt_depths, sim_depths, type, depth_cut=None):
    """
    Compute the type I or type II of the model.
    Type I = |FN - FP| / (FN + TP)
    Type II = (FN + FP) / (FN + TP)
    :param gt_depths: Ground truth depths at a single time step. [Numpy array]
    :param sim_depths: Simulation depths at a single time step. [Numpy array]
    :param type: Type 1 or 2 error (specify either 1 or 2).
    :param depth_cut: Depth over which a cell is considered inundated. [float in meters]
    :return: Type I or II of the model at that time step.
    """
    # Depth cutoff.
    if depth_cut is None:
        gt_inun = gt_depths
        sim_inun = sim_depths
    else:
        # Booleans where inundation has occurred.
        gt_inun = gt_depths > depth_cut
        sim_inun = sim_depths > depth_cut

    # False negatives.
    FN = np.sum(gt_inun & ~sim_inun)

    # True positives.
    TP = np.sum(gt_inun & sim_inun)

    # False positives.
    FP = np.sum(sim_inun & ~gt_inun)

    if type == 1:
        error = np.abs(FN - FP) / (FN + TP)
    elif type == 2:
        error = (FN + FP) / (FN + TP)
    else:
        error = None

    return error


def raster_value_at_location(locations, raster_path):
    """
    Finds the raster cell value corresponding to a data frame of locations.
    :param locations: Pandas data frame of locations with columns "Latitude" and "Longitude".
    :param raster_path: Path to a georeferenced geotiff.
    :return: The locations data frame with new column corresponding to the raster values.
    """
    # Make an iterable of x,y coordinates.
    coords = list(zip(list(locations.lon), list(locations.lat)))

    src = rasterio.open(raster_path)
    locations['Raster_Value'] = [x[0] if x[0] >= 0.0 else np.nan for x in src.sample(coords)]

    # Close the dataset.
    src.close()

    return locations


def satellite_groundtruth_rasters(comparison_timesteps, gt_plan_hdf_fp, cell_fp_idx_path, facepoint_coord_path,
                                  depth_path, cell_coord_path, cell_width_X, cell_width_Y, nodata, n_seeds, radius,
                                  uncertainty_type='Depth', depth_error=0.1, max_probability=1):

    uncertain_filepaths = {}
    for timestep in comparison_timesteps:
        # Create the depth geojson for a timestep.
        depth_gj = depth_geojson(gt_plan_hdf_fp, cell_fp_idx_path, facepoint_coord_path, depth_path, cell_coord_path,
                                 timestep)

        # Rasterize the depth geojson.
        raster_array, raster_fp = rasterize_depth_geojson(depth_gj, cell_width_X, cell_width_Y, nodata)

        # Create uncertainty rasters.
        if uncertainty_type == 'Depth':
            uncertain_array, alter_array = add_satellite_uncertainty(raster_array, n_seeds, radius,
                                                                     uncertainty_type='Depth',
                                                                     max_probability=max_probability,
                                                                     depth_error=depth_error)
        else:
            uncertain_array, alter_array = add_satellite_uncertainty(raster_array, n_seeds, radius,
                                                                     uncertainty_type='Binary',
                                                                     max_probability=max_probability)

        # Save uncertainty raster.
        uncertain_fp = create_error_geotiff(uncertain_array, raster_fp)
        uncertain_filepaths[timestep] = uncertain_fp

    return uncertain_filepaths


def satellite_groundtruth_rasters_v2(comparison_timesteps, gt_plan_hdf_fp, dem_fp, cell_fp_idx_path, facepoint_coord_path,
                                  depth_path, cell_coord_path, cell_width_X, cell_width_Y, nodata, n_seeds, radius,
                                  uncertainty_type='Depth', depth_error=0.1, max_probability=1):

    gt_raster_fps = {}
    for timestep in comparison_timesteps:
        print('CONVERTING GT RASTER TIMESTEP:', timestep)
        raster_array, raster_fp = land_cover_utils.terrain_water_depth(dem_fp, gt_plan_hdf_fp, cell_fp_idx_path,
                                        facepoint_coord_path, depth_path, cell_coord_path, timestep,
                                                                       out_fname_prefix='GT', nodata=-999.0)

        # # Create the depth geojson for a timestep.
        # depth_gj = depth_geojson(gt_plan_hdf_fp, cell_fp_idx_path, facepoint_coord_path, depth_path, cell_coord_path,
        #                          timestep)
        #
        # # Rasterize the depth geojson.
        # raster_array, raster_fp = rasterize_depth_geojson(depth_gj, cell_width_X, cell_width_Y, nodata)

        # TODO: DONT IMPLEMENT UNCERTAINTY YET
        # # Create uncertainty rasters.
        # if uncertainty_type == 'Depth':
        #     uncertain_array, alter_array = add_satellite_uncertainty(raster_array, n_seeds, radius,
        #                                                              uncertainty_type='Depth',
        #                                                              max_probability=max_probability,
        #                                                              depth_error=depth_error)
        # else:
        #     uncertain_array, alter_array = add_satellite_uncertainty(raster_array, n_seeds, radius,
        #                                                              uncertainty_type='Binary',
        #                                                              max_probability=max_probability)

        # # Save uncertainty raster.
        # uncertain_fp = create_error_geotiff(uncertain_array, raster_fp)
        gt_raster_fps[timestep] = raster_fp

    return gt_raster_fps


def satellite_success_metric(gt_sat_rasters, comparison_timesteps, sim_plan_hdf_fp, depth_path, cell_coord_path,
                             comparison_function, depth_cut=0.01):
    """

    :param gt_sat_rasters: Ground truth satellite rasters as a dictionary of file paths where key is the time step.
    :param comparison_timesteps: Time steps at which to compare the GT rasters to the model rasters.
    :param sim_plan_hdf_fp: Simulation HDF plan file path.
    :param depth_path: Path to depth data in the simulation hdf plan file.
    :param cell_coord_path: Path to cell coordinate path in the simulation hdf plan file.
    :param comparison_function: Comparison function to use to compare GT to simulation.
    :param depth_cut: Depth cutoff above which inundation has occurred. Default 1 cm = 0.01 m.
    :return:
    """
    # Ground truth and simulation depths.
    sim_depths = extract_depths(sim_plan_hdf_fp, depth_path, cell_coord_path)

    timestep_metrics = []
    for timestep, in comparison_timesteps:
        # Get corresponding uncertain_fp.
        uncertain_fp = gt_sat_rasters[timestep]

        # Get the raster depth at each location.
        locations_df = sim_depths[['lat', 'lon']]
        raster_values_df = raster_value_at_location(locations_df, uncertain_fp)

        # Compute the success metric between the uncertain ground truth and the simulation results.
        uncertain_values = raster_values_df['Raster_Value']
        sim_values = sim_depths[f'Time_{timestep}']
        timestep_metrics.append(comparison_function(uncertain_values, sim_values, depth_cut=depth_cut))

    # Combine the timestep metrics into a single metric.
    mean_metric = sum(timestep_metrics) / len(timestep_metrics)

    return mean_metric


def compute_raster_loss(gt_sat_rasters, comparison_timesteps, sim_plan_hdf_fp, dem_fp, depth_path, cell_coord_path,
                        cell_fp_idx_path, facepoint_coord_path, comparison_function, depth_cut = 0.01):
    """

    :param gt_sat_rasters: Ground truth satellite rasters as a dictionary of file paths where key is the time step.
    :param comparison_timesteps: Time steps at which to compare the GT rasters to the model rasters.
    :param sim_plan_hdf_fp: Simulation HDF plan file path.
    :param depth_path: Path to depth data in the simulation hdf plan file.
    :param cell_coord_path: Path to cell coordinate path in the simulation hdf plan file.
    :param comparison_function: Comparison function to use to compare GT to simulation.
    :param depth_cut: Depth cutoff above which inundation has occurred. Default 1 cm = 0.01 m.
    :return:
    """

    timestep_metrics = []
    for timestep in comparison_timesteps:
        # Get corresponding ground truth satellite raster file path.
        gt_sat_raster_fp = gt_sat_rasters[timestep]
        with rasterio.open(gt_sat_raster_fp, 'r') as ds:
            gt_array = ds.read(1)

        # Create a raster from the simulation data.
        sim_array, sim_raster_fp = land_cover_utils.terrain_water_depth(dem_fp, sim_plan_hdf_fp, cell_fp_idx_path,
                                            facepoint_coord_path, depth_path, cell_coord_path, timestep, nodata=-999.0)

        # Compute the loss metric.
        timestep_metrics.append(comparison_function(gt_array, sim_array, depth_cut=depth_cut))

    # Combine the timestep metrics into a single metric.
    mean_metric = sum(timestep_metrics) / len(timestep_metrics)

    return mean_metric


def update_gxx_file(gXX_fpath, params, param_names, all_param_names):
    """
    CURRENTLY UNUSED
    Writes new Manning's n parameter values in the gXX file,
    which is only present in the Windows distribution of HEC-RAS.
    :param gXX_fpath: Path to gXX file.
    :param params: Parameter values as a list.
    :param param_names: Parameter names, must match the order of params.
    :param all_param_names: All parameter names possible as a list.
    :return: Updates the parameter file with the new Manning's n values.
    """
    # Count number of non-nan parameters.
    num_not_nan = np.count_nonzero(~np.isnan(params))

    # Update the gXX file.
    with open(gXX_fpath, 'r') as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            # Try to get the land cover name from the line.
            land_cover_name = None
            try:
                land_cover_name = re.search('(.*),', line).group(1)
            except:
                pass

            if 'LCMann Time' in line:
                # Current datetime formatted as 'Jun/14/2021 20:42:33'
                dt_str = datetime.strftime(datetime.now(), '%b/%d/%Y %H:%M:%S')

                # Sub in new time.
                lcmann_time = re.sub('=(.*)', '=' + dt_str, line)
                new_lines.append(lcmann_time)

            elif 'LCMann Table' in line:
                # Add the number of paramters.
                lcmann_table = f'LCMann Table={num_not_nan}\n'
                new_lines.append(lcmann_table)

                # Add the new parameter values.
                for i in range(len(param_names)):
                    # Don't update if parameter value is a nan.
                    if np.isnan(params[i]):
                        continue
                    new_lines.append(f'{param_names[i]},{params[i]}\n')

            elif land_cover_name in all_param_names:
                # Skip any lines that already have parameter values present.
                print(f'Found {land_cover_name} in gXX file. Removing...')
                continue

            else:
                new_lines.append(line)

    # Print new lines added.
    print('NEW LINES', new_lines[-12:])

    # Write the new gXX file.
    with open(gXX_fpath, 'w') as f:
        f.writelines(new_lines)


def cell_lc_types(coords, raster_fp):
    """
    Get the land cover type of each cell in the domain.
    :param coords: Coordinates of each cell as a list of tuples in the same
        projection as the raster. [(655500, 4952000)].
    :param raster_fp: Path to land cover raster.
    :return: List of land cover types that correspond to the coordinate list passed in.
    """
    # Open geotiff with rasterio.
    with rasterio.open(raster_fp, "r") as src:
        ras_vals = [x[0] for x in src.sample(coords)]

    return ras_vals


def pickle_object(obj, fp):
    """
    Pickle an object to a specified file path.
    :param obj: Object to pickle.
    :param fp: File path to save to.
    :return: None
    """
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)


def unpickle_object(fp):
    """
    Unpickles a pickled file, returns object.
    :param fp: Path to pickled object.
    :return: Unpickled object.
    """
    with open(fp, 'rb') as f:
        obj = pickle.load(f)

    return obj


def cleanup_pickles(pickle_dir):
    """
    Removes any files with .pickle extension in the specified directory.
    :param pickle_dir: Directory to rid of pickles.
    :return: None, and a pickle-free directo-ree.
    """
    for file in os.listdir(pickle_dir):
        if os.path.splitext(file)[-1] == '.pickle':
            os.remove(os.path.join(pickle_dir, file))
    return


def fit_gp_regressor(X, Y, kernel):
    """
    Fits a GP regressor on data given kernel specifications.
    :param X: X training points.
    :param Y: Y training points.
    :param kernel: Sklearn kernel object.
    :return: A GP regressor object fit on the X, Y data.
    """
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, Y)
    return gpr


def gp_grid_predict(gp_regr, n_grid, bounds):
    """
    Predicts the value at all grid points on the
    :param gp_regr: GP regressor object.
    :param n_grid: Number of grid points on each side of the grid hypercube.
    :param bounds: Bounds of each parameter as a list of tuples.
    :return: Fully predicted domain as numpy array.
    """
    # Create empty grid.
    grid_dim = tuple([n_grid for _ in bounds])
    grid = np.zeros(grid_dim)

    # Axes from the bounds to predict on. X test data.
    # GP regressor is normalized from 0 to 1. So need to predict on this interval and then relate to bounds interval.
    # test_data = [list(np.linspace(bounds[i][0], bounds[i][1], n_grid)) for i in range(len(bounds))]
    test_data = [list(np.linspace(0, 1, n_grid)) for i in range(len(bounds))]

    # Create indices at which to place the predicted Y test data on the grid.
    test_idxs = [list(range(n_grid)) for _ in range(len(bounds))]

    # Iterproduct on the axes and index lists to get all possible combinations of points.
    test_data_full = np.array(list(itertools.product(*test_data)))
    test_idxs_full = list(itertools.product(*test_idxs))

    # Predict on the test data.
    # gp_pred = np.vectorize(lambda X: gp_regr.predict(np.array([[X]])))
    # y_test = gp_pred(test_data_full)
    y_test = gp_regr.predict(test_data_full)

    # Populate the grid.
    grid[tuple(np.array(test_idxs_full).T)] = y_test.T

    # Create data frame for inspection.
    try:
        df = pd.DataFrame({'grid': grid, 'test_data': test_data_full.ravel()})
    except:
        print('Test_data_full more than 1D, passing on DataFrame')

    return grid


def likelihood_transform(pred_domain, T):
    """
    Covert the prediction domain to likelihood values given a shape parameter T.
    The likelihood function used is (sigma_e / (n-2))^-T.
    :param pred_domain: Domain to be converted to a likelihood surface.
    :param T: Shape parameter ranging from 0 (all equal likelihood) to infinity (max value has all the likelihood).
    :return: Likelihood surface as a numpy array.
    """
    # Compute the likelihood at each point.
    L = (pred_domain / (pred_domain.size - 2)) ** -T

    # Normalize the likelihood.
    norm_L = L / np.sum(L[:])

    return norm_L


def calc_marginal_dists(distribution, save=False, save_fp=None, col_names=None):
    # Calculate the marginal distributions from the posterior likelihood.
    marginal_dists = margins(distribution)

    # Collect marginal distributions in a data frame and save.
    d = {}
    c = 0
    for marg in marginal_dists:
        d[c] = marg.ravel()
        c += 1
    marg_df = pd.DataFrame(d)

    if col_names is not None:
        marg_df.columns = col_names

    if save is True:
        marg_df.to_csv(save_fp, index=False)

    return marg_df


def optimal_param_from_dist(dist, bounds):
    """
    Returns the argmax parameter values given an n-dim distribution and list of bounds as tuples.
    :param dist: Distribution to argmaximize. Distribution should be square (i.e., all dimensions equal)
    :param bounds: Bounds of each parameter in order of the distribution dimensions. (low, high)
    :return: argmax parameters.
    """
    # Get the argmax indices.
    max_idxs = np.unravel_index(np.argmax(dist, axis=None), dist.shape)

    # Convert bounds into arrays.
    opt_param = []
    for i in range(dist.ndim):
        param_ar = np.linspace(bounds[i][0], bounds[i][1], dist.shape[0])
        opt_param.append(param_ar[max_idxs[i]])

    return opt_param


def normalize_dist(dist):
    "Normalizes distribution to sum to 1"
    # Normalize the likelihood.
    norm_dist = dist / np.sum(dist[:])

    return norm_dist


def normalize_parameters(params):
    """
    Creates a list of normalized parameters in the skopt style.
    :param params: List of parameters as skopt Real objects.
    :return: List of normalized parameters with the same name.
    """
    norm_params = []
    for param in params:
        # norm_params.append(Real(0, 1, name=param.name))
        norm_params.append(RangeParameter(name=param.name, parameter_type=ParameterType.FLOAT, lower=0, upper=1))

    return norm_params


def denorm_parameter(param_val, low, high):
    """
    Denorms parameter value back to original scale.
    :param parameter: Parameter value.
    :param low: Minimum value on real scale.
    :param high: Maximum value on real scale.
    :return: De-normalized parameter.
    """
    denorm_val = param_val * (high - low) + low

    return denorm_val


def generate_breach_timeseries(upstream_fp, k, lag, a, breach_start, breach_end):
    """
    Generates breach time series from upstream using a modified weir equation:
    Q(t) = (2/3) * k * (2g)^(1/2) * h(t-lag)^(3/2) + a
    Max lag must be no greater than the first upstream datetime - breach_start.

    :param upstream_fp: Path to upstream depth readings as data frame with columns | Datetime | Stage |
    :param k: Weir fitting constant.
    :param lag: Number of lag intervals to upstream depth that accounts for distance between the
        upstream gauge and the breach. Lag interval is 10 minutes, so the lag is multiplied by 10 in the model.
    :param a: Adjustment parameter that dictates how far between the peak and trough.
    :param breach_start: Start time for breach, inclusive. Format "%d/%m/%Y %H:%M"
    :param breach_end: End time for breach, inclusive. Format "%d/%m/%Y %H:%M"
    :return: 1-D Numpy array containing the breach outflow in upstream stage units^3/second.
    """
    # Load the upstream data. Convert datetime column to datetime.
    upstream_df = pd.read_csv(upstream_fp)
    upstream_df['Datetime'] = pd.to_datetime(upstream_df['Datetime'])

    # Generate the breach time series.
    breach_dates = []
    start_dt = datetime.strptime(breach_start, "%d/%m/%Y %H:%M")
    cur_dt = start_dt
    end_dt = datetime.strptime(breach_end, "%d/%m/%Y %H:%M")
    while cur_dt <= end_dt:
        breach_dates.append(cur_dt)
        cur_dt = cur_dt + timedelta(minutes=10)

    # Collect the upstream data to be processed.
    lag = np.float(lag)
    upstream_start = start_dt - timedelta(minutes=lag*10)
    upstream_end = end_dt - timedelta(minutes=lag*10)
    upstream_depth = upstream_df.loc[(upstream_df['Datetime'] >= upstream_start) &
                                     (upstream_df['Datetime'] <= upstream_end), 'Stage'].to_numpy()

    # Calculate the breach time series.
    Q = (2/3) * k * (2 * 9.81)**(1/2) * upstream_depth**(3/2) + a

    return Q


def extract_hec_breach_hydrograph(proj_path):

    # Kill any open hec ras processes.
    for p in psutil.process_iter():
        if p.name().lower() == 'ras.exe':
            p.kill()

    # Try this twice.
    rc = rascontrol.RasController(version='41')
    rc.open_project(proj_path)

    # Get results
    profiles = rc.get_profiles()
    lat_list = rc._simple_node_list('lateral_struct')[0]
    ls_id = lat_list.ls_id
    breach_flow = []
    times = []
    for i in range(1, len(profiles)):
        profile = profiles[i]
        times.append(profile.name)
        outflow = rc._get_node('lateral_struct', ls_id).value(profile, rascontrol.Q_WEIR)
        breach_flow.append(outflow)

    times = [make_dt(time) for time in times]

    del rc

    return breach_flow, times


def depth_geojson(pXX_hdf_fp, cell_fp_idx_path, facepoint_coord_path, depth_path, cell_coord_path, timestep,
                  elevation=False):
    """
    Makes a geojson of the flood domain with water depth value in each polygon.
    :param pXX_hdf_fp: Path to pXX.hdf file that contains geometry and depth information.
    :param cell_fp_idx_path: Path to the Cells FacePoint Indexes data.
    :param facepoint_coord_path: Path to the FacePoints Coordinate data.
    :param depth_path: Path to depth data.
    :param cell_coord_path: Path to cell coordinates in depth data.
    :param timestep: Time step to grab, zero-indexed.
    :param elevation: Set to true to add on the cell minimum elevation for ASL depth value.
    :return: Geojson feature collection of polygons.
    """
    # Read in file.
    print(pXX_hdf_fp)
    f = h5py.File(pXX_hdf_fp, 'r+')

    cell_fp_idx = f[cell_fp_idx_path][:]
    facepoint_coords = f[facepoint_coord_path][:]

    # Cell minimum elevation.
    cell_min_elev = f['Geometry/2D Flow Areas/Secchia_Panaro/Cells Minimum Elevation'][:]
    cell_min_elev = np.nan_to_num(cell_min_elev, nan=-999)

    # Extract ground truth depths.
    gt_dep_df = extract_depths(pXX_hdf_fp, depth_path, cell_coord_path)
    depths = gt_dep_df.iloc[:, timestep + 2].to_numpy()

    # Add on cell elevation.
    if elevation is True:
        depths = depths + cell_min_elev

    # fig, ax = plt.subplots(figsize=(7,7))
    feature_list = []
    for i in range(cell_fp_idx.shape[0]):
        fp_idxs = cell_fp_idx[i, :]
        cell_fp_X = facepoint_coords[fp_idxs[fp_idxs >= 0], 0]
        cell_fp_Y = facepoint_coords[fp_idxs[fp_idxs >= 0], 1]

        # Zip into list of tuples.
        cell_poly = list(zip(cell_fp_X, cell_fp_Y))
        mesh_poly = Polygon([cell_poly])
        feature = Feature(geometry=mesh_poly, properties={"Depth": depths[i]})
        feature_list.append(feature)

    # Create feature collection.
    feature_collection = FeatureCollection(feature_list)

    return feature_collection


def rasterize_depth_geojson(depth_geojson, cell_width_X, cell_width_Y, timestep, nodata=-999.0):
    """
    Rasterize the depth geojson into a numpy array.
    :param depth_geojson: Geojson FeatureCollection of flood cell polygons where each polygon has a property "Depth".
    :param x_cell_width: Width of raster cell in coordinate units (X-direction).
    :param y_cell_width: Width of raster cell in coordinate units (Y-direction).
    :param timestep: The model time step the depth geojson corresponds to. This function saves the raster with the
        timestep in the file name "mesh_raster_<timestep>.tif".
    :return: Rasterized geojson as an array. Saves the rasterized array in the current directory.
    """
    # Get current directory.
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    # Save the geojson feature collection temporarily.
    gjson_fp = os.path.join(cur_dir, 'Geo_Files', 'mesh_gjson.geojson')
    with open(gjson_fp, 'w') as gf:
        geojson.dump(depth_geojson, gf)

    # Raster path.
    raster_fp = os.path.join(cur_dir, 'Geo_Files', f'mesh_raster_{timestep}.tif')

    # Rasterize the geojson feature collection.
    layer_name = 'mesh_gjson'
    property_name = 'Depth'
    source_ds = ogr.Open(gjson_fp)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    gdal_command = ("gdal_rasterize -at -l {} -a {} -tr {} {} -a_nodata {} -te {} {} {} {} -ot Float32 "
                    "-of GTiff {} {}".format(layer_name, property_name, cell_width_X,
                                             cell_width_Y, nodata, x_min, y_min, x_max,
                                             y_max, gjson_fp, raster_fp))
    print(gdal_command)
    subprocess.call(gdal_command, shell=True)

    # Open back up the raster and read out the array.
    with rasterio.open(raster_fp) as dataset:
        raster_array = dataset.read(1)

    return raster_array, raster_fp


# def rasterize_depth_geojson_to_dem()


def add_satellite_uncertainty(raster_array, n_seeds, radius, uncertainty_type='Depth', max_probability=1,
                              depth_error=0.1):
    """
    Adds depth or binary uncertainty to the satellite image raster. Areas with errors are often spatially correlated,
    so this function adds uncertainty around a number of randomly-placed "seed" locations. The points are clustered
    around the seed locations using an exponential decay length scale where cells closest to the seed locations have
    the highest probability of being changed to spurious values.
    :param raster_array: Satellite image.
    :param n_seeds: Number of locations to add uncertainty.
    :param radius: Radius in number of cells from the seed at which the probability of an error is 0.05.
    :param uncertainty_type: "Depth" or "Binary". If "Depth", the error cells will be set to a value that is between
                    a +- depth_error.
    :param max_probability: Maximum probability that a cell will be erroneous. The individual probability contributions
        from each seed are summed, so some cells may have a probability of error greater than 1. This can be changed to
        a value of less than 1 if desired.
    :param depth_error: Maximum deviation in depth from the measured value. The depth error is added to each error cell
        and ranges from -depth error to +depth_error randomly chosen from a uniform distribution.
    :return: error_array - Satellite array with error cells added. alter_array - array of zeros and ones where the ones
        are where the error cells have been added.
    """

    # Create binary inundation array if uncertainty_type is 'Binary'.
    if uncertainty_type == 'Binary':
        binary_raster_array = raster_array.copy()
        binary_raster_array[binary_raster_array > 0] = 1

    # Indices of flooded cells as a list of tuples.
    flood_where = np.where(raster_array > 0.0)
    flood_idxs = np.array(list(zip(*flood_where)))

    # Choose n_seeds random points from flood_idxs to create misclassification zones.
    error_idxs = np.random.choice(np.arange(len(flood_idxs)), n_seeds, replace=False)
    seed_idxs = flood_idxs[error_idxs]

    # Assign misclassify probabilities to every cell in the raster with non-nodata value.
    # Compute the length scale.
    l = np.sqrt(-radius / (2 * np.log(0.05)))
    prob_dist = lambda d: np.exp(-d / (2 * l))

    L2_norm = lambda x1, y1, x2, y2: np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Compute distances to each of the seeds and assign probabilities.
    error_array = np.ones(raster_array.shape) * -999.0
    alter_array = np.zeros(raster_array.shape)
    for i in range(raster_array.shape[0]):
        for j in range(raster_array.shape[1]):
            cell_value = raster_array[i,j]
            if cell_value >= 0.0:
                # Compute L2 norm between seeds and cell location.
                probs = []
                for seed_idx in seed_idxs:
                    norm = L2_norm(i, j, seed_idx[0], seed_idx[1])
                    probs.append(prob_dist(norm))

                # Add probabilities together.
                error_prob = min(sum(probs), max_probability)

                # Determine whether cell should be misclassified.
                test_num = np.random.uniform(0,1)
                if test_num < error_prob and uncertainty_type == 'Depth':
                    error_array[i, j] = raster_array[i, j] + np.random.uniform(-depth_error, depth_error)
                    alter_array[i, j] = 1
                elif test_num >= error_prob and uncertainty_type == 'Depth':
                    error_array[i, j] = raster_array[i, j]
                elif test_num < error_prob and uncertainty_type == 'Binary':
                    error_array[i, j] = 1 - binary_raster_array[i, j]
                    alter_array[i, j] = 1
                elif test_num >= error_prob and uncertainty_type == 'Binary':
                    error_array[i, j] = binary_raster_array[i, j]

    # Make sure there are no negative depths other than the nodata value.
    error_array[(error_array < 0) & (error_array > -999)] = 0

    print('ALTER ARRAY SUM: {}'.format(np.sum(alter_array)))

    return error_array, alter_array


def create_error_geotiff(error_array, raster_fp):
    # Extract geo-metadata from the raster.
    with rasterio.open(raster_fp) as ds:
        crs = ds.crs
        transform = ds.transform

    # Write the error array to a new geotiff file. File name is "GT_<raster_fname>.tif"
    error_dir = os.path.dirname(raster_fp)
    error_fp = os.path.join(error_dir, "GT_" + os.path.basename(raster_fp))
    with rasterio.open(
            error_fp,
            'w',
            driver='GTiff',
            height=error_array.shape[0],
            width=error_array.shape[1],
            count=1,
            dtype=error_array.dtype,
            crs=crs,
            transform=transform,
            nodata=-999.0
    ) as dst:
        dst.write(error_array, 1)

    return error_fp


def make_dt(date_str):
    "Makes a date formatted as %d%b%Y %H%M into a datetime object, handling 24:00 edge case."
    if '2400' not in date_str:
        dt = datetime.strptime(date_str, '%d%b%Y %H%M')
    else:
        date_split = date_str.split(' ')
        dt = datetime.strptime(date_split[0], '%d%b%Y') + timedelta(days=1)

    return dt


if __name__ == '__main__':
    # a = np.array([1, 2, 3, 4, 5])
    # b = np.array([[1, 2, 3, 4, 5],
    #               [2, 3, 4, 5, 6],
    #               [3, 4, 5, 6, 7],
    #               [4, 5, 6, 7, 8],
    #               [5, 6, 7, 8, 9,]])
    # print(RMSE(a, b[:,0]))

    # Test run of the 1D HEC-RAS model.
    proj_path = r"C:\Users\ay434\Documents\Secchia_River_1D_2014_Sim\levee_breach_14.prj"
    # proj_path = r"C:\Users\ay434\Documents\Secchia_flood_2014_Alter\Secchia_Panaro.prj"
    proj_path = r"C:\Users\ay434\Documents\10_min_Runs\Secchia_flood_2014_10min_Sim\Secchia_Panaro.prj"
    controller_version = "RAS507.HECRASCONTROLLER"
    controller_hec_run(proj_path, controller_version)