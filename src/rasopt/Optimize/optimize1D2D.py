# ====================================================================================================
# Class containing all the optimization procedures. Full Bayesian Optimization wrapper for HEC-RAS.
# Alex Young
# June 2021
# ====================================================================================================

# Library imports.
import numpy as np
from numpy.random import default_rng
import pandas as pd
from datetime import datetime, timedelta
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_gaussian_process
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from rtree import index
import pickle
import os, sys
import subprocess
import shutil
import re
import time

# BoTorch Imports.
import ax
from ax import json_save, Runner
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.registry import Models
from ax.modelbridge import get_sobol
from ax.service.utils.report_utils import exp_to_df
from ax import RangeParameter, ParameterType
from ax.core.observation import ObservationFeatures
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition import qKnowledgeGradient

from ax import (
    ChoiceParameter,
    ComparisonOp,
    Experiment,
    FixedParameter,
    Metric,
    Objective,
    OptimizationConfig,
    OrderConstraint,
    OutcomeConstraint,
    ParameterType,
    RangeParameter,
    SearchSpace,
    SumConstraint,
)
from ax.modelbridge.registry import Models
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.service.managed_loop import optimize

# Append Flood_Sim directory to the current path.
flood_sim_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(flood_sim_path)

# Local imports.
from rasopt.Utilities import utils, alter_files
from rasopt.Plotting import plotting


# Model class.
class HecRasModel():

    def __init__(self, sim_ras_path_2d, bdy_fname_2d, plan_fname_2d, n_ts, map_interval,
                 sim_ras_path_1d, bdy_fname_1d, plan_fname_1d, start_datetime_1d, end_datetime_1d, output_interval):
        """
        2D Model Parameters
        :param sim_ras_path_2d: Path to simulation model directory.
        :param bdy_fname_2d: Name of boundary conditions file.
        :param plan_fname_2d: Name of plan file. Non-hdf version.
        :param n_ts: Number of hours to run the model.
        :param map_interval: Mapping interval in minutes.

        1D Model Parameters
        :param sim_ras_path_1d: Path to simulation model directory.
        :param bdy_fname_1d: Boundary conditions file name.
        :param plan_fname_1d: Plan file name.
        :param start_datetime_1d: Model start datetime, e.g., "21DEC2013,2400"
        :param end_datetime_1d: Model end datetime, e.g., "21DEC2013,2400"
        """
        # 2D model parameters.
        self.sim_ras_path_2d = sim_ras_path_2d
        self.bdy_fname_2d = bdy_fname_2d
        self.plan_fname_2d = plan_fname_2d
        self.n_ts = n_ts
        self.map_interval = map_interval
        
        # 1D model parameters.
        self.sim_ras_path_1d = sim_ras_path_1d
        self.bdy_fname_1d = bdy_fname_1d
        self.plan_fname_1d = plan_fname_1d
        self.start_datetime_1d = start_datetime_1d
        self.end_datetime_1d = end_datetime_1d
        self.output_interval = output_interval
        
        
    def model_2d_static_setup(self):
        """
        Performs 2D model static setup. 
        This includes setting the end date and the mapping interval.
        :return: None
        """
        # File paths.
        bdy_fp_2d = os.path.join(self.sim_ras_path_2d, self.bdy_fname_2d)
        plan_fp_2d = os.path.join(self.sim_ras_path_2d, self.plan_fname_2d)

        # Get the starting date time in the boundary condition file.
        start_dt = alter_files.get_start_date(bdy_fp_2d)

        # Iteration end datetime.
        end_dt = start_dt + timedelta(hours=self.n_ts)
        end_dt_str = datetime.strftime(end_dt, '%d%b%Y %H%M')

        # Edit the bXX file to update the ending datetime.
        alter_files.edit_end_date_bXX(bdy_fp_2d, end_dt_str)

        # Edit the pXX file to update the ending datetime.
        pxx_end_dt_str = datetime.strftime(end_dt, '%d%b%Y,%H%M')
        alter_files.edit_end_date_pXX(plan_fp_2d, pxx_end_dt_str)

        # Update the mapping output interval.
        alter_files.edit_mapping_interval_pXX(plan_fp_2d, self.map_interval)
        
    
    def model_1d_static_setup(self):
        """
        Performs 2D model static setup. 
        Sets start and end dates for the model.
        :return: None
        """
        # File paths.
        plan_fp_1d = os.path.join(self.sim_ras_path_1d, self.plan_fname_1d)

        # Update the start date.
        alter_files.edit_start_date_pXX(plan_fp_1d, self.start_datetime_1d)

        # Update the end date.
        alter_files.edit_end_date_pXX(plan_fp_1d, self.end_datetime_1d)

        # Update the output interval.
        alter_files.edit_output_interval(plan_fp_1d, self.output_interval)


    def satellite_comparison_setup(self, comparison_timesteps, gt_ras_path_2d, dem_fname, cell_facepoint_idx_path, facepoint_coord_path,
                                   depth_path, cell_coord_path, cell_width_X, cell_width_Y, nodata, n_seeds, radius,
                                   uncertainty_type, depth_error=0.1, max_probability=1):
        gt_plan_hdf_fp = os.path.join(gt_ras_path_2d, self.plan_fname_2d + '.hdf')
        dem_fp = os.path.join(gt_ras_path_2d, dem_fname)
        # self.gt_sat_rasters = utils.satellite_groundtruth_rasters(comparison_timesteps, gt_plan_hdf_fp,
        #                             cell_facepoint_idx_path, facepoint_coord_path, depth_path, cell_coord_path, cell_width_X,
        #                             cell_width_Y, nodata, n_seeds, radius, uncertainty_type=uncertainty_type,
        #                             depth_error=depth_error, max_probability=max_probability)
        self.gt_sat_rasters = utils.satellite_groundtruth_rasters_v2(comparison_timesteps, gt_plan_hdf_fp, dem_fp,
                                                                     cell_facepoint_idx_path, facepoint_coord_path,
                                  depth_path, cell_coord_path, cell_width_X, cell_width_Y, nodata, n_seeds, radius,
                                  uncertainty_type=uncertainty_type, depth_error=depth_error, max_probability=max_probability)

        print('GT RASTERS', self.gt_sat_rasters)


class HecRasMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters

            # Run HECRAS model.

            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    # in practice, the mean and sem will be looked up based on trial metadata
                    # but for this tutorial we will calculate them
                    "mean": (params["x1"] + 2 * params["x2"] - 7) ** 2
                    + (2 * params["x1"] + params["x2"] - 5) ** 2,
                    "sem": 0.0,
                }
            )
        return Data(df=pd.DataFrame.from_records(records))

    def is_available_while_running(self) -> bool:
        return True


# Optimization class.
class RASOpt():

    def __init__(self, gt_ras_path_2d, sim_ras_path_2d, res_output_dir_2d, prj_fname_2d, bdy_fname_2d, flow_fname_2d,
                 plan_fname_2d, geom_fname_2d, dem_fname, cell_coord_path, depth_path, cell_facepoint_idx_path,
                 facepoint_coord_path, cal_table_path, start_ts, nstarts, nevals,
                 map_interval, param_names_2d, param_bounds_2d, loss_func, locs, sim_ras_path_1d, res_output_dir_1d,
                 prj_fname_1d, bdy_fname_1d, flow_fname_1d, plan_fname_1d, geom_fname_1d, hydrograph_path,
                 param_names_1d, param_bounds_1d, breach_start, comparison_type='Sensor', depth_cut=None,
                 binary_timesteps=None, run_1D=True, gt_sat_rasters=None):

        # 2D model arguments.
        self.gt_ras_path_2d = gt_ras_path_2d
        self.sim_ras_path_2d = sim_ras_path_2d
        self.res_output_dir_2d = res_output_dir_2d
        self.prj_fname_2d = prj_fname_2d
        self.bdy_fname_2d = bdy_fname_2d
        self.flow_fname_2d = flow_fname_2d
        self.plan_fname_2d = plan_fname_2d
        self.plan_fname_2d_hdf = plan_fname_2d + '.hdf'
        self.geom_fname_2d = geom_fname_2d
        self.dem_fname = dem_fname
        self.cell_coord_path = cell_coord_path
        self.depth_path = depth_path
        self.cell_facepoint_idx_path = cell_facepoint_idx_path
        self.facepoint_coord_path = facepoint_coord_path
        self.cal_table_path = cal_table_path
        self.start_ts = start_ts
        self.nstarts = nstarts
        self.nevals = nevals
        self.map_interval = map_interval
        self.param_names_2d = param_names_2d
        self.param_bounds_2d = param_bounds_2d
        self.loss_func = loss_func
        self.locs = locs

        self.gt_plan_df_fp_2d = os.path.join(gt_ras_path_2d, plan_fname_2d + '.hdf')

        self.res_output_path_2d = os.path.join(sim_ras_path_2d, res_output_dir_2d)
        self.prj_fp_2d = os.path.join(sim_ras_path_2d, prj_fname_2d)
        self.bdy_fp_2d = os.path.join(sim_ras_path_2d, bdy_fname_2d)
        self.flow_fp_2d = os.path.join(sim_ras_path_2d, flow_fname_2d)
        self.plan_fp_2d = os.path.join(sim_ras_path_2d, plan_fname_2d)
        self.plan_hdf_fp_2d = os.path.join(sim_ras_path_2d, plan_fname_2d + '.hdf')
        self.geom_fp_2d = os.path.join(sim_ras_path_2d, geom_fname_2d)
        self.dem_fp = os.path.join(sim_ras_path_2d, dem_fname)

        # 1D model arguments.
        self.sim_ras_path_1d = sim_ras_path_1d
        self.res_output_dir_1d = res_output_dir_1d
        self.prj_fname_1d = prj_fname_1d
        self.bdy_fname_1d = bdy_fname_1d
        self.flow_fname_1d = flow_fname_1d
        self.plan_fname_1d = plan_fname_1d
        self.geom_fname_1d = geom_fname_1d
        self.hydrograph_path = hydrograph_path
        self.param_names_1d = param_names_1d
        self.param_bounds_1d = param_bounds_1d
        self.breach_start = breach_start

        self.res_output_path_1d = os.path.join(sim_ras_path_1d, res_output_dir_1d)
        self.prj_fp_1d = os.path.join(sim_ras_path_1d, prj_fname_1d)
        self.bdy_fp_1d = os.path.join(sim_ras_path_1d, bdy_fname_1d)
        self.flow_fp_1d = os.path.join(sim_ras_path_1d, flow_fname_1d)
        self.plan_fp_1d = os.path.join(sim_ras_path_1d, plan_fname_1d)
        self.plan_hdf_fp_1d = os.path.join(sim_ras_path_1d, plan_fname_1d + '.hdf')
        self.geom_fp_1d = os.path.join(sim_ras_path_1d, geom_fname_1d)

        # Index polygons and feature list. Open from saved objects if they already exist to bypass indexing.
        # polygon_index_path = os.path.join(sim_ras_path_2d, 'rtree_index')
        # feature_list_path = os.path.join(sim_ras_path_2d, 'feature_list.pkl')
        # if os.path.exists(polygon_index_path) and os.path.exists(feature_list_path):
        #     # Load feature list and polygon index if they exist.
        #     self.polygon_index = index.Rtree('path/to/gridIndex')
        #     with open(feature_list_path, "rb") as f:
        #         self.feature_list = pickle.load(f)
        # else:
        #     self.polygon_index, self.feature_list = utils.index_polygons(self.plan_hdf_fp_2d)
        #     # Save index and feature list.
        #     with open(feature_list_path, "wb") as f:
        #         self.feature_list = pickle.dump(self.feature_list, f)

        self.polygon_index, self.feature_list = utils.index_polygons(self.plan_hdf_fp_2d)

        # Create single parameter bounds variable.
        self.param_bounds = self.param_bounds_2d + self.param_bounds_1d
        self.norm_bounds = utils.normalize_parameters(self.param_bounds)

        # Create a parameter types variable.
        self.param_names = self.param_names_2d + self.param_names_1d
        self.param_types = [i.name[-1] for i in self.param_bounds]

        # Loss value counter for saving optimal plan files.
        # Case where optimal loss is smallest value.
        if self.loss_func == 'MSE':
            self.best_loss = 1000
        # Case where optimal loss is largest value.
        elif self.loss_func == 'NSE':
            self.best_loss = -1000

        # Comparison Type.
        self.comparison_type = comparison_type
        self.depth_cut = depth_cut
        self.binary_timesteps = binary_timesteps
        self.gt_sat_rasters = gt_sat_rasters

        # Run 1D model.
        self.run_1D = run_1D

        # Check if output paths are created, if not, create them.
        if not os.path.isdir(self.res_output_path_2d):
            os.mkdir(self.res_output_path_2d)
        if not os.path.isdir(self.res_output_path_1d) and self.run_1D is True:
            os.mkdir(self.res_output_path_1d)


    def optimize_parameters(self):
        # Run BayesOpt procedure. Make sure the convergence plot path is specified correctly.
        res = self._optimizer()

        return res


    def _optimizer(self):

        best_parameters, values, experiment, model = optimize(
            parameters=[{"name": p.name, "type": "range", "bounds": [p.lower, p.upper]} for p in self.param_bounds],
            experiment_name="hec_ras_opt",
            objective_name="_run_with_loss",
            evaluation_function=self._run_with_loss,
            minimize=True,  # Optional, defaults to False.
            total_trials=self.nevals,  # Optional.
        )

        return experiment


    def _optimizer_dev(self):

        # Create search space.
        search_space = ax.SearchSpace(self.param_bounds)

        # Create optimization config.
        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=Hartmann6Metric(name="hartmann6", param_names=param_names),
                minimize=True,
            )
        )

        # Define a runner.
        class MyRunner(Runner):
            def run(self, trial):
                trial_metadata = {"name": str(trial.index)}
                return trial_metadata

        # Create an experiment.
        exp = ax.core.Experiment(
            name="hec_ras_opt",
            search_space=search_space,
            optimization_config=optimization_config,
            runner=MyRunner(),
        )

        # Perform optimization.
        NUM_SOBOL_TRIALS = self.nstarts
        NUM_BOTORCH_TRIALS = self.nevals

        # Sobol sampling.
        sobol = Models.SOBOL(search_space=exp.search_space)

        for i in range(NUM_SOBOL_TRIALS):
            # Produce a GeneratorRun from the model, which contains proposed arm(s) and other metadata
            generator_run = sobol.gen(n=1)
            # Add generator run to a trial to make it part of the experiment and evaluate arm(s) in it
            trial = exp.new_trial(generator_run=generator_run)
            # Start trial run to evaluate arm(s) in the trial
            trial.run()
            # Mark trial as completed to record when a trial run is completed
            # and enable fetching of data for metrics on the experiment
            # (by default, trials must be completed before metrics can fetch their data,
            # unless a metric is explicitly configured otherwise)
            trial.mark_completed()

        # BoTorch trials.
        for i in range(NUM_BOTORCH_TRIALS):
            print(
                f"Running BO trial {i + NUM_SOBOL_TRIALS + 1}/{NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS}..."
            )
            # Reinitialize GP+EI model at each step with updated data.
            gpei = Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data())
            generator_run = gpei.gen(n=1)
            trial = exp.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()

        # Extract trial data.
        trial_data = exp.fetch_trials_data([NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS - 1])
        result_df = trial_data.df

        return result_df


    def _optimizer_OLD(self):
        # # SKOPT BayesOpt
        # res = gp_minimize(self._run_with_loss,  # the function to minimize
        #                   # self.param_bounds,  # the bounds on each dimension of x
        #                   self.norm_bounds, # Use normalized parameter values.
        #                   acq_func="EI",  # the acquisition function
        #                   n_calls=self.nevals,  # the number of evaluations of f
        #                   n_initial_points=self.nstarts,  # the number of random initialization points
        #                   random_state=None,
        #                   verbose=True)  # the random seed

        # BoTorch BayesOpt
        # TODO: Handle parameters correctly.
        search_space_1 = ax.SearchSpace(self.param_bounds)
        experiment_hec = ax.core.Experiment(
            name='hec_ras_opt',
            search_space=search_space_1,
            objective_name='_run_with_loss',
            minimize=True,  # Specify whether you want to minimize or maximize the objective
        )

        # TODO: Make sure that the loss function optimal at its maximum since this is what BoTorch searches for.

        # Initial samples from search space.
        sobol = get_sobol(experiment_hec.search_space)
        experiment_hec.new_batch_trial(generator_run=sobol.gen(self.nstarts))

        max_f = []
        argmax_x = []
        for i in range(self.nevals):
            # Simple BoTorch Setup.
            model_bridge_with_GPEI = Models.BOTORCH_MODULAR(
                experiment=experiment_hec,
                surrogate=Surrogate(SingleTaskGP),
                data=experiment_hec.eval(),
                botorch_acqf_class=qNoisyExpectedImprovement,
                # botorch_acqf_class=qKnowledgeGradient,
            )

            # Suggest next point.
            generator_run = model_bridge_with_GPEI.gen(n=1)
            experiment_hec.new_trial(generator_run)

            results_df = exp_to_df(experiment_hec)
            result_sort_df = results_df.sort_values(by=['_run_with_loss'], ascending=False)
            result_sort_df.reset_index(inplace=True)
            print(result_sort_df.head())
            self.x_names = [p.name for p in self.param_bounds]
            func_param_select = ['_run_with_loss'] + self.x_names
            best_vals = result_sort_df.loc[0, func_param_select]
            print('CURRENT RESULTS', result_sort_df.loc[:, func_param_select])
            max_f.append(best_vals[0])
            max_param = []
            for j, _ in enumerate(self.x_names):
                max_param.append(best_vals[j+1])
            argmax_x.append(tuple(max_param))

            print('MAX F', max_f)
            print('ARGMAX X', argmax_x)
            print('RUN', i)

        # # Predict on fine grid and choose the maximum point.
        # fg, cov, X = self._predict_on_grid(model_bridge_with_GPEI, '_run_with_loss')
        # max_x_idx = np.argmax(fg)
        # x0 = np.array([X[max_x_idx,:]]) # TODO: This may not behave the same for 1-D vs multi-D
        # x_best = self._maximize_gp(x0, model_bridge_with_GPEI, '_run_with_loss')

        return experiment_hec


    def _predict_on_grid(self, model, model_name):
        """
        Predict on a grid of observations.
        :param model: Model used for prediction.
        :param model_name: Name of the model for extracting the mean and covariance.
        :return: Mean function and covariance at prediction points. Array of prediction points.
        """

        ngrid = 100

        param_meshgrid = []
        for i, param in enumerate(self.param_bounds):
            param_array = np.linspace(param.lower, param.upper, ngrid)
            param_meshgrid.append(param_array)

        p_vals = np.zeros((ngrid ** len(self.param_bounds), len(self.param_bounds)))
        mg = np.meshgrid(*param_meshgrid)
        for i, p_ar in enumerate(mg):
            p_vals[:, i] = np.ravel(p_ar)

        # Make the observation features.
        obs_feats = []
        for row in range(p_vals.shape[0]):
            X = p_vals[row, :]
            param_dict = {self.x_names[i]: X[i] for i in range(len(X))}
            obs_feats.append(ObservationFeatures(parameters=param_dict))

        # Predict on the grid.
        fg, cov = model.predict(obs_feats)
        fg = fg[model_name]
        cov = cov[model_name][model_name]

        return fg, cov, p_vals


    def _maximize_gp(self, x0, model, model_name):
        """
        Maximizes the GP mean function to find the parameter that produces the maximum.
        :param x0: Initial guess.
        :param model: GP model.
        :param model_name: Name of the optimized parameter.
        :return: X that produces maximum function value (i.e., argmax_x f(x))
        """

        def model_predict(X):
            """
            Predict on the model.
            :param X: Parameter values as numpy array in order used to train the model.
            :return: Mean function value at X.
            """

            # Create observation features.
            # param_dict = {f'x{i + 1}': X[i] for i in range(len(X))}
            param_dict = {self.x_names[i]: X[i] for i in range(len(X))}
            obs_feats = [ObservationFeatures(parameters=param_dict)]

            f_mean, f_cov = model.predict(obs_feats)

            return -f_mean[model_name][0] # Negative because we have to use a minimizer to find the argmax(f)

        # Scipy only has a minimizer, so we're minimizing -f, which is equivalent to maximizing f.
        res = minimize(model_predict, x0, bounds=Bounds(0, 1), method='BFGS')

        return res.x


    def _run_hec(self, params):
        # Update the file parameters.
        self._update_parameters(params)

        # Pause for a moment.
        time.sleep(2)

        # Run the 1D model.
        if self.run_1D is True:
            utils.controller_hec_run(self.prj_fp_1d, "RAS41.HECRASController")

        # Extract hydrograph from 1D model and add as input into 2D model.
        if self.run_1D is True:
            self._update_hydrograph()

        # Run the 2D model.
        utils.controller_hec_run(self.prj_fp_2d, "RAS507.HECRASController")


    def _run_with_loss(self, params):
        """
        Routing method for handling all processing that needs to be done dynamically during optimzation.
        :return:
        """

        ax_param_names = [p.name for p in self.param_bounds]
        params = [params[ax_param_names[i]] for i in range(len(params))]
        print('PARAMS', params)

        # Denorm parameters.
        denorm_params = []
        for i, pval in enumerate(params):
            denorm_params.append(utils.denorm_parameter(params[i], self.param_bounds[i].lower, self.param_bounds[i].upper))
        # params = denorm_params

        self._run_hec(params)

        # Compute the loss function.
        loss = self._compute_loss()

        # # Save the plan files from the models if the loss is better than the last time.
        # self._save_best_parameters(loss)

        # # Plot the point time series.
        # plotting.plot_point_time_series(self.gt_ras_path_2d, self.sim_ras_path_2d, self.locs, self.cell_coord_path,
        #                                 self.depth_path, params, output_path=self.res_output_path_2d)

        return loss


    def _update_parameters(self, params):
        """
        Distributes parameters to the correct update method.
        :param params: Geometry and breach parameters.
        :return: None
        """

        breach_params = {}
        geom_params = []
        for i in range(len(params)):
            if self.param_types[i] == 'g':
                geom_params.append(params[i])
            elif self.param_types[i] == 'b':
                name = self.param_names[i]
                breach_params[name] = params[i]

        if geom_params != []:
            self._update_geom_params(geom_params)

        if breach_params:
            self._update_breach_params(breach_params)


    def _update_breach_params(self, params):
        """
        Updates breach parameters.
        :param params: Dictionary of parameter names and values.
        :return: None
        """
        for pname, val in params.items():
            if pname == 'weir_coefficient':
                alter_files.edit_weir_coef(self.plan_fp_1d, val)
            elif pname == 'bottom_width':
                alter_files.edit_final_bottom_width(self.plan_fp_1d, val)
            elif pname == 'formation_time':
                alter_files.edit_formation_time(self.plan_fp_1d, val)


    def _update_geom_params(self, params):
        """
        Updates parameter file with the new parameter set.
        :param params: Parameter values to update in order as specified in inputs section.
        :return: Updates the parameter file.
        """
        # Get parameter names from file.
        file_params = utils.get_set_dataset(self.geom_fp_2d, self.cal_table_path)
        file_pnames = [i[0].decode('ascii') for i in file_params]

        # Get the parameter values from the file.
        file_pvals = utils.get_set_dataset(self.geom_fp_2d, self.cal_table_path)

        # Set the new parameter values at the index of the correct parameter name.
        for i in range(len(self.param_names)):
            for j in range(len(file_pnames)):
                if self.param_names[i] == file_pnames[j]:
                    file_pvals[j][1] = params[i]

        # Set the parameters that aren't being optimized to the default values.
        for i in range(len(file_pnames)):
            if file_pnames[i] not in self.param_names:
                file_pvals[i][1] = np.nan

        # Update the parameter file with new parameter set.
        correct_set = utils.get_set_dataset(self.geom_fp_2d, self.cal_table_path, get_ds=False, new_ds=file_pvals)
        assert correct_set, 'Parameters not set correctly.'


    def _update_hydrograph(self):
        """
        Extracts the hydrograph from the 1D model and updates it in the 2D model.
        :return: None.
        """
        # Extract hydrograph from 1D model.
        hydrograph, times = utils.extract_hec_breach_hydrograph(self.prj_fp_1d)

        # Clip the hydrograph to the 2D model date range.
        breach_start_dt = datetime.strptime(self.breach_start, '%d%b%Y %H%M')
        breach_idx = times.index(breach_start_dt)
        hydrograph = hydrograph[breach_idx:]

        # Update hydrograph in 2D model.
        alter_files.update_input_hydrograph(self.flow_fp_2d, hydrograph)


    def _compute_loss(self):

        if self.comparison_type == 'Sensor':
            # Locations.
            sensor_lats = [i[0] for i in self.locs]
            sensor_lons = [i[1] for i in self.locs]

            # Get ground truth and simulation depths.
            sim_depths = utils.depth_at_locations(self.plan_hdf_fp_2d, self.dem_fp, sensor_lats, sensor_lons,
                                                  self.polygon_index, self.feature_list, depth_cutoff=self.depth_cut)
            gt_depths = utils.depth_at_locations(self.gt_plan_df_fp_2d, self.dem_fp, sensor_lats, sensor_lons,
                                                 self.polygon_index, self.feature_list, depth_cutoff=self.depth_cut)

            # Compute loss for each of the locations using type I or type II comparison.
            loss_vals = []
            for loc, gt_depth_ts in gt_depths.items():
                # If GT depth time series is empty, skip the location.
                if len(gt_depth_ts) == 0:
                    continue

                # Simulation depth time series.
                sim_depth_ts = sim_depths[loc]

                # Start clip index to signal when the sensor has started to collect data.
                # -1 to account for the data being 0-indexed.
                start_clip_idx = int(self.start_ts * 60 / self.map_interval - 1)
                gt_depth_ts = gt_depth_ts[start_clip_idx:]
                sim_depth_ts = sim_depth_ts[start_clip_idx:]

                # If both arrays are all zeros, skip.
                if not np.any(sim_depth_ts) or not np.any(gt_depth_ts):
                    continue

                # Make arrays the same length.
                end_clip_idx = min([len(gt_depth_ts), len(sim_depth_ts)])
                gt_depth_ts = gt_depth_ts[:end_clip_idx]
                sim_depth_ts = sim_depth_ts[:end_clip_idx]

                # Compute the desired loss function.
                if self.loss_func == 'MSE':
                    loss_vals.append(utils.MSE(sim_depth_ts, gt_depth_ts))

                elif self.loss_func == 'NNSE':
                    normNSE = utils.NNSE(sim_depth_ts, gt_depth_ts)
                    loss_vals.append(normNSE)

                elif self.loss_func == 'RMSE':
                    loss_vals.append(utils.RMSE(sim_depth_ts, gt_depth_ts))

            # Compute mean of loss function values.
            if loss_vals != []:
                loss = np.mean(np.array(loss_vals))

                # Negate the loss for maximization.
                loss = loss
            else:
                raise Exception('No depth time series found.')
                loss = None

        return loss


    def _compute_loss_OLD(self):
        """
        Computes the loss function on the simulated and observed depths.
        :return: Loss function value.
        """
        # Extract the depth data from the simulation.
        sim_dep_df = utils.extract_depths(self.plan_hdf_fp_2d, self.depth_path, self.cell_coord_path)

        # Extract the depth data from the measured depths.
        meas_dep_df = utils.extract_depths(self.gt_plan_df_fp_2d, self.depth_path, self.cell_coord_path)


        if self.comparison_type == 'Sensor':
            # Compute loss for each of the locations using type I or type II comparison.
            loss_vals = []
            total_sim = []
            total_meas = []
            count = 1
            # fig, ax = plt.subplots()
            for loc in self.locs:
                print(count)
                count += 1
                # Select closest row from simulated data set.
                sim_row = utils.closest_row(sim_dep_df, loc)
                total_sim.extend(sim_row)

                # Select closest row from measured data set.
                meas_row = utils.closest_row(meas_dep_df, loc)
                meas_row = meas_row[:len(sim_row)]
                total_meas.extend(meas_row)

                # Start clip index to signal when the sensor has started to collect data.
                # -1 to account for the data being 0-indexed.
                start_clip_idx = int(self.start_ts * 60 / self.map_interval - 1)
                sim_row = sim_row[start_clip_idx:]
                meas_row = meas_row[start_clip_idx:]

                # ax.scatter(meas_row, sim_row)


                # Check if at least one array is all zero.
                if not np.any(sim_row) or not np.any(meas_row):
                    if self.loss_func == 'MSE':
                        print('Zero array present, setting MSE to 1 at', loc)
                        loss_vals.append(1)
                    if self.loss_func == 'NNSE':
                        print('Zero array present, setting NNSE to 1 at', loc)
                        loss_vals.append(1)
                    continue

                # Compute the desired loss function.
                if self.loss_func == 'MSE':
                    # Add some noise to the measured data.
                    # rng = np.random.default_rng(2021)
                    # meas_row = meas_row + rng.normal(0, 0.02, size=meas_row.shape)

                    # Make sure that the sim row and meas row have the same length.
                    loss_vals.append(utils.MSE(sim_row, meas_row))

                elif self.loss_func == 'NNSE':
                    # Use the negative of the NNSE so that -1 is the optimal value.
                    normNSE = utils.NNSE(sim_row, meas_row)
                    loss_vals.append(-normNSE)

                elif self.loss_func == 'RMSE':
                    # Add some noise to the measured data.
                    # rng = np.random.default_rng(2022)
                    # meas_row = meas_row + rng.normal(0, 0.02, size=meas_row.shape)

                    loss_vals.append(utils.RMSE(sim_row, meas_row))

            plt.show()
            # Compute mean of loss function values.
            if loss_vals != []:
                loss = np.mean(np.array(loss_vals))
            else:
                loss = None

            # # Compute the loss on all non-zero sensor points.
            # c = np.zeros((len(total_meas), 2))
            # c[:, 0] = total_meas
            # c[:, 1] = total_sim
            # c = c[np.all(c != 0, axis=1)]
            # loss = utils.MSE(c[:,1], c[:,0])
            loss = -loss # Since we're maximizing.

            # # Compute log(2) of the values.
            # SMALL = 0.0001 # Add small value so there is no log(0).
            # loss = np.log(loss+SMALL) / np.log(1.1)

        elif self.comparison_type == 'Sensor_Max':
            # Simulated values.
            sim_depths = sim_dep_df.iloc[self.locs, 2:].to_numpy()

            # Measured values.
            gt_depths = meas_dep_df.iloc[self.locs, 2:].to_numpy()
            gt_depths = gt_depths[:, :sim_depths.shape[1]]

            # Compute loss function.
            if self.loss_func == 'RMSE':
                loss = np.mean(np.sqrt(np.sum(np.square(gt_depths - sim_depths), axis=1) / sim_depths.shape[1]))
                loss = -loss
            # fig, ax = plt.subplots()
            # ax.scatter(gt_depths[:, :], sim_depths[:, :])
            # plt.show()

        elif self.comparison_type == 'Binary':
            # loss = utils.satellite_success_metric(self.gt_sat_rasters, self.binary_timesteps, self.plan_fp_2d,
            #                                self.depth_path, self.cell_coord_path, utils.inun_fit,
            #                                depth_cut=self.depth_cut)

            loss = utils.compute_raster_loss(self.gt_sat_rasters, self.binary_timesteps, self.plan_hdf_fp_2d, self.dem_fp,
                                             self.depth_path, self.cell_coord_path, self.cell_facepoint_idx_path,
                                             self.facepoint_coord_path, utils.inun_fit, depth_cut = self.depth_cut)
            loss = loss

        return loss


    def _save_best_parameters(self, new_loss):
        """
        Saves the plan files from the 1D and 2D models if the loss is the best.
        :return: None
        """
        # When optimal loss is the smallest possible.
        if self.loss_func == 'MSE':
            if new_loss < self.best_loss:
                # Copy over 2D plan file.
                source_2d = os.path.join(self.sim_ras_path_2d, self.plan_fname_2d_hdf)
                dest_2d = os.path.join(self.res_output_path_2d, self.plan_fname_2d_hdf)
                shutil.copy(source_2d, dest_2d)

                # Copy over 1D plan file.
                source_1d = os.path.join(self.sim_ras_path_1d, self.plan_fname_1d)
                dest_1d = os.path.join(self.res_output_path_1d, self.plan_fname_1d)
                shutil.copy(source_1d, dest_1d)

                self.best_loss = new_loss

        # When optimal loss is the largest possible.
        if self.loss_func == 'NSE':
            if new_loss < self.best_loss:
                # Copy over 2D plan file.
                source_2d = os.path.join(self.sim_ras_path_2d, self.plan_fname_2d)
                dest_2d = os.path.join(self.res_output_path_2d, self.plan_fname_2d)
                shutil.copy(source_2d, dest_2d)

                # Copy over 1D plan file.
                source_1d = os.path.join(self.sim_ras_path_1d, self.plan_fname_1d)
                dest_1d = os.path.join(self.res_output_path_1d, self.plan_fname_1d)
                shutil.copy(source_1d, dest_1d)

                self.best_loss = new_loss


# Output class.
class Output():

    def __init__(self, gt_ras_path_2d, sim_ras_path_2d, res_output_dir_2d, result=None):

        self.gt_ras_path_2d = gt_ras_path_2d
        self.sim_ras_path_2d = sim_ras_path_2d
        self.res_output_path_2d = res_output_dir_2d
        self.result = result


    def write_results_to_text(self):
        """
        Writes the result object to a text file in the output file location.
        :return: None
        """
        with open(os.path.join(self.res_output_path_2d, 'Result.txt'), 'w') as f:
            f.write(str(self.result))


    def save_botorch_experiment(self, experiment, suffix):
        """
        Saves the experiment from BoTorch.
        :return: None.
        """
        save_path = os.path.join(self.res_output_path_2d, f'BT_exp_{suffix}.json')
        json_save.save_experiment(experiment, save_path)


    def plot_hydrograph_time_series(self, locations, cell_coord_path, depth_path, plot_ID):
        """
        Plots hydrograph time series comparison between the ground truth and simulation.
        :param locations: Locations in the domain to plot the time series. (lat, lon)
        :param cell_coord_path: Path to cell coordinates in the 2D plan file.
        :param depth_path: Path to cell depths in the 2D plan file.
        :param plot_ID: Unique ID to give the plots.
        :return: None
        """
        # TODO: Add the parameter file name on the end.
        plotting.plot_point_time_series(self.gt_ras_path_2d, self.sim_ras_path_2d, locations, cell_coord_path,
                                        depth_path, plot_ID, output_path=None)



