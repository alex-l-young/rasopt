# ====================================================================================================
# Python model wrapper for HEC-RAS.
# Alex Young
# 2024
# ====================================================================================================

# Library imports.
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import yaml

# Rasopt imports.
from rasopt.Utilities import utils, alter_files


# Model class.
class HecRasModel():

    def __init__(self, path_to_config):
        self.path_to_config = path_to_config

        # Parse configuration.
        self._parse_config()

        # Perform model setup.
        self._model_setup()


    def _parse_config(self):
        """
        Parse the configuration file.
        :param model_dir: Path to simulation model directory.
        :param boundary_file: Name of boundary conditions file.
        :param plan_file: Name of plan file. Non-hdf version.
        :param project_file: Name of the project file.
        :param geometry_file: Name of the geometry file.
        :param n_ts: Number of hours to run the model.
        :param map_interval: Mapping interval in minutes.
        """
        # Load in configuration file.
        with open(self.path_to_config, "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        # 2D model parameters.
        self.n_ts = cfg['run_parameters']['n_ts']
        self.map_interval = cfg['run_parameters']['map_interval']

        # Construct file paths.
        self.model_dir = Path(cfg['model_files']['model_dir'])
        self.boundary_filepath = self.model_dir / cfg['model_files']['boundary_file']
        self.plan_filepath = self.model_dir / cfg['model_files']['plan_file']
        self.plan_hdf_filepath = self.model_dir / cfg['model_files']['plan_hdf_file']
        self.project_filepath = self.model_dir / cfg['model_files']['project_file']
        self.geometry_filepath = self.model_dir / cfg['model_files']['geometry_file']

        # Paths within HDF files.
        self.cell_coord_path = cfg['hdf_paths']['cell_coord_path']
        self.depth_path = cfg['hdf_paths']['depth_path']
        self.cal_table_path = cfg['hdf_paths']['cal_table_path']
        self.cell_facepoint_idx_path = cfg['hdf_paths']['cell_facepoint_idx_path']
        self.facepoint_coord_path = cfg['hdf_paths']['facepoint_coord_path']


    def _model_setup(self):
        """
        Performs 2D model static setup.
        This includes setting the end date and the mapping interval.
        :return: None
        """
        # # File paths.
        # boundary_filepath = os.path.join(self.model_dir, self.boundary_file)
        # plan_filepath = os.path.join(self.model_dir, self.plan_file)

        # Get the starting date time in the boundary condition file.
        start_dt = alter_files.get_start_date(self.boundary_filepath)

        # Iteration end datetime.
        end_dt = start_dt + timedelta(hours=self.n_ts)
        end_dt_str = datetime.strftime(end_dt, '%d%b%Y %H%M')

        # Edit the boundary file to update the ending datetime.
        alter_files.edit_end_date_bXX(self.boundary_filepath, end_dt_str)

        # Edit the plan file to update the ending datetime.
        plan_end_dt_str = datetime.strftime(end_dt, '%d%b%Y,%H%M')
        alter_files.edit_end_date_pXX(self.plan_filepath, plan_end_dt_str)

        # Update the mapping output interval.
        alter_files.edit_mapping_interval_pXX(self.plan_filepath, self.map_interval)


    def run(self):
        """
        Run the model.
        :return: No return.
        """
        utils.controller_hec_run(self.project_filepath, "RAS507.HECRASController")


    def update_manning_n(self, param_names, param_values):
        """
        Update the Manning's n values in the model.
        :param param_names: Names of the parameters that correspond to names in Manning's n land cover file. [list]
        :param param_values: Corresponding values of the parameter names. [list]
        :return: No return.
        """
        # Get parameter names from file.
        file_params = utils.get_set_dataset(self.geometry_filepath, self.cal_table_path)
        file_pnames = [i[0].decode('ascii') for i in file_params]

        # Get the parameter values from the file.
        file_pvals = utils.get_set_dataset(self.geometry_filepath, self.cal_table_path)

        # Set the new parameter values at the index of the correct parameter name.
        for i in range(len(param_names)):
            for j in range(len(file_pnames)):
                if param_names[i] == file_pnames[j]:
                    file_pvals[j][1] = param_values[i]

        # Set the parameters that aren't being optimized to the default values.
        for i in range(len(file_pnames)):
            if file_pnames[i] not in param_names:
                file_pvals[i][1] = np.nan

        # Update the parameter file with new parameter set.
        correct_set = utils.get_set_dataset(self.geometry_filepath, self.cal_table_path, get_ds=False,
                                            new_ds=file_pvals)
        assert correct_set, 'Parameters not set correctly.'