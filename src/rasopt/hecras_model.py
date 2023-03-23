# ======================================================
# HEC-RAS Model Class
# Contains methods for automatically running HEC-RAS.
# Alex Young
# December 2022
# ======================================================

# Library imports.



# ===============================
# HEC-RAS Base Class.
class HecRasModel():

    def run(self, proj_path, controller_version):
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


    def update_parameters(self, params):
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



# HEC-RAS 1D Class.
class HecRasModel1D(HecRasModel):

    def __init__(self, sim_ras_path_1d, bdy_fname_1d, plan_fname_1d, start_datetime_1d, end_datetime_1d, output_interval):
        """
        1D Model Parameters
        :param sim_ras_path_1d: Path to simulation model directory.
        :param bdy_fname_1d: Boundary conditions file name.
        :param plan_fname_1d: Plan file name.
        :param start_datetime_1d: Model start datetime, e.g., "21DEC2013,2400"
        :param end_datetime_1d: Model end datetime, e.g., "21DEC2013,2400"
        """

        # 1D model parameters.
        self.sim_ras_path_1d = sim_ras_path_1d
        self.bdy_fname_1d = bdy_fname_1d
        self.plan_fname_1d = plan_fname_1d
        self.start_datetime_1d = start_datetime_1d
        self.end_datetime_1d = end_datetime_1d
        self.output_interval = output_interval

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


class HecRasModel2D(HecRasModel):

    def __init__(self, sim_ras_path_2d, bdy_fname_2d, plan_fname_2d, n_ts, map_interval):
        """
        2D Model Parameters
        :param sim_ras_path_2d: Path to simulation model directory.
        :param bdy_fname_2d: Name of boundary conditions file.
        :param plan_fname_2d: Name of plan file. Non-hdf version.
        :param n_ts: Number of hours to run the model.
        :param map_interval: Mapping interval in minutes.
        """
        # 2D model parameters.
        self.sim_ras_path_2d = sim_ras_path_2d
        self.bdy_fname_2d = bdy_fname_2d
        self.plan_fname_2d = plan_fname_2d
        self.n_ts = n_ts
        self.map_interval = map_interval


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
