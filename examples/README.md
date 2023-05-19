# Optimizaton Instructions

Optimization requirements:
- Configuration file (config_1D2D.yml is provided as a template).
- 2D HEC-RAS instantiation.
- 1D HEC-RAS instantiaton if running the 1D portion of the optimization.

## Configuration file

The configuration file contains the configurations that are required by the optimization script. These are commented 
in the file "config_1D2D.yml" and will need alteration if you plan to change the model directories or optimization 
settings. The most common variables that need to be changed are listed below.

#### optim

- nstarts: Number of model initializations before the Bayesian optimization algorithm is run.
- nevals: Number of Bayesian optimization iterations (evaluations).
- n_ts: Number of simulation time steps to run.
- map_interval: HEC-RAS mapping interval in minutes. Output file will contain data with this interval.

#### dir_2d

- gt_ras_path_2d: Path to the ground truth model directory.
- sim_ras_path_2d: Path to the simulation model directory.

#### model_2d  

- sensor_locations: Path to csv file containing the sensor latitude and longitudes.

#### param_2d

- param_names_2d: Parameter names to be optimized by Bayesian optimization procedure. They must match the names
found in the model.
  
- range_param_names_2d: The way the parameters will be named in the output files. Put "_g" after Manning's n parameters.
- param_lower_2d: Lower bounds of the parameters' search space specified in param_names_2d.
- param_upper_2d: Upper bounds of the parameters' search space specified in param_names_2d.

## 2D HEC-RAS instantiation

The 2D HEC-RAS model will be evaluated multiple times during the calibration procedure as specified by the number of 
initializations and evaluations in the "optim" secion of the configuration file. 
For this reason, the model must be ready to evaluate and all bugs should be worked out in the HEC-RAS GUI beforehand. 

Before running the optimization procedure.
The HEC-RAS model must be run with the desired parameterization (See HECRAS_Geometry_Instructions.docx). 
For example, if Manning's n for four roughness classes are to be calibrated, 
the 2D model must first be run in the GUI with the correct land cover map. 
This ensures that the optimization procedure will be able to access the parameters. 
If this is not done (e.g., an old model version is used with a three-parameter land cover map), 
the optimization algorithm will attempt to optimize four parameters when they do not really exist, 
thus invalidating the procedure. In effect, the optimization algorithm can only work with the current 
state of the model setup and will not perform model configuration beyond parameter value alterations.

## 1D HEC-RAS instantiation

In general, the 1D model doesn't need to be run in most cases as long as the input hydrograph for the 2D model is already in place. The setting the 1D model run flag "flag_1d: run_1d" to True will run the 1D model and then insert the breach hydrograph into the 2D model. This procedure has only been tested for a single levee breach, so it may not work when multiple levee breaches are modeled.

## Optimizing Parameters

To run the HEC-RAS optimization procedure, use the jupyter notebook calibration_example.ipynb