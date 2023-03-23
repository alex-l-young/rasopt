% Inundation statistics.
% 
% Computation of 
%   - Sensitivity
%   - Type I Error
%   - Type II Error
%   - Fit

close all
clear

% File paths.
gt_plan_fp = "C:\Users\ayoun\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Roughness_Output\Secchia_Panaro.p23_GT.hdf";
sim_plan_fp = "C:\Users\ayoun\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Roughness_Output\Secchia_Panaro.p23_c1smax.hdf";

% HDF Paths.
% Path to cell coordinates.
cell_coord_path = '/Geometry/2D Flow Areas/Secchia_Panaro/Cells Center Coordinate';

% Path to water depths.
depth_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Secchia_Panaro/Depth';

% Load in the depths.
gt_depths = h5read(gt_plan_fp, depth_path);
sim_depths = h5read(sim_plan_fp, depth_path);

% Cell locations.
cell_coords = h5read(gt_plan_fp, cell_coord_path)';
lats = zeros(ncell,1);
lons = zeros(ncell,1);
for i = 1:ncell
    [lat, lon] = utm2deg(cell_coords(i,1),cell_coords(i,2),'32 N');
    lats(i) = lat;
    lons(i) = lon;
end


