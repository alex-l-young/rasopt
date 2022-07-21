% Dynamic CDF plotting.

close all
clear

% Specify the file names.
data_dir = '/Users/alexyoung/Library/CloudStorage/Box-Box/Research/Flood_Sim_Materials/BayesOpt_Paper/Data/Mannings_Sensitivity/';
gt_data_dir = '/Users/alexyoung/Library/CloudStorage/Box-Box/Research/Flood_Sim_Materials/BayesOpt_Paper/Data/Roughness_Output/';
gt_plan_name = "Secchia_Panaro.p23_GT.hdf";
sim_plan_names = [
    "Secchia_Panaro.p23_veg1_0.03.hdf",
    "Secchia_Panaro.p23_veg1_0.05.hdf",
    "Secchia_Panaro.p23_veg1_0.07.hdf",
    "Secchia_Panaro.p23_veg1_0.09.hdf",
    "Secchia_Panaro.p23_veg1_0.11.hdf",
%     "Secchia_Panaro.p23_c1s5.hdf",
%     "Secchia_Panaro.p23_c4s2.hdf",
%     "Secchia_Panaro.p23_c4s1.hdf",
    ];
sim_ids = ["$n=0.03$", 
    "$n=0.05$",
    "$n=0.07$",
    "$n=0.09$",
    "$n=0.11$",
%     "$N_{class}=1,N_{sensor}=5$",
%     "$N_{class}=4,N_{sensor}=2$",
%     "$N_{class}=4,N_{sensor}=1$",
    ];

% Create file paths.
gt_plan_fp = gt_data_dir + gt_plan_name;
sim_plan_fps = string();
for i = 1:length(sim_plan_names)
    sim_plan_fps(i) = data_dir + sim_plan_names(i);
end

% HDF Paths.
% Path to cell coordinates.
cell_coord_path = '/Geometry/2D Flow Areas/Secchia_Panaro/Cells Center Coordinate';

% Path to water depths.
depth_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Secchia_Panaro/Depth';

% Time step in hours.
ts_len = 1/6;

%% Process the ground truth data.
gt_depths = h5read(gt_plan_fp, depth_path);
[ncell, Nt] = size(gt_depths);
time = (0:Nt-1) .* ts_len;

% Only include first N time steps.
Nts = 849;
gt_depths = gt_depths(:,1:Nts);

% Cell locations.
cell_coords = h5read(gt_plan_fp, cell_coord_path)';
lats = zeros(ncell,1);
lons = zeros(ncell,1);
for i = 1:ncell
    [lat, lon] = utm2deg(cell_coords(i,1),cell_coords(i,2),'32 N');
    lats(i) = lat;
    lons(i) = lon;
end

% Remove cells that do not hit critical threshold based on GT values.
dcrit = 0.1;
gt_depths_dc = gt_depths(max(gt_depths,[],2) >= dcrit,:);
lats = lats(max(gt_depths,[],2) >= dcrit);
lons = lons(max(gt_depths,[],2) >= dcrit);

% Ground truth time to critical depth.
[~,gt_tc] = max(gt_depths_dc >= dcrit,[],2);
[tc_rows,~] = size(gt_tc);

%% Compute the time discrepancy for each file.
delta_tc_ar = zeros(tc_rows, length(sim_plan_fps));
for i = 1:length(sim_plan_fps)
    % Load in the file.
    plan_fp = sim_plan_fps(i);
    sim_depths = h5read(plan_fp, depth_path);
    
    % Only include first N time steps.
    sim_depths = sim_depths(:,1:Nts);
    
    % Remove cells that do not hit critical threshold based on GT values.
    sim_depths_dc = sim_depths(max(gt_depths,[],2) >= dcrit,:);
    
    % Time to critical depth.
    [~,sim_tc] = max(sim_depths_dc >= dcrit,[],2);
    
    % Compute the difference in critical depth timing. 
    delta_tc = (gt_tc - sim_tc) * ts_len;
    delta_tc_ar(:,i) = delta_tc;
    
end

%% Histogram of time discrepancy.
bins = 51;
figure()
hold on
for i = 1:length(sim_plan_fps)
    histogram(delta_tc_ar(:,i), bins, 'DisplayName', sim_ids(i))%, 'Normalization','probability')
end
hold off
xlabel('\delta_{tc} (hr)')
ylabel('Number of Cells')
set(gca, 'FontSize', 16)
legend()

%% CDF of time discrepancy.

figure()
hold on
for i = 1:length(sim_plan_fps)
    pdf_counts = zeros(size(time));
    all_times = [-fliplr(time) time];
    delta_tc_vec = delta_tc_ar(:,i);
    for ti = 1:Nt * 2
        t = all_times(ti);
        pdf_counts(ti) = numel(delta_tc_vec(delta_tc_vec == t));
    end

    % CDFs.
    delta_cdf = cumsum(pdf_counts) / sum(pdf_counts);

    plot(all_times, delta_cdf, 'LineWidth', 2, 'DisplayName', sim_ids(i));
end
xline(0, 'r--', 'HandleVisibility', 'off')
hold off
leg = legend();
set(leg, 'interpreter', 'latex')
set(gca, 'FontSize', 15)
xlab = xlabel('$$t \ (hr)$$');
set(xlab, 'interpreter', 'latex')
ylab = ylabel('$$Pr(\delta_{tc} < t)$$');
set(ylab, 'interpreter', 'latex')
xlim([-25 25])

%% CDF of absolute value of time discrepancy.

figure()
hold on
for i = 1:length(sim_plan_fps)
    pdf_counts = zeros(size(time));
    delta_tc_vec = abs(delta_tc_ar(:,i));
    for ti = 1:Nt
        t = time(ti);
        pdf_counts(ti) = numel(delta_tc_vec(delta_tc_vec == t));
    end

    % CDFs.
    delta_cdf = cumsum(pdf_counts) / sum(pdf_counts);

    plot(time, delta_cdf, 'LineWidth', 2, 'DisplayName', sim_ids(i));
end
% xline(0, 'r--', 'HandleVisibility', 'off')
hold off
leg = legend();
set(leg, 'interpreter', 'latex')
set(gca, 'FontSize', 18)
xlab = xlabel('$$t \ (hr)$$');
set(xlab, 'interpreter', 'latex')
ylab = ylabel('$$Fraction(|\delta_{tc}| < t)$$');
set(ylab, 'interpreter', 'latex')
xlim([0 5])

%%