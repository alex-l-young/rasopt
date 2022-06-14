% Depth comparison.
% Compare the ground truth to simulated depths.

close all 
clear

% Load in the depth data.
gt_plan_fp = "C:\Users\ayoun\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Roughness_Output\Secchia_Panaro.p23_GT.hdf";
sim_best_plan_fp = "C:\Users\ayoun\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Roughness_Output\Secchia_Panaro.p23_2veg.hdf";
sim_guess_plan_fp = "C:\Users\ayoun\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\Roughness_Output\Secchia_Panaro.p23_2veg.hdf";

% Timestep length in hours.
ts_len = 1/6;

% HDF Paths.
% Path to cell coordinates.
cell_coord_path = '/Geometry/2D Flow Areas/Secchia_Panaro/Cells Center Coordinate';

% Path to water depths.
depth_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Secchia_Panaro/Depth';

% Extract the depths.
gt_depths = h5read(gt_plan_fp, depth_path);
sim_best_depths = h5read(sim_best_plan_fp, depth_path);
sim_guess_depths = h5read(sim_guess_plan_fp, depth_path);
[ncell, Nt] = size(gt_depths); 
time = (0:Nt-1) .* ts_len;

% Only include first N time steps.
Nts = 849;
gt_depths = gt_depths(:,1:Nts);
sim_best_depths = sim_best_depths(:,1:Nts);
sim_guess_depths = sim_guess_depths(:,1:Nts);

% Cell locations.
cell_coords = h5read(gt_plan_fp, cell_coord_path)';
lats = zeros(ncell,1);
lons = zeros(ncell,1);
for i = 1:ncell
    [lat, lon] = utm2deg(cell_coords(i,1),cell_coords(i,2),'32 N');
    lats(i) = lat;
    lons(i) = lon;
end

%% Remove cells that do not hit critical threshold based on GT values.
dcrit = 0.1;
gt_depths_dc = gt_depths(max(gt_depths,[],2) >= dcrit,:);
sim_best_depths_dc = sim_best_depths(max(gt_depths,[],2) >= dcrit,:);
sim_guess_depths_dc = sim_guess_depths(max(gt_depths,[],2) >= dcrit,:);
lats = lats(max(gt_depths,[],2) >= dcrit);
lons = lons(max(gt_depths,[],2) >= dcrit);

%% Determine time to critical depth.
[~,gt_tc] = max(gt_depths_dc >= dcrit,[],2);
[~,sim_best_tc] = max(sim_best_depths_dc >= dcrit,[],2);
[~,sim_guess_tc] = max(sim_guess_depths_dc >= dcrit,[],2);

delta_best_tc = (gt_tc - sim_best_tc) * ts_len;
delta_guess_tc = (gt_tc - sim_guess_tc) * ts_len;

t_cut = 10.125;
delta_best_cut = delta_best_tc;
delta_best_cut(delta_best_cut < -t_cut) = -t_cut;
delta_best_cut(delta_best_cut > t_cut) = t_cut;
delta_guess_cut = delta_guess_tc;
delta_guess_cut(delta_guess_cut < -t_cut) = -t_cut;
delta_guess_cut(delta_guess_cut > t_cut) = t_cut;
bins = -t_cut:0.25:t_cut;
bins = 51;
figure()
hold on
histogram(delta_guess_cut, bins, 'DisplayName', 'Guess')%, 'Normalization','probability')
histogram(delta_best_cut, bins, 'DisplayName', 'Optim.')%, 'Normalization','probability')
hold off
xlabel('\delta_{tc} (hr)')
ylabel('Number of Cells')
set(gca, 'FontSize', 16)
legend()
% ylim([0 21000])
% xlim([-8 8])

%% Plot the difference geographically.
addpath('customcolormap')
J = customcolormap([0 0.5 1], {'#0008ff','#ffffff','#ff1100'},21);
figure()
geoscatter(lats, lons, 5, delta_best_tc, 'filled')
cmap = colormap(J);
c = colorbar();
% c.Label = '\delta_{tc}';
caxis([-10 10])
set(gca, 'FontSize', 16)

%% PDF, CDF, and Exceedance Probability

% Absolute value.
Bdtc_abs = abs(delta_best_tc);
Gdtc_abs = abs(delta_guess_tc);

% Log transform histogram.
Bdtc_abs_log = log10(Bdtc_abs);
Gdtc_abs_log = log10(Gdtc_abs);

% Histogram.
figure()
hold on
Bhist = histogram(Bdtc_abs);
Ghist = histogram(Gdtc_abs);
hold off

Bpdf = Bhist.BinCounts / sum(Bhist.BinCounts);
Gpdf = Ghist.BinCounts / sum(Ghist.BinCounts);

Bpdf_counts = zeros(size(time));
Gpdf_counts = zeros(size(time));
for ti = 1:Nt
    t = time(ti);
    Bpdf_counts(ti) = numel(Bdtc_abs(Bdtc_abs == t));
    Gpdf_counts(ti) = numel(Gdtc_abs(Gdtc_abs == t));
end

% CDFs.
Bcdf = cumsum(Bpdf_counts) / sum(Bpdf_counts);
Gcdf = cumsum(Gpdf_counts) / sum(Gpdf_counts);

figure()
hold on
plot(time, Bcdf, 'LineWidth', 2);
plot(time, Gcdf, 'LineWidth', 2);
hold off
leg = legend('$\hat{n}=0.0575$', '$n=0.07$');
set(leg, 'Interpreter', 'latex')
set(gca, 'FontSize', 15)
xlab = xlabel('$$t \ (hr)$$');
set(xlab, 'interpreter', 'latex')
ylab = ylabel('$$Pr(|\delta_{tc}| < t)$$');
set(ylab, 'interpreter', 'latex')
xlim([0 25])

% Exceedance probability.
figure()
hold on
plot(time, 1 - Bcdf, 'LineWidth', 2);
plot(time, 1 - Gcdf, 'LineWidth', 2);
hold off
leg = legend('$\hat{n}=0.0575$', '$n=0.07$');
set(leg, 'Interpreter', 'latex')
set(gca, 'FontSize', 15)
xlab = xlabel('$$t \ (hr)$$');
set(xlab, 'interpreter', 'latex')
ylab = ylabel('$$1-Pr(|\delta_{tc}| < t)$$');
set(ylab, 'interpreter', 'latex')
xlim([0 25])

% figure()
% plot(time, Bcdf - Gcdf)

% figure()
% hold on
% scatter(Bcdf, Gcdf)
% plot([0 max(Bcdf)], [0 max(Bcdf)])
% hold off

%% Non absolute pdf/cdf.

Bpdf_counts = zeros(size(time));
Gpdf_counts = zeros(size(time));
all_times = [-fliplr(time) time];
for ti = 1:Nt * 2
    t = all_times(ti);
    Bpdf_counts(ti) = numel(delta_best_tc(delta_best_tc == t));
    Gpdf_counts(ti) = numel(delta_guess_tc(delta_guess_tc == t));
end

% CDFs.
Bcdf = cumsum(Bpdf_counts) / sum(Bpdf_counts);
Gcdf = cumsum(Gpdf_counts) / sum(Gpdf_counts);

figure()
hold on
plot(all_times, Bcdf, 'LineWidth', 2);
plot(all_times, Gcdf, 'LineWidth', 2);
xline(0, 'r--')
hold off
leg = legend('$\hat{n}=0.0575$', '$n=0.07$', '$0 \ hr$');
set(leg, 'Interpreter', 'latex')
set(gca, 'FontSize', 15)
xlab = xlabel('$$t \ (hr)$$');
set(xlab, 'interpreter', 'latex')
ylab = ylabel('$$Pr(\delta_{tc} < t)$$');
set(ylab, 'interpreter', 'latex')
xlim([-25 25])
% xlim([min(delta_best_tc) max(delta_best_tc)])

% figure()
% plot(all_times, Bcdf - Gcdf)
