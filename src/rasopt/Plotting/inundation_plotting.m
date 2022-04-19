% Inundation plotting.

close all
clear

% Load in inundation arrays.
gt_and_sim_struct = load("C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\1_veg\gt_and_sim_1veg_camp005.mat");
gt_and_notsim_struct = load("C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\1_veg\gt_and_notsim_1veg_camp005.mat");
notgt_and_sim_struct = load("C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\1_veg\notgt_and_sim_1veg_camp005.mat");
gt_or_sim_struct = load("C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\1_veg\gt_or_sim_1veg_camp005.mat");
gt_xor_sim_struct = load("C:\Users\ay434\Box\Research\Flood_Sim_Materials\BayesOpt_Paper\Data\1_veg\gt_xor_sim_1veg_camp005.mat"); 

gt_and_sim = gt_and_sim_struct.gt_and_sim;
gt_and_notsim = gt_and_notsim_struct.gt_and_notsim;
notgt_and_sim = notgt_and_sim_struct.notgt_and_sim;
gt_or_sim = gt_or_sim_struct.gt_or_sim;
gt_xor_sim = gt_xor_sim_struct.gt_xor_sim;

% Coordinates.
top_left = [652093.732276195 4972124.219236948]; % lon, lat
bot_right = [682516.732276195, 4945405.219236948]; % lon, lat
dx = 20;
dy = 20;
[Ny, Nx] = size(gt_and_sim);
lons = linspace(top_left(1), bot_right(1), Nx);
lats = linspace(bot_right(2), top_left(2), Ny);

xtick_select = round(linspace(1,Nx,5));
ytick_select = round(linspace(1,Ny,5));
utmzone = repmat('32 N', 5, 1);
[ytick_labels, xtick_labels] = utm2deg(lons(round(linspace(1,Nx,5))), ...
    lats(round(linspace(1,Ny,5))), utmzone); 

% Plot the match and mismatched areas.
match = gt_and_sim;
match(~isnan(gt_and_notsim)) = 2;
match(~isnan(notgt_and_sim)) = 3;
figure()
h1 = axes;
gas = imagesc(match,'AlphaData',~isnan(match));
colormap(h1, cbrewer('qual', 'Dark2',3))
axis ij
c = colorbar;
c.Ticks = [1.333 2 2.66666];
c.TickLabels = {'$$A_o \cap A_e$$', '$$A_o \cap A_e^c$$', '$$A_o^c \cap A_e$$'};
c.TickLabelInterpreter = 'latex';
c.FontSize = 16;
xticks(xtick_select)
xticklabels(xtick_labels)
yticks(ytick_select)
yticklabels(ytick_labels)
set(gca, 'FontSize', 12)
xlabel('Longitude')
ylabel('Latitude')

