# ====================================================================================================
# Script containing plotting functions for BayesOpt.
# Alex Young
# June 2021
# ====================================================================================================

# Library imports.
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_gaussian_process
import os

# Local imports.
from rasopt.Utilities import utils

# ====================================================================================================
def convergence_plot(res, fig_save_path, xlabel="Number of Calls", ylabel="minf(x) after j calls",
                     title="Convergence Plot", figsize=(10,10)):
    """
    Plots the convergence of the BayesOpt procedure. Saves the figure to the specified location.
    :param res: Result object from BO.
    :param fig_save_path: Path to save figure to.
    :param xlabel: Label for x-axis.
    :param ylabel: Label for y-axis.
    :param title: Title of plot.
    :param figsize: Figure size.
    :return: None
    """
    # Plot the convergence.
    fig, ax = plt.subplots(figsize=figsize)
    plot_convergence(res, ax)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    plt.savefig(fig_save_path)
    plt.cla()

    return


def plot_GP(res, n_iter, gp_fig_name, acq_fig_name, f_wo_noise=None, noise_level=0, show_legend=True):
    """
    Plots the Gaussian process and acquisition function from the Bayesian optimization procedure.
    :param res: Results object from BO.
    :param n_iter: Number of iterations to plot.
    :param gp_fig_name: Gaussian process figure save path.
    :param acq_fig_name: Acquisition function figure save path.
    :param f_wo_noise: Function without noise if desired.
    :param noise_level: Level of noise present in the data.
    :param show_legend: Show the legend for the plots.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plot_gaussian_process(res, ax=ax, n_calls=n_iter,
                               objective=f_wo_noise,
                               noise_level=noise_level,
                               show_legend=show_legend, show_title=False,
                               show_next_point=False, show_acq_func=False)
    # ax.set_ylabel("")
    # ax.set_xlabel("")
    ax.set_xlabel("Parameter Value", fontsize=18)
    ax.set_ylabel("Function Value", fontsize=18)
    ax.set_title("Gaussian Process on Objective Function", fontsize=20)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=14)
    plt.savefig(gp_fig_name)
    plt.cla()

    # Plot Acquisition
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plot_gaussian_process(res, ax=ax, n_calls=n_iter,
                               show_legend=show_legend, show_title=False,
                               show_mu=False, show_acq_func=True,
                               show_observations=False,
                               show_next_point=True)
    ax.set_xlabel("Parameter Value", fontsize=18)
    ax.set_ylabel("Function Value", fontsize=18)
    ax.set_title("Gaussian Process on Objective Function", fontsize=20)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=14)
    plt.savefig(acq_fig_name)
    plt.cla()

    return


def plot_param_time_series(time_steps, param_values, save_fp):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(time_steps, param_values)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Optimal Parameter Value')
    ax.set_title('Parameter Value as Simulation Progresses')
    plt.savefig(save_fp, dpi=300)
    plt.cla()


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
    gt_dep_df = utils.extract_depths(gt_fp, depth_path, cell_coord_path)

    # Extract simulated depths.
    sim_dep_df = utils.extract_depths(sim_fp, depth_path, cell_coord_path)

    for loc in locations:
        # Select closest row from simulated data set.
        sim_row = utils.closest_row(sim_dep_df, loc)

        # Select closest row from measured data set.
        gt_row = utils.closest_row(gt_dep_df, loc)

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