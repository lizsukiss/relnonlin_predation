# src/main.py
import os
import numpy as np
from scipy import stats
from ode import predator_prey, full_system
from analysis_tools import plot_individual_case
from parameter_study import ParameterCase
from plotting import coexistence_plot_with_lines
from utils import make_filename, snail_order, simulate_and_save
import matplotlib.pyplot as plt
from plotting import simple_d1d2
import glob
import shutil
from functools import partial

import concurrent.futures
    
def main():
    """
    
    Main function to run the models for different parameter sets, analyze and plot results

    """


    # Test script
    params = {
        'a1': 1, 'h1': 0, 'a2': 0.5, 'h2': 1.0,
        'aP': 1, 'dP': 0.25, 'hP': 0,
        'resolution': 100
    }

    # Run and save
    study = ParameterCase(params)
    study.run(['lin_lin_pred'])
    study.save()

    # Load back
    study2 = ParameterCase(params)
    study2.update_results()

    # Check structure
    print(study2.results['lin_lin_pred'].keys())
    # Should print: dict_keys(['coexistence_lin_lin_pred', 'predatordensity_lin_lin_pred', 'stability_lin_lin_pred'])

    """


    # baseline parameter set
    baseline = {
    'a1': 1,
    'h1': 0,
    'a2': 4,
    'h2': 1,
    'aP': 1,
    'dP': 0.25,
    'hP': 0,
    'resolution': 10
    }

    varying_params = {
        'a2': np.logspace(-2,5, num=8, endpoint=True, base=2.0),       # values for a2
        'h2': np.logspace(-2,5, num=8, endpoint=True, base=2.0),       # values  for h2
        #'aP': [0.125, 0.25, 0.5, 2, 4],       # values for aP
        #'dP': [0.0625, 0.125, 0.5, 0.75, 1]   # values for dP
    }

    # append to param_sets all combinations of baseline with one varying parameter
    param_sets = []

    for a2_value in varying_params['a2']:
        for h2_value in varying_params['h2']:
            params = baseline.copy()
            params['a2'] = a2_value
            params['h2'] = h2_value
            param_sets.append(params)
    
    #param_sets = [baseline.copy()] + param_sets
    
    for i, params in enumerate(param_sets):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"Set {i+1}: {param_str}")
    
    
    # Run all three models on all parameter sets in parallel
    run_parametrized_models_in_parallel(
        param_sets, 
        model_names=['lin_lin_pred','lin_sat_pred','lin_sat']
    )
    print("All simulations done!")
    
    
    # all matrices done, now compute statistics
    for params in param_sets:
        stats = matrix_to_statistics(params)
        print(f"Statistics for params {params}: {stats}")
    

    display_names = {
        'lin_sat': 'Rel. nonlinearity',
        'lin_sat_pred': 'Rel. nonlinearity + predation',
        'lin_lin_pred': 'Predation'
        }

    # for each a2, make a plot with varying h2
    for a2_value in varying_params['a2']:
        filtered_params = [p for p in param_sets if p['a2'] == a2_value]
        fig = load_and_plot_statistics(filtered_params,
                                       stat_keys=['statistics_lin_sat_pred'],
                                       varying_param='h2')
        # put an extra note on the plot
        fig.suptitle(f'$a_2={a2_value}$', x = .9, y=.98, fontsize=12)
        # save the plot
        plot_filename = make_filename(f'results/plots/linsatpred_a2_{a2_value}', [], extension='.png')
        fig.savefig(plot_filename)
        print(f"Statistics plot saved for a2={a2_value}")
        # loop through all h2 values for this a2 and save individual plots as well
        for params in filtered_params:
            model_name_for_individual_plots = 'lin_sat_pred' # model to use for individual plots
            h2_value = params['h2']
            fig2, ax2 = plt.subplots()
            with np.load(make_filename('results/matrices/matrices', params)) as data:
                coexistence_matrix = data[f'coexistence_{model_name_for_individual_plots}']
            coexistence_plot_with_lines(coexistence_matrix=coexistence_matrix,
                                        params=params,title = f'$a_2={a2_value}, h_2={h2_value}$',
                                        ax=ax2)
            # add an extra note on the plot whether it's rel. nonlin., predation or both
            # based on model_name_for_individual_plots
            if model_name_for_individual_plots in display_names:
                ax2.text(0.02, 0.98,
                    display_names[model_name_for_individual_plots],
                    transform=ax2.transAxes,
                    fontsize=10,
                    va='top', ha='left'
                )
            plot_filename = make_filename(f'results/plots/linlinpred_a2_{a2_value}_h2_{h2_value}', [], extension='.png')
            fig2.savefig(plot_filename)
            plt.close(fig2)
            print(f"Individual plot saved for a2={a2_value}, h2={h2_value}")
    """

def run_single_study(params, model_names):
    """Run ParameterCase for a single parameter set."""
    study = ParameterCase(params)
    study.run(model_names)
    study.save()
    return study

def run_parametrized_models_in_parallel(param_sets, model_names, max_workers=None):
    """

    Run ParameterCase on multiple parameter sets in parallel.
    
    Args:
        param_sets (list): List of parameter dictionaries
        model_names (list): List of model names to run (e.g., ['lin_sat', 'lin_sat_pred', 'lin_lin_pred'])
        max_workers (int, optional): Number of parallel workers. Defaults to CPU count.
    
    """
    if max_workers is None:
        max_workers = os.cpu_count()
    
    # Create a partial function with model_names bound
    run_func = partial(run_single_study, model_names=model_names)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_func, params) for params in param_sets]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in parallel execution: {e}")

def matrix_to_statistics(params=None, filename=None): # provide either params or filename
    # determine filename
    if filename is None:
        if params is None:
            raise ValueError("Either params or filename must be provided.")
        filename = make_filename('results/matrices/matrices', params)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No matrix file found: {filename}")
    with np.load(filename, allow_pickle=True) as data:
        # compute stats for all matrices that are present for these parameters
        stats = {}
        for key in data.files:
            if key.startswith('coexistence_'): # these are the matrices
                matrix = data[key]
                num_point = np.sum(matrix == 1) # number of fixed point equilibria
                num_cycle = np.sum(matrix == 2) # number of limit cycle equilibria
                stats_key = 'statistics_' + key.split('coexistence_')[1]
                stats[stats_key] = {'num_points': int(num_point), 'num_cycles': int(num_cycle)}
    
        # Append stats to data and save in the same file
        data_dict = {key: data[key].copy() for key in data.files} # copy since otherwise it became nan
        for stats_key, stats_val in stats.items():
            data_dict[stats_key] = stats_val
        np.savez(filename, **data_dict)
    return stats

def load_and_plot_statistics(param_list, stat_keys=None, varying_param=None):
    """
    Load statistics for multiple parameter sets and plot them.
    param_list: list of parameter dicts (or filenames)
    stat_keys: list of statistics keys to plot (e.g., ['statistics_lin_sat', ...])
    varying_param: name of the parameter that varies between param sets (for x-axis)
    """

    if stat_keys is None:
        stat_keys = ['statistics_lin_sat', 'statistics_lin_sat_pred', 'statistics_lin_lin_pred']

    # Prepare data
    x_vals = []
    resolutions = []
    stats_data = {key: {'num_points': [], 'num_cycles': []} for key in stat_keys}

    for params in param_list:
        # Get x value for plotting
        if isinstance(params, dict):
            x_val = params[varying_param] if varying_param else str(params) # retrieves the value of the varying parameter
            resolution = params['resolution']
            filename = make_filename('results/matrices/matrices', params)
        else:
            filename = params
            if varying_param:
                # Try to extract param from filename if possible
                with np.load(filename, allow_pickle=True) as data:
                    param_dict = data['params'].item() 
                    x_val = param_dict[varying_param]
                    resolution = param_dict['resolution']
            else:
                x_val = filename

        x_vals.append(x_val) # all the values on the x axis
        resolutions.append(resolution)
        with np.load(filename, allow_pickle=True) as data:
            for key in stat_keys:
                if key in data.files:
                    stat = data[key].item()
                    # store all data in stats_data
                    stats_data[key]['num_points'].append(stat['num_points'])  
                    stats_data[key]['num_cycles'].append(stat['num_cycles'])
                else:
                    stats_data[key]['num_points'].append(np.nan)
                    stats_data[key]['num_cycles'].append(np.nan)

    if min(resolutions) != max(resolutions):
        print("Warning: Different resolutions found.")

    # Plotting from stats_data
    fig, ax = plt.subplots(figsize=(12, 5))

    latex_labels = {
            'a2': r'$a_2$',
            'h2': r'$h_2$',
            'aP': r'$a_P$',
            'dP': r'$d_P$',
            'hP': r'$h_P$',
            'a1': r'$a_1$',
            'h1': r'$h_1$',
            'd1': r'$d_1$',
            'd2': r'$d_2$'
        }

    if len(stat_keys) == 1: # only one model, stacked diagram
        key = stat_keys[0]
        # Convert to numpy arrays for relative coex area
        y1 = np.array(stats_data[key]['num_points']) / (np.array(resolutions)**2)
        y2 = np.array(stats_data[key]['num_cycles']) / (np.array(resolutions)**2)


        print(f"DEBUG: y1 = {y1}")  # Add this
        print(f"DEBUG: y2 = {y2}")  # Add this
        print(f"DEBUG: y1 + y2 = {y1 + y2}")  # Add this

        ax.stackplot(x_vals, y1, y2, labels=['Fixed Points', 'Limit Cycles'], alpha=1, colors=['#707070', '#202020'])   
        ax.set_title(f'Stacked Diagram for {key}')
        ax.set_xlabel(latex_labels.get(varying_param, varying_param) if varying_param else 'Parameter Set')

        ax.set_ylabel('Relative coexistence area')

        print(f"DEBUG: x_vals = {x_vals}")  # Check what values you have
        if all(x > 0 for x in x_vals):
            ax.set_xscale('log')
            ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        else:
            print(f"WARNING: Cannot use log scale, x_vals contain non-positive values: {x_vals}")

        ax.legend()
    
    else: # multiple models, line plots
        
        display_names = {
        'statistics_lin_sat': 'Rel. nonlinearity',
        'statistics_lin_sat_pred': 'Rel. nonlinearity + predation',
        'statistics_lin_lin_pred': 'Predation'
        }

        # choose colors: either explicit map or default cycler
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_map = {
            'statistics_lin_sat': 'red',
            'statistics_lin_sat_pred': 'blue',
            'statistics_lin_lin_pred': 'green'
        }
        for i, key in enumerate(stat_keys):
            # Convert to numpy arrays for relative coex area
            y1 = np.array(stats_data[key]['num_points']) / (np.array(resolutions)**2)
            y2 = np.array(stats_data[key]['num_cycles']) / (np.array(resolutions)**2)
            color = color_map.get(key, default_colors[i % len(default_colors)])
            ax.plot(x_vals, y1 + y2, marker='o', label=display_names.get(key, 'Unknown'), color=color)
        ax.set_title('Relative size of coexistence region')
        ax.set_xlabel(latex_labels.get(varying_param, varying_param) if varying_param else 'Parameter Set')

        ax.set_ylabel('Relative coexistence area')
        ax.set_xscale('log')
        ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend()
        plt.tight_layout()
        #plt.show()

    return fig


if __name__ == "__main__":
    main()             # Uncomment to run simulations
