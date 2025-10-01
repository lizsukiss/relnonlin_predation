# src/main.py
from fileinput import filename
import os
from unittest import result
import numpy as np
from scipy import stats
from ode import predator_prey, full_system
from analysis_tools import plot_individual_case
from coexistence import lin_sat, lin_lin_pred, sat_sat, sat_sat_pred, lin_sat_pred, lin_sat_additional_mortality, sat_sat_additional_mortality
from plotting import coexistence_plot_with_lines, summary_plot, summary_dynamics_plots
from utils import make_filename, snail_order
import matplotlib.pyplot as plt
from plotting import simple_d1d2
import glob
import shutil
from tqdm import tqdm
    
def main():
    # baseline parameter set
    baseline = {
    'a1': 1,
    'h1': 0,
    'a2': 4,
    'h2': 1,
    'aP': 1,
    'dP': 0.25,
    'hP': 0,
    'resolution': 100
    }

    varying_params = {
        'a2': [0.125, 0.25, 0.5, 1, 2, 8]       # values for a2
        #'h2': [0.125, 0.25, 0.5, 2, 4, 8],       # values  for h2
        #'aP': [0.125, 0.25, 0.5, 2, 4],          # values for aP
        #'dP': [0.0625, 0.125, 0.5, 0.75, 1]   # values for dP
    }

    # append to param_sets all combinations of baseline with one varying parameter
    param_sets = []

    for param, values in varying_params.items():
        for val in values:
            params = baseline.copy()
            params[param] = val
            param_sets.append(params)

    param_sets = [baseline.copy()] + param_sets
    for i, params in enumerate(param_sets):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"Set {i+1}: {param_str}")
    '''
    run_many_cases_parallel('lin_sat',param_sets)
    print("All lin_sat models done")
    run_many_cases_parallel('lin_sat_pred',param_sets)
    print("All lin_sat_pred models done")
    run_many_cases_parallel('lin_lin_pred',param_sets)
    print("All lin_lin_pred models done")
    
    run_many_cases_parallel('lin_sat_additional_mortality',param_sets)
    
    print("All lin_sat_additional_mortality models done")
    run_many_cases_parallel('sat_sat',param_sets)
    print("All sat_sat models done")
    run_many_cases_parallel('sat_sat_pred',param_sets)
    print("All sat_sat_pred models done")
    run_many_cases_parallel('sat_sat_additional_mortality',param_sets)
    print("All sat_sat_additional_mortality models done")
    '''
    '''
    for h2 in np.arange(0, 4.5, 0.5):
        params = {'a1': 1, 'a2': 4, 'aP': .25, # or a set with [] when model_name = 'all'
                            'h2': h2, 'hP': 0, 'dP': 0.1,
                            'resolution': 100} 
        model_name = 'lin_sat_pred'  # options: 'lin_sat', 'lin_sat_pred', 
                            # 'sat_sat', 'sat_sat_pred',
                            # 'lin_lin', 'lin_lin_pred', 
                            # 'lin_sat_additional_mortality',
                            # 'sat_sat_additional_mortality',
                            # 'all'
        if model_name != 'all':
            run_one_case(model_name, params)
        else:
            run_all_cases([params])
    '''
    for filename in glob.glob('results/matrices/matrices_*.npz'):
        stats = matrix_to_statistics(filename=filename)
    filenames = glob.glob('results/matrices/matrices_1_4_1_1_0_*_50.npz')
    load_and_plot_statistics(param_list=param_sets,varying_param='a2', stat_keys=['statistics_lin_sat_pred', 'statistics_lin_sat', 'statistics_lin_lin_pred','statistics_lin_sat_additional_mortality']) 

def explore_results():  # not really finished yet
    
    # List all result files
    files = glob.glob('results/matrices/matrices_*.npz')
    print("Available simulations:")
    for i, f in enumerate(files):
        try:
            data = np.load(f, allow_pickle=True)
            params = data['params'].item()  # assuming params is saved as a dict
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"{i}: {param_str}")
        except Exception as e:
            print(f"{i}: Could not read parameters from {f} ({e})")

    idx = int(input("Select a simulation by number: "))
    data = np.load(files[idx], allow_pickle=True)
    params = data['params'].item()  # if saved as dict

    print("Loaded parameters:", params)
    # Now ask for mortality rates and plot as needed
    d1_input = float(input("Provide d1 (mortality rate of C1): "))
    d2_input = float(input("Provide d2 (mortality rate of C2): "))

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    aP = params['aP']
    dP = params['dP']
    resolution = params['resolution']

    maxd1 = a1
    maxd2 = a2 / (1 + h2 * a2)
    d1_grid = np.linspace(0, maxd1, resolution + 2)[1:-1]
    d2_grid = np.linspace(0, maxd2, resolution + 2)[1:-1]

    # Find nearest grid point indices
    d1_idx = (np.abs(d1_grid - d1_input)).argmin()
    d2_idx = (np.abs(d2_grid - d2_input)).argmin()
    d1_val = d1_grid[d1_idx]
    d2_val = d2_grid[d2_idx]

    print(f"Nearest grid point for d1: {d1_val}, index: {d1_idx}")
    print(f"Nearest grid point for d2: {d2_val}, index: {d2_idx}")

    params = {'a1': a1, 'a2': a2, 'aP': aP, 'h2': h2, 'dP': dP, 'd1': d1_val, 'd2': d2_val}
    summary_dynamics_plots(params) # defined in plotting
    plt.show(block=False)

def run_one_case(model_name, params):
    
    old_data = {} # for potentially updating the saved file
    coexistence_matrix = None # to check if has been simulated already

    filename = make_filename('results/matrices/matrices', params)
    key = 'coexistence_' + model_name # name of the matrix variable

    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        old_data = dict(data) # will be updated with new results if any
        key = 'coexistence_' + model_name # name of the matrix variable
        if key in data.files: # matrix already saved
            coexistence_matrix = data[key]
            print(f"Results already exist for {filename}. Skipping simulation.")
        
    if coexistence_matrix is None: # matrix not yet saved
        model_func = globals()[model_name]
        result = model_func(params) # might return multiple matrices
        coexistence_matrix = result[0] if isinstance(result, tuple) else result # coexistence matrix: always the first matrix
        # Update old_data with new results
        old_data[key] = coexistence_matrix
        old_data['params'] = params
        np.savez(filename, **old_data)
        print(f"Simulation done and results saved to {filename}.")

    # Plotting
    plot_title = model_name.replace('_', ' ')
    #coexistence_plot_with_lines(coexistence_matrix, params, plot_title)
    simple_d1d2(coexistence_matrix, params, plot_title)
    plt.show()

def run_all_cases(param_sets):
    for params in param_sets:
        for model_name in ["lin_sat_additional_mortality", "sat_sat_additional_mortality"]:
            run_one_case(model_name, params)
    # After all simulations, update DVC and clean up
    os.system("dvc add results") 
    os.system("git add results.dvc")
    os.system("dvc push")
    # not committing to git!
    shutil.rmtree("results/timeseries", ignore_errors=True)  # remove the entire folder
    os.makedirs("results/timeseries", exist_ok=True) # recreate an empty folder

    # plot
    tqdm.write("Plotting starts")
    fig = summary_plot(params)[0]
    plot_filename = make_filename('results/plots/summary', params, extension='.png')
    fig.savefig(plot_filename)    
    tqdm.write("Plotting done")

def matrix_to_statistics(params=None, filename=None): # provide either params or filename
    # determine filename
    if filename is None:
        if params is None:
            raise ValueError("Either params or filename must be provided.")
        filename = make_filename('results/matrices/matrices', params)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No matrix file found: {filename}")
    data = np.load(filename, allow_pickle=True)
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
    data_dict = dict(data)
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
        import matplotlib.pyplot as plt

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
                    data = np.load(filename, allow_pickle=True)
                    param_dict = data['params'].item() 
                    x_val = param_dict[varying_param]
                    resolution = param_dict['resolution']
                else:
                    x_val = filename

            x_vals.append(x_val) # all the values on the x axis
            resolutions.append(resolution)
            data = np.load(filename, allow_pickle=True)
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

        # Sort data by x_vals (so that the plot lines go from left to right)
        sorted_indices = np.argsort(x_vals)

        # Sort x_vals and resolutions
        x_vals = [x_vals[i] for i in sorted_indices]
        resolutions = [resolutions[i] for i in sorted_indices]

        # Sort stats_data for each key
        for key in stat_keys:
            stats_data[key]['num_points'] = [stats_data[key]['num_points'][i] for i in sorted_indices]
            stats_data[key]['num_cycles'] = [stats_data[key]['num_cycles'][i] for i in sorted_indices]


        # Plotting from stats_data
        fig, ax = plt.subplots(figsize=(12, 5))

        if len(stat_keys) == 1: # only one model, stacked diagram
            key = stat_keys[0]
            # Convert to numpy arrays for relative coex area
            y1 = np.array(stats_data[key]['num_points']) / np.array(resolutions)**2
            y2 = np.array(stats_data[key]['num_cycles']) / np.array(resolutions)**2
    
            ax.stackplot(x_vals, y1, y2, labels=['Fixed Points', 'Limit Cycles'], alpha=0.7)
            ax.set_title(f'Stacked Diagram for {key}')
            ax.set_xlabel(varying_param if varying_param else 'Parameter Set')
            ax.set_ylabel('Relative coexistence area')
            ax.legend()
            plt.show()
 
        else: # multiple models, line plots
           
            for key in stat_keys: # cycle through all models specified
                # Convert to numpy arrays for relative coex area
                y1 = np.array(stats_data[key]['num_points']) / np.array(resolutions)**2
                y2 = np.array(stats_data[key]['num_cycles']) / np.array(resolutions)**2
                ax.plot(x_vals, y1+y2, marker='o', label=key.replace('statistics_', '').replace('_', ' '))
            ax.set_title('Relative size of coexistence region')
            ax.set_xlabel(varying_param if varying_param else 'Parameter Set')
            ax.set_ylabel('Relative coexistence area')
            ax.legend()
            plt.tight_layout()
            plt.show()

import concurrent.futures

def run_many_cases_parallel(model_name, param_sets, max_workers=os.cpu_count()):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one_case, model_name, params) for params in param_sets]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Will raise exception if any occurred
            except Exception as e:
                print(f"Error in parallel run: {e}")


if __name__ == "__main__":
    main()             # Uncomment to run simulations
    #explore_results()  # Uncomment to explore results