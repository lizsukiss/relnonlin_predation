# src/main.py
import os
import numpy as np
from ode import predator_prey, full_system
from analysis_tools import plot_individual_case
from coexistence import lin_sat, lin_lin_pred, sat_sat, sat_sat_pred, lin_sat_pred, lin_sat_additional_mortality, sat_sat_additional_mortality
from plotting import coexistence_plot_with_lines, summary_plot, summary_dynamics_plots
from utils import make_filename
import matplotlib.pyplot as plt
from plotting import simple_d1d2
import glob
import shutil
from tqdm import tqdm
    
def main():

    params = {'a1': 1, 'a2': 4, 'aP': 1, # or a set with [] when model_name = 'all'
                           'h2': 1, 'hP': 0, 'dP': 0.25,
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
    plt.show()

def run_one_case(model_name, params):
    model_func = globals()[model_name]
    coexistence_matrix = model_func(params)[0]
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

if __name__ == "__main__":
    main()             # Uncomment to run simulations
    #explore_results()  # Uncomment to explore results