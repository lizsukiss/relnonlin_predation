# src/main.py
import os
import numpy as np
from ode import predator_prey, full_system
from analysis_tools import plot_individual_case
from coexistence import lin_sat, lin_lin_pred, sat_sat, sat_sat_pred, lin_sat_pred
from plotting import summary_plot, summary_dynamics_plots
import matplotlib.pyplot as plt

import glob
import shutil

from tqdm import tqdm
    
def main():

    param_sets = [
    # varying aP
     {'a1': 1, 'a2': 4, 'aP': 0.5,
     'h2': 1,
     'dP': 0.25,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1,
     'h2': 1,
     'dP': 0.25,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1.5,
     'h2': 1,
     'dP': 0.25,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 2,
     'h2': 1,
     'dP': 0.25,
     'resolution': 50},
    # varying dP
     {'a1': 1, 'a2': 4, 'aP': 1,
     'h2': 1,
     'dP': 0.0625,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1,
     'h2': 1,
     'dP': 0.125,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1,
     'h2': 1,
     'dP': 0.25,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1,
     'h2': 1,
     'dP': 0.375,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1,
     'h2': 1,
     'dP': 0.5,
     'resolution': 50}
    ]
    print("Parameters set")

    for params in tqdm(param_sets, desc="Parameter sets"):
        tqdm.write(f"Running with params: {params}")

        filename = f'results/matrices/matrices_{params["a1"]}_{params["a2"]}_{params["aP"]}_{params["h2"]}_{params["dP"]}_{params["resolution"]}.npz'

        if os.path.exists(filename):
            results = np.load(filename, allow_pickle=True)
            
        else:
            results = {}
            results["params"] = params
        
        # run the coex functions
        if "coexistence_lin_sat" not in results:  
            tqdm.write("Starting LinSat")
            coexistence_lin_sat = lin_sat(params)
            tqdm.write("LinSat done")
            results['coexistence_lin_sat'] = coexistence_lin_sat
            np.savez(filename, **results)  # Overwrites file, but keeps all arrays so far

        if "coexistence_lin_sat_pred" not in results:  
            tqdm.write("Starting LinSatPred")
            coexistence_lin_sat_pred = lin_sat_pred(params)
            tqdm.write("LinSatPred done")
            results['coexistence_lin_sat_pred'] = coexistence_lin_sat_pred
            np.savez(filename, **results)  # Overwrites file, but keeps all arrays so far

        if "coexistence_sat_sat" not in results:
            tqdm.write("Starting SatSat")
            coexistence_sat_sat = sat_sat(params)
            tqdm.write("SatSat done")
            results['coexistence_sat_sat'] = coexistence_sat_sat
            np.savez(filename, **results)  # Overwrites file, but keeps all arrays so far

        if "coexistence_sat_sat_pred" not in results:
            tqdm.write("Starting SatSatPred")
            coexistence_sat_sat_pred = sat_sat_pred(params)
            tqdm.write("SatSatPred done")
            results['coexistence_sat_sat_pred'] = coexistence_sat_sat_pred
            np.savez(filename, **results)  # Overwrites file, but keeps all arrays so far

        if "coexistence_lin_lin_pred" not in results:
            tqdm.write("Starting LinLinPred")
            coexistence_lin_lin_pred = lin_lin_pred(params)
            tqdm.write("LinLinPred done")
            results['coexistence_lin_lin_pred'] = coexistence_lin_lin_pred
            np.savez(filename, **results)

        # save
            filename = f'results/matrices/matrices_{params["a1"]}_{params["a2"]}_{params["aP"]}_{params["h2"]}_{params["dP"]}_{params["resolution"]}.npz'
            np.savez(filename,
                    coexistence_lin_sat=coexistence_lin_sat,
                    coexistence_lin_sat_pred=coexistence_lin_sat_pred,
                    coexistence_sat_sat=coexistence_sat_sat,
                    coexistence_sat_sat_pred=coexistence_sat_sat_pred,
                    coexistence_lin_lin_pred=coexistence_lin_lin_pred,
                    params=params)
            tqdm.write("Saving everything")

            # ... run simulation and save results ...
            os.system("dvc add results")  # or the specific file
            os.system("git add results.dvc")
            os.system("dvc push")
            # not committing to git!
            # Remove local data to free up space
            shutil.rmtree("results/timeseries")  # remove the entire folder
            os.makedirs("results/timeseries", exist_ok=True) # recreate an empty folder

        # plot
        fig, axs = summary_plot(params)
        plot_filename = f'results/plots/summary_{params["a1"]}_{params["a2"]}_{params["aP"]}_{params["h2"]}_{params["dP"]}_{params["resolution"]}.png'
        fig.savefig(plot_filename)    
        tqdm.write("Plotting done and saved")


def explore_results():
    
    # List all result files (with CoPilot)
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

if __name__ == "__main__":
    main()             # Uncomment to run simulations
    #explore_results()  # Uncomment to explore results