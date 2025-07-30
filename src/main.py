# src/main.py
import numpy as np
from ode import predator_prey, full_system
from analysis_tools import plot_individual_case
from coexistence import lin_sat, lin_lin_pred, sat_sat, sat_sat_pred, lin_sat_pred
from plotting import summary_plot
import matplotlib.pyplot as plt

import glob

from tqdm import tqdm
    
def main():

    param_sets = [
    {'a1': 1, 'a2': 8, 'aP': 1,
     'h2': 0.5,
     'dP': 0.25,
     'resolution': 50},
     #
     {'a1': 1, 'a2': 4, 'aP': 1,
     'h2': 1,
     'dP': 0.25,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1,
     'h2': 2,
     'dP': 0.25,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1,
     'h2': 0.5,
     'dP': 0.25,
     'resolution': 50},
     #
     {'a1': 1, 'a2': 4, 'aP': 0.5,
     'h2': 1,
     'dP': 0.25,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1.5,
     'h2': 2,
     'dP': 0.25,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 2,
     'h2': 0.5,
     'dP': 0.25,
     'resolution': 50},
      #
     {'a1': 1, 'a2': 4, 'aP': 0.5,
     'h2': 1,
     'dP': 0.125,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 1.5,
     'h2': 2,
     'dP': 0.375,
     'resolution': 50},
     {'a1': 1, 'a2': 4, 'aP': 2,
     'h2': 0.5,
     'dP': 0.5,
     'resolution': 50}
    ]
    print("Parameters set")

    for params in tqdm(param_sets, desc="Parameter sets"):
        tqdm.write(f"Running with params: {params}")

        # run the coex functions
        tqdm.write("Starting LinSat")
        coexistence_lin_sat = lin_sat(params)
        tqdm.write("LinSat done")

        tqdm.write("Starting LinSatPred")
        coexistence_lin_sat_pred = lin_sat_pred(params)
        tqdm.write("LinSatPred done")

        tqdm.write("Starting SatSat")
        coexistence_sat_sat = sat_sat(params)
        tqdm.write("SatSat done")

        tqdm.write("Starting SatSatPred")
        coexistence_sat_sat_pred = sat_sat_pred(params)
        tqdm.write("SatSatPred done")

        tqdm.write("Starting LinLinPred")
        coexistence_lin_lin_pred = lin_lin_pred(params)
        tqdm.write("LinLinPred done")

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
    d1_input = int(input("Provide d1 (mortality rate of C1): "))
    d2_input = int(input("Provide d2 (mortality rate of C2): "))

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    resolution = params['resolution']

    maxd1 = a1
    maxd2 = a2 / (1 + h2 * a2)
    d1_grid = np.linspace(0, maxd1, resolution + 2)[1:-1]
    d2_grid = np.linspace(0, maxd2, resolution + 2)[1:-1]

    # Find nearest grid point indices
    i = (np.abs(d1_grid - d1_input)).argmin()
    j = (np.abs(d2_grid - d2_input)).argmin()

    print(f"Nearest grid point for d1: {d1_grid[i]}, index: {i}")
    print(f"Nearest grid point for d2: {d2_grid[j]}, index: {j}")

    
    # plot_individual_case(params)
    # plt.show()

if __name__ == "__main__":
    main()             # Uncomment to run simulations
    explore_results()  # Uncomment to explore results