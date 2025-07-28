# src/main.py
import numpy as np
from ode import predator_prey, full_system
from analysis_tools import plot_individual_case
from coexistence import lin_sat, lin_lin_pred, sat_sat, sat_sat_pred, lin_sat_pred
from plotting import summary_plot
import matplotlib.pyplot as plt

def main():

    param_sets = [
    {'a1': 1, 'a2': 8, 'aP': 1,
     'h2': 0.5,
     'dP': 0.25,
     'resolution': 5}
    ]
    print("Parameters set")

    for params in param_sets:
        # run the coex functions
        coexistence_lin_sat = lin_sat(params)
        print("LinSat done")

        coexistence_lin_sat_pred = lin_sat_pred(params)
        print("LinSatPred done")

        coexistence_sat_sat = sat_sat(params)
        print("SatSat done")

        coexistence_sat_sat_pred = sat_sat_pred(params)
        print("SatSatPred done")

        coexistence_lin_lin_pred = lin_lin_pred(params)
        print("LinLinPred done")

       # save
        filename = f'results/matrices/matrices_{params["a1"]}_{params["a2"]}_{params["aP"]}_{params["h2"]}_{params["dP"]}.npz'
        np.savez(filename,
                 coexistence_lin_sat=coexistence_lin_sat,
                 coexistence_lin_sat_pred=coexistence_lin_sat_pred,
                 coexistence_sat_sat=coexistence_sat_sat,
                 coexistence_sat_sat_pred=coexistence_sat_sat_pred,
                 coexistence_lin_lin_pred=coexistence_lin_lin_pred,
                 params=params)
        print("Saving everything")

       # plot
        fig, axs = summary_plot(params)
        plot_filename = f'results/plots/summary_{params["a1"]}_{params["a2"]}_{params["aP"]}_{params["h2"]}_{params["dP"]}_{params["resolution"]}.png'
        fig.savefig(plot_filename)    
        print("Plotting done and saved")

if __name__ == "__main__":
    main()
