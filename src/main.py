# src/main.py
from ode import predator_prey, full_system
import plotting
from analysis_tools import plot_individual_case

def main():
    # 1. Set parameters
    #params = {...}  # Define your ODE parameters here
    params = {
    'a1': 1,
    'a2': 8,
    'aP': 1,
    'h1': 0,
    'h2': 0.5,
    'hP': 0,
    'd1': 0.2,
    'd2': 1,
    'dP': 0.25
    }

    plot_individual_case(params)

    #parameter_sets = [
    #    {'a1': 1, 'a2': 8, 'h2': 0.5, 'aP': 1, 'dP': 0.25},
    #    {'a1': 1, 'a2': 8, 'h2': 1.0, 'aP': 1, 'dP': 0.25},
    #    {'a1': 1, 'a2': 8, 'h2': 2.0, 'aP': 1, 'dP': 0.25},
        # Add more sets as needed
    #]

    #for params in parameter_sets:
        # Run your analysis function(s) here, passing params as arguments
        #run_analysis(params)


    # Parameters for all models
    '''
    resolution = 35

    maxd1 = a1            # maximum mortality rate of C1
    maxd2 = a2/(1+h2*a2)  # maximum mortality rate of C2

    # d1 and d2 variable
    pd1 = np.linspace(0,maxd1,resolution + 2)
    pd2 = np.linspace(0,maxd2,resolution + 2)

    d1  = pd1[1:-1]
    d2  = pd2[1:-1]

    params = {
    'a1': 1,
    'a2': 8,
    'h2': 0.5,
    'd1': np.linspace(0, 1, 35),
    'd2': np.linspace(0, 8/(1+0.5*8), 35),
    'resolution': 35
    }
    result = lin_sat(params)

    # 2. Run simulation
    
    # 3. Analyze results
    # 4. Save results (optional)
    # save_results(results, 'data/results.csv')

    filename = f'matrices_{a1}_{a2}_{aP}_{h2}_{dP}.npz'
    #np.savez(filename,
    #     coexistence_lin_sat=coexistence_lin_sat,
    #     coexistence_sat_sat=coexistence_sat_sat,
    #     coexistence_lin_lin_pred=coexistence_lin_lin_pred,
    #     coexistence_lin_sat_pred=coexistence_lin_sat_pred,
    #     coexistence_sat_sat_pred=coexistence_sat_sat_pred,
    #     params=params)

    # 5. Generate plots
    #plot_results(analysis)
    '''

if __name__ == "__main__":
    main()
