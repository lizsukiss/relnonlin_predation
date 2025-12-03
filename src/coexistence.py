import numpy as np
from scipy import integrate as integ
from functools import partial
from ode import predator_prey, full_system
from utils import check_params, get_grid, make_filename, simulate_and_save, snail_order
import os
import shutil
import psutil
import sympy as sp


def lin_sat(params):

    check_params(params, {'a1', 'a2', 'h2', 'resolution'})

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    resolution = params['resolution']

    d1 = get_grid(a1, 0, resolution)
    d2 = get_grid(a2, h2, resolution)
    ###########################################################

    invasionrate_C1 = np.zeros([resolution,resolution])
    invasionrate_C2 = np.zeros([resolution,resolution])

    # C2 can invade C1 if f2(R_{C_1}^*) > 0 (R_{C_1}^* is denoted by R_Eq)
    for i in range(resolution):
        R_Eq = d1[i]/(a1)      
        invasionrate_C2[i,:] = a2*R_Eq/(1+a2*h2*R_Eq) - d2

    invasionrate_C2[invasionrate_C2>0]=1
    invasionrate_C2[invasionrate_C2<0]=0 

    # C1 can invade C2 if f1(\overline{R_{C_2}}) > 0 (\overline{R_{C_2}} is denoted by R_Ave)

    initial_density = [0.01, 0.01] # initial density for the first loop 
    for i in np.arange(resolution-1,-1,-1):
        
        # Simulation of population dynamics
        tend  = 100000
        tstep = 0.1
        time_array = np.arange(0, tend, tstep) # time for simulation
        
        rc_simulation_params = {'a': a2, 'h': h2, 'd': d2[i]}

        filename = make_filename('results/timeseries/RC/timeseries_RC', rc_simulation_params)
        predator_prey_partial = lambda time, density: predator_prey(density, time, rc_simulation_params)

        density_timeseries = simulate_and_save( # last 10% of the time series 
            filename=filename,
            ode_func=predator_prey_partial,
            x0=initial_density,
            t=time_array,
            params=rc_simulation_params
        )

        initial_density = density_timeseries[:,-1] * 1.01 # initial density for the next run
        
        # Check for NaN in timeseries
        if np.any(np.isnan(density_timeseries)):
            invasionrate_C1[:,i] = np.nan
            initial_density = [0.001,0.001]
        elif np.any(density_timeseries < 0):
            invasionrate_C1[:,i] = np.nan
            initial_density = [0.001,0.001]
        else:
            average_R_density = np.mean(density_timeseries[0, :])
            invasionrate_C1[:,i] = a1*average_R_density - d1


    invasionrate_C1[invasionrate_C1>0]=1
    invasionrate_C1[invasionrate_C1<0]=0 

    coexistence_gleanerbasic = invasionrate_C1*invasionrate_C2

    coexistence_gleanerbasic = coexistence_gleanerbasic*2 # limit cycle

    # delete timeseries for all d2, given a2 and h2
    filename = make_filename('results/timeseries/RC/timeseries_RC', rc_simulation_params) # only match until h_...
    filenameprefix = filename.rsplit('_d', 1)[0]  # remove d and everything after
    # delete all files starting with this prefix
    for file in os.listdir(os.path.dirname(filenameprefix)):
        if file.startswith(os.path.basename(filenameprefix)):
            os.remove(os.path.join(os.path.dirname(filenameprefix), file))
    
    return coexistence_gleanerbasic

def lin_lin_pred(params): # the parameters are the same as in the original model, the linearization is done in the function
    check_params(params, {'a1', 'a2', 'aP', 'h2', 'dP', 'hP', 'resolution'})

    a1 = params['a1']
    a2 = params['a2']
    aP = params['aP']
    hP = params['hP']
    h2 = params['h2']
    dP = params['dP']
    resolution = params['resolution']

    d1 = get_grid(a1, 0, resolution)
    d2 = get_grid(a2, h2, resolution)

    coexistence_lin_lin_predator = np.zeros([resolution,resolution])
    stability_lin_lin_predator = np.zeros([resolution,resolution])

    P_star = np.zeros([resolution,resolution])
    R_star = np.zeros([resolution,resolution])
    C1_star = np.zeros([resolution,resolution])
    C2_star = np.zeros([resolution,resolution])

    # check if the fixed point is positive everywhere
    for idx in range(resolution * resolution):
        i = idx // resolution
        j = idx % resolution

        aLin = (1-d2[j]*h2)*a2
        R_star[i,j] = (d1[i]-d2[j])/(a1-aLin)
        C1_star[i,j] = ( 1-R_star[i,j]- dP/aP * aLin ) /(a1-aLin)
        C2_star[i,j] = dP/aP-C1_star[i,j]
        P_star[i,j] = ( d1[i] * aLin - d2[j] * a1 ) / ( a1 - aLin )

        # Define symbolic variables
        R, C1, C2, P = sp.symbols('R C1 C2 P', real=True, positive=True)
        # Define the system symbolically (same structure as full_system)
        Rdot = ((1-R) - a1*C1 - aLin*C2)*R
        C1dot = (a1*R - d1[i] - aP*P/(1+aP*hP*(C1+C2)))*C1
        C2dot = (aLin*R - d2[j] - aP*P/(1+aP*hP*(C1+C2)))*C2
        Pdot = (aP*(C1+C2)/(1+aP*hP*(C1+C2)) - dP)*P
        # Create vector of state variables and equations
        state_vars = [R, C1, C2, P]
        myequations = [Rdot,
                     C1dot,
                     C2dot,
                     Pdot]

        # Compute Jacobian matrix
        mymatrix = sp.Matrix(myequations)
        jacobian = mymatrix.jacobian(state_vars)

        # To use it numerically, convert to a function:
        jacobian_func = sp.lambdify([state_vars], jacobian, 'numpy')
        eigenvalues = np.linalg.eigvals(jacobian_func([R_star[i,j], C1_star[i,j], C2_star[i,j], P_star[i,j]]))
        if all(eigenvalues.real < 0):
            stability_lin_lin_predator[i,j] = 1  # stable fixed point
        
   
    # Check for positive equilibrium (all species present)
    coexistence_positive = (C1_star > 0) & (C2_star > 0) & (P_star > 0) & (R_star > 0)
    coexistence_positive = coexistence_positive.astype(int)
    
    # Only count as coexistence if equilibrium is both positive AND stable
    coexistence_lin_lin_predator = coexistence_positive & (stability_lin_lin_predator == 1)
    coexistence_lin_lin_predator = coexistence_lin_lin_predator.astype(int) # stable fixed point equilibrium

    # no timeseries to delete
        
    return coexistence_lin_lin_predator, coexistence_positive, P_star, stability_lin_lin_predator

def lin_sat_pred(params):
    check_params(params, {'a1', 'a2', 'aP', 'h2', 'dP', 'resolution'})

    a1 = params['a1']
    a2 = params['a2']
    aP = params['aP']
    h2 = params['h2']
    dP = params['dP']
    resolution = params['resolution']

    d1 = get_grid(a1, 0, resolution)
    d2 = get_grid(a2, h2, resolution)

    coexistence_mixed = np.zeros([resolution,resolution])
    predator_mixed = np.zeros([resolution,resolution])

    # simulate for all d1 and d2 parameters with continuation (outward spiral)
    spiral_order = snail_order(resolution)

    initial_density = [0.01,0.01,0.01,0.01] # initial density for the first simulation
    
    total = len(spiral_order)

    for idx, (i, j) in enumerate(spiral_order):

        # Print progress every 50 iterations
        if idx % 50 == 0 or idx == total - 1:
            pid = os.getpid()
            try:
                n_open = len(psutil.Process(pid).open_files())
            except Exception:
                n_open = 'N/A'
            print(f"[Worker {pid}] lin_sat_pred progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%), open files: {n_open}")

        # Simulation of population dynamics
        tend  = 100000 
        tstep = 0.1
        time_array = np.arange(0, tend, tstep) # time for simulation
        simulation_params = {'a1':a1, 'a2':a2, 'aP':aP, 'h1':0, 'h2':h2, 'hP':0, 'd1':d1[i], 'd2':d2[j], 'dP':dP}

        filename = make_filename(prefix='results/timeseries/linsatpred/timeseries_RC1C2P_linsatpred', params=simulation_params)

        full_system_partial = lambda time, density: full_system(density, time, simulation_params)
        density_timeseries = simulate_and_save( # last 10% of the time series
            filename=filename,
            ode_func=full_system_partial,
            x0=initial_density,
            t=time_array,
            params=simulation_params
        )

        dens_Ave = np.mean(density_timeseries[:, :], axis=1) # average densities after transient dynamics 
        

        if all(dens_Ave > 10**-10):
            dens_CV = np.std(density_timeseries[:, :], axis=1) / dens_Ave
            
            if all(dens_CV < 0.01):
                coexistence_mixed[i,j] = 1 # fixed point
            else:
                coexistence_mixed[i,j] = 2 # cycle 
        if any(np.isnan(dens_Ave)):
            coexistence_mixed[i,j] = np.nan # extinction due to numerical issues

        # record predator density
        if dens_Ave[3] > 10**-10:
            predator_mixed[i,j] = dens_Ave[3]


        initial_density = density_timeseries[:, -1] * 1.01 # use the last density as initial density for the next simulation
        
        
    # delete timeseries
    folder_keys = ["a1", "h1", "a2", "h2", "aP", "hP", "dP"]
    folder_str = "_".join(f"{k}_{params[k]}" for k in folder_keys if k in params)
    folder_path = 'results/timeseries/linsatpred/timeseries_RC1C2P_linsatpred/' + folder_str
    print(folder_str)
    shutil.rmtree(folder_path)
        
    return coexistence_mixed, predator_mixed

def coexistence_general(params):

    check_params(params, {'a1', 'a2', 'aP', 'h1', 'h2', 'hP', 'dP', 'resolution'})

    a1 = params['a1']
    a2 = params['a2']
    aP = params['aP']
    h1 = params['h1']
    h2 = params['h2']
    hP = params['hP']
    dP = params['dP']
    resolution = params['resolution']

    d1 = get_grid(a1, h1, resolution)
    d2 = get_grid(a2, h2, resolution)

    coexistence_general = np.zeros([resolution,resolution])

    # simulate for all d1 and d2 parameters with continuation (outward spiral)
    spiral_order = snail_order(resolution)

    initial_density = [0.01,0.01,0.01,0.01] # initial density for the first simulation

    # simulate for all d1 and d2 parameters 
    for i,j in spiral_order:

        # Simulation of population dynamics
        initial_density = [0.01,0.01,0.01,0] # initial density

        tend  = 100000 # quite short
        tstep = 0.1
        time_array = np.arange(0, tend, tstep) # time for simulation
        simulation_params = {'a1':a1, 'a2':a2, 'aP':aP, 'h1':h1, 'h2':h2, 'hP':hP, 'd1':d1[i], 'd2':d2[j], 'dP':dP}

        filename = make_filename('results/timeseries/general/timeseries_RC1C2P_general', simulation_params)

        full_system_partial = lambda time, density: full_system(density, time, simulation_params)
        density_timeseries = simulate_and_save( # last 10% of the time series
            filename=filename, 
            ode_func=full_system_partial,
            x0=initial_density,
            t=time_array,
            params=simulation_params
        )

        average_density = np.mean(density_timeseries[:, :], axis=1) # average densities after transient dynamics
        
        if all(average_density > 10**-10):
            dens_CV = np.std(density_timeseries[:, :], axis=1) / average_density
            
            if all(dens_CV < 0.01):
                coexistence_general[i,j] = 1 # fixed point
            else:
                coexistence_general[i,j] = 2 # cycle  
        
        if any(np.isnan(average_density)):
            coexistence_general[i,j] = np.nan # extinction due to numerical issues

        initial_density = density_timeseries[:, -1] * 1.01 # use the last density as initial density for the next simulation

    # delete timeseries
    folder_keys = ["a1", "h1", "a2", "h2", "aP", "hP", "dP"]
    folder_str = "_".join(f"{k}_{params[k]}" for k in folder_keys if k in params)
    folder_path = 'results/timeseries/general/timeseries_RC1C2P_general/' + folder_str
    shutil.rmtree(folder_path)
    
    return coexistence_general

# perhaps lin_lin_sat and lin_sat_sat