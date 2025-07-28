import numpy as np
from scipy import integrate as integ
from functools import partial
from ode import predator_prey, full_system
from utils import check_params
import os

def lin_sat(params):

    check_params(params, {'a1', 'a2', 'h2', 'resolution'})

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    resolution = params['resolution']

    maxd1 = a1
    maxd2 = a2/(1+h2*a2)
    d1 = np.linspace(0, maxd1, resolution + 2)[1:-1]
    d2 = np.linspace(0, maxd2, resolution + 2)[1:-1]
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
    for i in np.arange(0,resolution,1):
        tend  = 10000
        tstep = 0.1
        t  = np.arange(0,tend,tstep) # time for simulation
        x0 = [0.01,0.01] # initial density
        short_params = {'a':a2, 'h':h2, 'd':d2[i]}

        filename = f'results/timeseries/timeseries_RC_{short_params["a"]}_{short_params["h"]}_{short_params["d"]}.npz'
        if os.path.exists(filename):
            data = np.load(filename)
            x = data['timeseries']
        else:
            predator_prey_partial = lambda x, t: predator_prey(x, t, short_params)
            x  = integ.odeint(predator_prey_partial, x0, t,rtol = 10**(-14), atol = 10**(-12))
        
            np.savez(filename,
                    timeseries=x,
                    params=short_params)

        R_Ave = np.mean(x[2000:, 0]) # average resource density after transient dynamics 
        invasionrate_C1[:,i] = a1*R_Ave - d1
        
    invasionrate_C1[invasionrate_C1>0]=1
    invasionrate_C1[invasionrate_C1<0]=0 

    coexistence_gleanerbasic = invasionrate_C1*invasionrate_C2

    return coexistence_gleanerbasic

def lin_lin_pred(params):
    check_params(params, {'a1', 'a2', 'aP', 'h2', 'dP', 'resolution'})

    a1 = params['a1']
    a2 = params['a2']
    aP = params['aP']
    h2 = params['h2']
    dP = params['dP']
    resolution = params['resolution']

    maxd1 = a1
    maxd2 = a2/(1+h2*a2)
    d1 = np.linspace(0, maxd1, resolution + 2)[1:-1]
    d2 = np.linspace(0, maxd2, resolution + 2)[1:-1]
    
    coexistence_lin_lin_predator = np.zeros([resolution,resolution])

    P_star = np.zeros([resolution,resolution])
    R_star = np.zeros([resolution,resolution])
    C1_star = np.zeros([resolution,resolution])
    C2_star = np.zeros([resolution,resolution])

    # check if the fixed point is positive everywhere
    for i in np.arange(0,resolution,1):
        for j in np.arange(0,resolution,1):
            aLin = (1-d2[j]*h2)*a2
            R_star[i,j] = (d1[i]-d2[j])/(a1-aLin)
            C1_star[i,j] = ( 1-R_star[i,j]- dP/aP * aLin ) /(a1-aLin)
            C2_star[i,j] = dP/aP-C1_star[i,j]
            P_star[i,j] = ( d1[i] * aLin - d2[j] * a1 ) / ( a1 - aLin )

    coexistence_lin_lin_predator = (C1_star > 0) & (C2_star > 0) & (P_star > 0) & (R_star > 0)
    coexistence_lin_lin_predator = coexistence_lin_lin_predator.astype(int)

    return coexistence_lin_lin_predator

def lin_sat_pred(params):
    check_params(params, {'a1', 'a2', 'aP', 'h2', 'dP', 'resolution'})

    a1 = params['a1']
    a2 = params['a2']
    aP = params['aP']
    h2 = params['h2']
    dP = params['dP']
    resolution = params['resolution']

    maxd1 = a1
    maxd2 = a2/(1+h2*a2)
    d1 = np.linspace(0, maxd1, resolution + 2)[1:-1]
    d2 = np.linspace(0, maxd2, resolution + 2)[1:-1]
    
    coexistence_mixed = np.zeros([resolution,resolution])

    # simulate for all d1 and d2 parameters 
    for i in np.arange(0,resolution,1):
        for j in np.arange(0,resolution,1):

            x0 = [0.01,0.01,0.01,0.01] # initial density

            tend  = 10000 # quite short
            tstep = 0.1
            t  = np.arange(0,tend,tstep) # time for simulation
            short_params = {'a1':a1, 'a2':a2, 'aP':aP, 'h1':0, 'h2':h2, 'hP':0, 'd1':d1[i], 'd2':d2[j], 'dP':dP}
            
            filename = f'results/timeseries/timeseries_RC1C2P_linsatpred_{short_params["a1"]}_{short_params["a2"]}_{short_params["aP"]}_{short_params["h2"]}_{short_params["d1"]}_{short_params["d2"]}_{short_params["dP"]}.npz'
            
            if os.path.exists(filename):
                data = np.load(filename)
                x = data['timeseries']
            else:
                full_system_partial = lambda x, t: full_system(x, t, short_params)
                x  = integ.odeint(full_system_partial, x0, t, rtol = 10**(-14), atol = 10**(-12))
                np.savez(filename,
                        timeseries=x,
                        params=short_params)

            dens_Ave = np.mean(x[5000:, :], axis=0) # average resource density after transient dynamics 

            if all(dens_Ave > 0.001):
                coexistence_mixed[i,j] = 1   

    return coexistence_mixed

def sat_sat_pred(params):
    check_params(params, {'a1', 'a2', 'aP', 'h2', 'dP', 'resolution'})

    a1 = params['a1']
    a2 = params['a2']
    aP = params['aP']
    h2 = params['h2']
    dP = params['dP']
    resolution = params['resolution']

    maxd1 = a1
    maxd2 = a2/(1+h2*a2)
    d1 = np.linspace(0, maxd1, resolution + 2)[1:-1]
    d2 = np.linspace(0, maxd2, resolution + 2)[1:-1]
    
    coexistence_sat_sat_pred = np.zeros([resolution,resolution])

    # simulate for all d1 and d2 parameters 
    for i in np.arange(0,resolution,1):
        for j in np.arange(0,resolution,1):

            # convert functional response to saturating
            gamma = a1/a2 + h2*d1[i]
            sat_a = gamma*a2
            sat_h = h2/gamma
            sat_d = d1[i]

            x0 = [0.01,0.01,0.01,0.01] # initial density

            tend  = 10000 # quite short
            tstep = 0.1
            t  = np.arange(0,tend,tstep) # time for simulation
            short_params = {'a1':sat_a, 'a2':a2, 'aP':aP, 'h1':sat_h, 'h2':h2, 'hP':0, 'd1':sat_d, 'd2':d2[j], 'dP':dP}
            
            filename = f'results/timeseries/timeseries_RC1C2P_satsatpred_{short_params["a1"]}_{short_params["a2"]}_{short_params["aP"]}_{short_params["h1"]}_{short_params["h2"]}_{short_params["d1"]}_{short_params["d2"]}_{short_params["dP"]}.npz'
            
            if os.path.exists(filename):
                data = np.load(filename)
                x = data['timeseries']
            else:
                full_system_partial = lambda x, t: full_system(x, t, short_params)
                x  = integ.odeint(full_system_partial, x0, t, rtol = 10**(-14), atol = 10**(-12))
                np.savez(filename,
                        timeseries=x,
                        params=short_params)

            dens_Ave = np.mean(x[5000:, :], axis=0) # average resource density after transient dynamics

            if all(dens_Ave > 0.001):
                coexistence_sat_sat_pred[i,j] = 1   

    return coexistence_sat_sat_pred

def sat_sat(params):
    check_params(params, {'a1', 'a2', 'h2', 'resolution'})

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    resolution = params['resolution']

    maxd1 = a1
    maxd2 = a2/(1+h2*a2)
    d1 = np.linspace(0, maxd1, resolution + 2)[1:-1]
    d2 = np.linspace(0, maxd2, resolution + 2)[1:-1]
    
    coexistence_sat_sat = np.zeros([resolution,resolution])

    # simulate for all d1 and d2 parameters 
    for i in np.arange(0,resolution,1):
        for j in np.arange(0,resolution,1):

            # convert functional response to saturating
            gamma = a1/a2 + h2*d1[i]
            sat_a = gamma*a2
            sat_h = h2/gamma
            sat_d = d1[i]
            
            x0 = [0.01,0.01,0.01,0] # initial density

            tend  = 10000 # quite short
            tstep = 0.1
            t  = np.arange(0,tend,tstep) # time for simulation
            short_params = {'a1':sat_a, 'a2':a2, 'aP':0, 'h1':sat_h, 'h2':h2, 'hP':0, 'd1':sat_d, 'd2':d2[j], 'dP':0}

            filename = f'results/timeseries/timeseries_RC1C2_satsat_{short_params["a1"]}_{short_params["a2"]}_{short_params["h1"]}_{short_params["h2"]}_{short_params["d1"]}_{short_params["d2"]}.npz'

            if os.path.exists(filename):
                data = np.load(filename)
                x = data['timeseries']
            else:
                full_system_partial = lambda x, t: full_system(x, t, short_params)
                x  = integ.odeint(full_system_partial, x0, t, rtol = 10**(-14), atol = 10**(-12))

                np.savez(filename,
                    timeseries=x,
                    params=short_params)

            dens_Ave = np.mean(x[5000:, :], axis=0) # average resource density after transient dynamics 
            if all(dens_Ave > 0.001):
                coexistence_sat_sat[i,j] = 1   

    return coexistence_sat_sat

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

    maxd1 = a1/(1+h1*a1)
    maxd2 = a2/(1+h2*a2)
    d1 = np.linspace(0, maxd1, resolution + 2)[1:-1]
    d2 = np.linspace(0, maxd2, resolution + 2)[1:-1]
        
    coexistence_general = np.zeros([resolution,resolution])

    # simulate for all d1 and d2 parameters 
    for i in np.arange(0,resolution,1):
        for j in np.arange(0,resolution,1):

            x0 = [0.01,0.01,0.01,0] # initial density

            tend  = 10000 # quite short
            tstep = 0.1
            t  = np.arange(0,tend,tstep) # time for simulation
            short_params = {'a1':a1, 'a2':a2, 'aP':aP, 'h1':h1, 'h2':h2, 'hP':hP, 'd1':d1[i], 'd2':d2[j], 'dP':dP}

            filename = f'results/timeseries/timeseries_RC1C2P_general_{short_params["a1"]}_{short_params["a2"]}_{short_params["aP"]}_{short_params["h1"]}_{short_params["h2"]}_{short_params["hP"]}_{short_params["d1"]}_{short_params["d2"]}_{short_params["dP"]}.npz'

            if os.path.exists(filename):
                data = np.load(filename)
                x = data['timeseries']
            else:
                full_system_partial = lambda x, t: full_system(x, t, short_params)
                x  = integ.odeint(full_system_partial, x0, t, rtol = 10**(-14), atol = 10**(-12))
                np.savez(filename,
                        timeseries=x,
                        params=short_params)

            dens_Ave = np.mean(x[5000:, :], axis=0) # average resource density after transient dynamics 
            if all(dens_Ave > 0.001):
                coexistence_general[i,j] = 1   

    return coexistence_general