import numpy as np
from scipy import integrate as integ
from functools import partial
from ode import predator_prey, full_system
from utils import check_params

def lin_sat(params):

    check_params(params, ['a1', 'a2', 'h2', 'd1', 'd2', 'resolution'])

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    d1 = params['d1']
    d2 = params['d2']
    resolution = params['resolution']
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
        predator_prey_partial = partial(predator_prey, a=a2, h=h2, d=d2[i])
        x  = integ.odeint(predator_prey_partial, x0, t,rtol = 10**(-14), atol = 10**(-12))
        R_Ave = np.mean(x[2000:, 0]) # average resource density after transient dynamics 
        invasionrate_C1[:,i] = a1*R_Ave - d1
        
    invasionrate_C1[invasionrate_C1>0]=1
    invasionrate_C1[invasionrate_C1<0]=0 

    coexistence_gleanerbasic = invasionrate_C1*invasionrate_C2

    return coexistence_gleanerbasic

def lin_lin_pred(params):

    check_params(params, ['a1', 'a2', 'aP', 'h2', 'd1', 'd2', 'dP', 'resolution'])

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    d1 = params['d1']
    d2 = params['d2']
    aP = params['aP']
    dP = params['dP']
    resolution = params['resolution']
    
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
    
    return coexistence_lin_lin_predator

def lin_sat_pred(params):
    check_params(params, ['a1', 'a2', 'aP', 'h2', 'd1', 'd2', 'dP', 'resolution'])

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    d1 = params['d1']
    d2 = params['d2']
    aP = params['aP']
    dP = params['dP']
    resolution = params['resolution']
    
    coexistence_mixed = np.zeros([resolution,resolution])

    # simulate for all d1 and d2 parameters 
    for i in np.arange(0,resolution,1):
        for j in np.arange(0,resolution,1):

            x0 = [0.01,0.01,0.01,0.01] # initial density

            tend  = 10000 # quite short
            tstep = 0.1
            t  = np.arange(0,tend,tstep) # time for simulation
            full_system_partial = partial(full_system, a1=a1, a2=a2, aP=aP, h2=h2, d1=d1[i], d2=d2[j], dP=dP)
            x  = integ.odeint(full_system_partial, x0, t, rtol = 10**(-14), atol = 10**(-12))
            dens_Ave = np.mean(x[5000:, :], axis=0) # average resource density after transient dynamics 
            if all(dens_Ave > 0.001):
                coexistence_mixed[i,j] = 1   

    return coexistence_mixed