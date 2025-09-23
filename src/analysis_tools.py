# filepath: c:\Users\ToniDok\Documents\Lilla docs\GitRepos\relnonlin_predation\relnonlin_predation\src\analysis_tools.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integ
from functools import partial

from ode import full_system

def get_equilibriums(a1, a2, aP, h2, d1, d2, dP): # compute the two equilibriums of the full system (lin_sat_pred)

    G_p = (a1 - a2 - (d1 - d2) * h2 * a2) / (a1 * h2 * a2)
    G_q = -(d1 - d2) / (a1 * h2 * a2)
    G1_Rstar = -G_p / 2 + np.sqrt(G_p**2 / 4 - G_q) # "positive" root
    G2_Rstar = -G_p / 2 - np.sqrt(G_p**2 / 4 - G_q) # "negative" root
    G1_C2 = (G1_Rstar + a1 * dP / aP - 1) / (a1 - a2 / (1 + h2 * a2 * G1_Rstar))
    G2_C2 = (G2_Rstar + a1 * dP / aP - 1) / (a1 - a2 / (1 + h2 * a2 * G2_Rstar))
    G1_C1 = dP / aP - G1_C2
    G2_C1 = dP / aP - G2_C2
    G1_P = (a1 * G1_Rstar - d1) / aP
    G2_P = (a1 * G2_Rstar - d1) / aP
    x0_G1 = np.array([G1_Rstar, G1_C1, G1_C2, G1_P]) # "positive" root
    x0_G2 = np.array([G2_Rstar, G2_C1, G2_C2, G2_P]) # "negative" root
    return x0_G1, x0_G2

def plot_individual_case(params, x0 = [0.01, 0.01, 0.01, 0.01], tend=10000, tstep=0.1, tstart=9000):
    t = np.arange(0, tend, tstep)
    full_system_partial = lambda x, t: full_system(x, t, params)
    x = integ.odeint(full_system_partial, x0, t, rtol=1e-14, atol=1e-12)
    plt.figure(figsize=(3,3))
    plt.plot(t[tstart:], x[tstart:,0], '-g', label='R')
    plt.plot(t[tstart:], x[tstart:,1], '-b', label='C1')
    plt.plot(t[tstart:], x[tstart:,2], '-r', label='C2')
    plt.plot(t[tstart:], x[tstart:,3], '-k', label='P')
    plt.xlabel('time')
    plt.ylabel('density')
    plt.legend()
    plt.title('Population dynamics')
    plt.ylim(0,1)
    plt.show()