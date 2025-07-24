# filepath: c:\Users\ToniDok\Documents\Lilla docs\GitRepos\relnonlin_predation\relnonlin_predation\src\analysis_tools.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integ
from functools import partial

from ode import full_system

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