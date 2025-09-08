import numpy as np
from scipy import integrate as integ
import os

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

def check_params(params, required): # check if all required parameters are present in the params dictionary
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing parameters: {', '.join(missing)}")

def get_grid(a, h, resolution): # mortality grid in dimension
    maxd = a/(1+h*a)
    return np.linspace(0, maxd, resolution + 2)[1:-1]

def simulate_and_save(filename, ode_func, x0, t, params): # generates, stores and returns the time series
    if os.path.exists(filename):
        data = np.load(filename)
        x = data['timeseries']
    else:
        x = integ.odeint(ode_func, x0, t, rtol=1e-14, atol=1e-12)
        np.savez(filename, timeseries=x, params=params)
    return x

def make_filename(prefix, params, extension = '.npz'): # generate filename based on prefix (e.g., 'timeseries_RC1C2_satsat') and parameters
    key_order = ["a1", "a2", "aP", "h1", "h2", "hP", "d1", "d2", "dP", "resolution"]
    values = [str(params[k]) for k in key_order if k in params]
    return f"{prefix}_{'_'.join(values)}{extension}"