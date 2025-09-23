import numpy as np
from scipy import integrate as integ
import os

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
    key_order = ["a","h","d","a1", "a2", "aP", "h1", "h2", "hP", "d1", "d2", "dP", "resolution"]
    values = [str(params[k]) for k in key_order if k in params]
    
    filename = f"{prefix}_{'_'.join(values)}{extension}"
    # If running from 'notebooks', prepend '../'
    
    if os.path.basename(cwd) == "notebooks" or os.path.basename(cwd) == "src":
        filename = os.path.join("..\\", filename)
    return filename
