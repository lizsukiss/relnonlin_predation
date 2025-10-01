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

def simulate_and_save(filename, ode_func, x0, t, params, force_simulate=False): # generates, stores and returns the time series
    
    if os.path.exists(filename) and not force_simulate:
        data = np.load(filename)
        x = data['timeseries']
    else:
        x = integ.odeint(ode_func, x0, t, rtol=1e-14, atol=1e-12)
        x = x[-int(np.floor(len(x)/10)):] # keep only last 10% of the time series
        np.savez(filename, timeseries=x, params=params)
    return x

def make_filename(prefix, params, extension = '.npz'): # generate filename based on prefix (e.g., 'timeseries_RC1C2_satsat') and parameters

    if 'd1' in params and 'd2' in params:
        # Main parameters for folder name
        folder_keys = ["a1", "h1", "a2", "h2", "aP", "hP", "dP"]
        folder_str = "_".join(f"{k}_{params[k]}" for k in folder_keys if k in params)
        folder_path = os.path.join(prefix, folder_str)
        os.makedirs(folder_path, exist_ok=True)
        
        # Only d1 and d2 for file name
        file_keys = ["d1", "d2"]
        file_str = "_".join(f"{k}_{params[k]}" for k in file_keys if k in params)
        filename = f"{file_str}{extension}"

        # Full path
        full_path = os.path.join(folder_path, filename)
        # If running from 'notebooks' or 'src', prepend '../'
        cwd = os.getcwd()
        if os.path.basename(cwd) in ["notebooks", "src"]:
            full_path = os.path.join("..", full_path)
        return full_path
    else:
        # Fallback to original behavior if d1 and d2 are not present
        key_order = ["a","h","d","a1", "a2", "aP", "h1", "h2", "hP", "d1", "d2", "dP", "resolution"]
        param_strs = [f"{k}_{params[k]}" for k in key_order if k in params]
        filename = f"{prefix}_{'_'.join(param_strs)}{extension}"
        # If running from 'notebooks', prepend '../'
        cwd = os.getcwd()
        if os.path.basename(cwd) == "notebooks" or os.path.basename(cwd) == "src":
            filename = os.path.join("..\\", filename)
        return filename

def snail_order(n):
    order = []
    x = n // 2
    y = n // 2
    if n % 2 == 0:
        x -= 1
        y -= 1
    dx = [0, 1, 0, -1]  # up, right, down, left, up, right
    dy = [1, 0, -1, 0]  # clockwise spiral
    dir = 0  # directional changes
    step_size = 1
    index = 0

    while index < n * n:
        for _ in range(2):
            for _ in range(step_size):
                if index >= n * n:
                    return np.array(order)
                order.append((x, y))
                x += dx[dir]
                y += dy[dir]
                index += 1
            dir = (dir + 1) % 4
        step_size += 1
    return np.array(order)