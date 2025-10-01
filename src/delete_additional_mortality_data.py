import numpy as np
import glob
import os

# List all matrix files
matrix_files = glob.glob('results/matrices/matrices_*.npz')

for filename in matrix_files:
    data = np.load(filename, allow_pickle=True)
    data_dict = dict(data)
    keys_to_remove = [
        'coexistence_lin_sat_additional_mortality',
        'coexistence_sat_sat_additional_mortality',
        'statistics_lin_sat_additional_mortality',
        'statistics_sat_sat_additional_mortality'
    ]
    removed = False
    for key in keys_to_remove:
        if key in data_dict:
            del data_dict[key]
            removed = True
    if removed:
        np.savez(filename, **data_dict)
        print(f"Updated {filename}: removed additional mortality data.")
    else:
        print(f"No additional mortality data found in {filename}.")