import numpy as np
import glob
import os

# Recursively find all .npz files in the results directory
for filename in glob.glob('results/**/*.npz', recursive=True):
    try:
        with np.load(filename) as data:
            pass  # File is OK
    except Exception as e:
        print(f"Corrupted file: {filename} ({e})")
        # Delete the corrupted file:
        os.remove(filename)
        print(f"Deleted: {filename}")