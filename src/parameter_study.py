"""

Parameter Study - Simple interface for running multiple coexistence models on the same parameters

"""

from fileinput import filename
import numpy as np
from coexistence import lin_sat, lin_sat_pred, lin_lin_pred, coexistence_general
from utils import make_filename


class ParameterCase:
    """

    Run multiple coexistence models on a single parameter set.
    
    Example:
        parameterCase = ParameterStudy(params)
        results = parameterCase.run(['lin_sat', 'lin_sat_pred', 'lin_lin_pred'])
        parameterCase.save()
    
    """
    
    # Available models
    MODELS = {
        'lin_sat': lin_sat,
        'lin_sat_pred': lin_sat_pred,
        'lin_lin_pred': lin_lin_pred,
        'general': coexistence_general
    }
    
    def __init__(self, params):
        """

        Initialize a model set with given parameters.
        
        Args:
            params (dict): Parameter dictionary with model parameters
        
        """
        
        self.params = params
        self.results = {}
    
    def run(self, model_names, force_simulation=False):
        """

        Run specified models and return results.
        
        Args:
            model_names (list): List of model names to run (e.g., ['lin_sat', 'lin_sat_pred'])
        
        Returns:
            dict: Results for each model {model_name: result}

        """

        for model_name in model_names:
            if model_name not in self.MODELS: # wrong input
                print(f"Warning: Unknown model '{model_name}', skipping.")
                continue
            
            if any(model_name in key for key in self.results.get(model_name, {})) and not force_simulation: # already run
                print(f"Model '{model_name}' already run, skipping.")
            else:
                print(f"Running {model_name}...") # run simulations
                model_func = self.MODELS[model_name]
                matrices = model_func(self.params) # runs the coexistence function

                # Convert result to dictionary structure
                if isinstance(matrices, tuple):
                    if model_name == 'lin_sat_pred':
                        # Returns (coexistence, predator)
                        self.results[model_name] = {
                            f"coexistence_{model_name}": matrices[0],
                            f"predatordensity_{model_name}": matrices[1]
                        }
                    elif model_name == 'lin_lin_pred':
                        # Returns (coexistence, predator, stability)
                        self.results[model_name] = {
                            f"coexistence_{model_name}": matrices[0],
                            f"predatordensity_{model_name}": matrices[1],
                            f"stability_{model_name}": matrices[2]
                        }
                    else:
                        # Generic handling
                        self.results[model_name] = {
                            f"{model_name}_{idx}": matrix 
                            for idx, matrix in enumerate(matrices)
                        }
                else:
                    # Single matrix
                    self.results[model_name] = {
                        f"coexistence_{model_name}": matrices
                    }
        
        return self.results
    
    def save(self, filename=None, update_models=None):
        """

        Save all results to a single .npz file.
        If file exists, load existing data and only update specified models.
        
        Args:
            filename (str, optional): Output filename. If None, auto-generates from parameters.
            update_models (list, optional): List of model names to update. If None, updates all models in self.results.
        
        """

        if filename is None:
            filename = make_filename('results/matrices/matrices', self.params)
        
        # Determine which models to update
        if update_models is None:
            update_models = list(self.results.keys())  # Update all models that were run
        
        # Load existing data if file exists
        import os
        if os.path.exists(filename):
            with np.load(filename, allow_pickle=True) as existing:
                data_to_save = {key: existing[key] for key in existing.files}
        else:
            data_to_save = {'params': self.params}
        
        # Update only specified models
        for model_name in update_models:
            if model_name not in self.results:
                print(f"Warning: Model '{model_name}' not in results, skipping.")
                continue
            
            model_results = self.results[model_name]
    
            # Add all keys from this model to data_to_save
            for key, value in model_results.items():
                data_to_save[key] = value
        
        # Save to file
        with open(filename, "wb") as f:
            np.savez(f, **data_to_save)
        
        print(f"Results saved to {filename}")
        return filename
    
    def get_result(self, model_name):
        """
        
        Get result for a specific model.
        
        """
        
        return self.results.get(model_name)
    
    def update_results(self):
        """
        
        Update results from the saved files.

        """

        filename = make_filename('results/matrices/matrices', self.params)
        
        if not os.path.exists(filename):
            print(f"File {filename} does not exist, cannot update results.")
            return
        
        with np.load(filename, allow_pickle=True) as data:
            for model_name in self.MODELS.keys():
                # Find all keys belonging to this model
                model_keys = [key for key in data.files 
                            if model_name in key and key != 'params']
                
                if model_keys:
                    # Store as dictionary
                    self.results[model_name] = {
                        key: data[key] for key in model_keys
                    }
    
        print(f"Results updated from {filename}")