# BaseModel Design Document

## Current Problems

Looking at your three active models (`lin_sat`, `lin_lin_pred`, `lin_sat_pred`), they share:

1. **Grid creation**: All use `get_grid()` to create d1/d2 grids
2. **Matrix initialization**: All create coexistence matrices (and some create additional matrices)
3. **Simulation loop pattern**: Most iterate through grid points with continuation
4. **Equilibrium detection**: Most detect fixed points vs limit cycles using CV threshold
5. **Timeseries cleanup**: Most delete temporary files after completion
6. **Parameter extraction**: All extract params into local variables

But they differ in:
- **Computation method**: Some use analytical (lin_lin_pred), some use simulation (lin_sat, lin_sat_pred)
- **Iteration order**: lin_sat uses backwards iteration, others use spiral
- **What they return**: Some return 1 matrix, some return multiple
- **ODE system**: lin_sat uses predator_prey, others use full_system

---

## BaseModel Architecture

### Core Concept

Create an **abstract base class** that handles common patterns, with **template methods** that subclasses override for their specific behavior.

```python
from abc import ABC, abstractmethod
import numpy as np
from utils import check_params, get_grid, snail_order
import os
import shutil

class CoexistenceModel(ABC):
    """
    Base class for coexistence analysis models.
    
    Template Method Pattern:
    - compute() orchestrates the full workflow
    - Subclasses implement specific steps via abstract methods
    """
    
    def __init__(self, params):
        """Initialize model with parameters."""
        self.params = params
        self.validate_params()
        self.extract_params()
        self.initialize_grids()
        self.initialize_matrices()
    
    @abstractmethod
    def get_required_params(self):
        """Return set of required parameter names."""
        pass
    
    def validate_params(self):
        """Validate that all required params are present."""
        check_params(self.params, self.get_required_params())
    
    def extract_params(self):
        """Extract parameters into instance variables."""
        for key, value in self.params.items():
            setattr(self, key, value)
    
    def initialize_grids(self):
        """Create mortality rate grids (d1, d2)."""
        self.d1 = get_grid(self.a1, getattr(self, 'h1', 0), self.resolution)
        self.d2 = get_grid(self.a2, self.h2, self.resolution)
    
    @abstractmethod
    def initialize_matrices(self):
        """Initialize result matrices (coexistence, stability, etc.)."""
        pass
    
    @abstractmethod
    def compute_grid_point(self, i, j):
        """
        Compute coexistence for a single grid point (d1[i], d2[j]).
        
        Returns:
            dict: Results for this grid point (e.g., {'coexistence': 1, 'predator': 0.5})
        """
        pass
    
    def get_iteration_order(self):
        """
        Return the order in which to iterate through grid points.
        Override this for different iteration strategies.
        """
        # Default: spiral order for continuation
        return snail_order(self.resolution)
    
    def compute(self):
        """
        Main computation workflow (Template Method).
        
        1. Iterate through grid in specified order
        2. Compute each grid point
        3. Store results
        4. Cleanup
        5. Return results
        """
        iteration_order = self.get_iteration_order()
        
        # For models that need continuation
        self.initial_density = self.get_initial_density()
        
        for idx, (i, j) in enumerate(iteration_order):
            self.report_progress(idx, len(iteration_order))
            result = self.compute_grid_point(i, j)
            self.store_result(i, j, result)
        
        self.cleanup()
        return self.get_results()
    
    @abstractmethod
    def get_initial_density(self):
        """Return initial density for simulations."""
        pass
    
    def report_progress(self, idx, total):
        """Print progress updates (can override for custom reporting)."""
        if idx % 50 == 0 or idx == total - 1:
            print(f"{self.__class__.__name__} progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)")
    
    @abstractmethod
    def store_result(self, i, j, result):
        """Store the result from compute_grid_point into matrices."""
        pass
    
    def cleanup(self):
        """
        Cleanup temporary files.
        Override if your model creates temporary files.
        """
        pass
    
    @abstractmethod
    def get_results(self):
        """
        Return final results (matrices).
        Can return single matrix or tuple of matrices.
        """
        pass
```

---

## Example Implementation: LinSatPred

Here's how `lin_sat_pred` would look using the BaseModel:

```python
from base_model import CoexistenceModel
from ode import full_system
from utils import make_filename, simulate_and_save
import numpy as np
import shutil

class LinSatPredModel(CoexistenceModel):
    """
    Relative nonlinearity model with predation.
    Uses numerical simulation with spiral continuation.
    """
    
    def get_required_params(self):
        return {'a1', 'a2', 'aP', 'h2', 'dP', 'resolution'}
    
    def initialize_matrices(self):
        """Initialize coexistence and predator density matrices."""
        self.coexistence_matrix = np.zeros([self.resolution, self.resolution])
        self.predator_matrix = np.zeros([self.resolution, self.resolution])
    
    def get_initial_density(self):
        """Start with small positive densities for all species."""
        return [0.01, 0.01, 0.01, 0.01]
    
    def compute_grid_point(self, i, j):
        """Run simulation for this mortality combination."""
        # Setup simulation
        tend = 100000
        tstep = 0.1
        time_array = np.arange(0, tend, tstep)
        
        sim_params = {
            'a1': self.a1, 'a2': self.a2, 'aP': self.aP,
            'h1': 0, 'h2': self.h2, 'hP': 0,
            'd1': self.d1[i], 'd2': self.d2[j], 'dP': self.dP
        }
        
        filename = make_filename(
            prefix='results/timeseries/linsatpred/timeseries_RC1C2P_linsatpred',
            params=sim_params
        )
        
        full_system_partial = lambda time, density: full_system(density, time, sim_params)
        density_timeseries = simulate_and_save(
            filename=filename,
            ode_func=full_system_partial,
            x0=self.initial_density,
            t=time_array,
            params=sim_params
        )
        
        # Analyze results
        dens_Ave = np.mean(density_timeseries[:, :], axis=1)
        
        result = {'coexistence': 0, 'predator': 0}
        
        if all(dens_Ave > 10**-10):
            dens_CV = np.std(density_timeseries[:, :], axis=1) / dens_Ave
            
            if all(dens_CV < 0.01):
                result['coexistence'] = 1  # fixed point
            else:
                result['coexistence'] = 2  # cycle
        
        if any(np.isnan(dens_Ave)):
            result['coexistence'] = np.nan
        
        # Record predator density
        if dens_Ave[3] > 10**-10:
            result['predator'] = dens_Ave[3]
        
        # Update initial density for next iteration (continuation)
        self.initial_density = density_timeseries[:, -1] * 1.01
        
        return result
    
    def store_result(self, i, j, result):
        """Store coexistence and predator values in matrices."""
        self.coexistence_matrix[i, j] = result['coexistence']
        self.predator_matrix[i, j] = result['predator']
    
    def cleanup(self):
        """Delete temporary timeseries files."""
        folder_keys = ["a1", "h1", "a2", "h2", "aP", "hP", "dP"]
        folder_str = "_".join(f"{k}_{self.params[k]}" for k in folder_keys if k in self.params)
        folder_path = 'results/timeseries/linsatpred/timeseries_RC1C2P_linsatpred/' + folder_str
        
        try:
            shutil.rmtree(folder_path)
            print(f"Cleaned up: {folder_str}")
        except FileNotFoundError:
            pass  # Already deleted
    
    def get_results(self):
        """Return both coexistence and predator matrices."""
        return self.coexistence_matrix, self.predator_matrix


# Usage in main.py:
# model = LinSatPredModel(params)
# coexistence, predator = model.compute()
```

---

## Example Implementation: LinLinPred

This model is different because it uses **analytical computation** instead of simulation:

```python
from base_model import CoexistenceModel
import numpy as np
import sympy as sp

class LinLinPredModel(CoexistenceModel):
    """
    Linearized predation model.
    Uses analytical equilibrium calculation and stability analysis.
    """
    
    def get_required_params(self):
        return {'a1', 'a2', 'aP', 'h2', 'dP', 'hP', 'resolution'}
    
    def initialize_matrices(self):
        """Initialize coexistence, stability, and equilibrium density matrices."""
        self.coexistence_matrix = np.zeros([self.resolution, self.resolution])
        self.stability_matrix = np.zeros([self.resolution, self.resolution])
        self.P_star = np.zeros([self.resolution, self.resolution])
        self.R_star = np.zeros([self.resolution, self.resolution])
        self.C1_star = np.zeros([self.resolution, self.resolution])
        self.C2_star = np.zeros([self.resolution, self.resolution])
    
    def get_iteration_order(self):
        """Use simple grid iteration (not spiral)."""
        # Return all (i, j) pairs
        return [(i, j) for i in range(self.resolution) for j in range(self.resolution)]
    
    def get_initial_density(self):
        """Not needed for analytical model."""
        return None
    
    def compute_grid_point(self, i, j):
        """Compute equilibrium analytically and check stability."""
        # Linearized attack rate
        aLin = (1 - self.d2[j] * self.h2) * self.a2
        
        # Equilibrium densities
        R = (self.d1[i] - self.d2[j]) / (self.a1 - aLin)
        C1 = (1 - R - self.dP/self.aP * aLin) / (self.a1 - aLin)
        C2 = self.dP/self.aP - C1
        P = (self.d1[i] * aLin - self.d2[j] * self.a1) / (self.a1 - aLin)
        
        # Check if all positive
        coexists = (C1 > 0) and (C2 > 0) and (P > 0) and (R > 0)
        
        # Stability analysis via Jacobian
        stable = self._check_stability(R, C1, C2, P, aLin, i, j)
        
        return {
            'coexistence': int(coexists),
            'stability': int(stable),
            'R': R, 'C1': C1, 'C2': C2, 'P': P
        }
    
    def _check_stability(self, R, C1, C2, P, aLin, i, j):
        """Check stability using Jacobian eigenvalues."""
        # Symbolic variables
        R_sym, C1_sym, C2_sym, P_sym = sp.symbols('R C1 C2 P', real=True, positive=True)
        
        # System equations
        Rdot = ((1-R_sym) - self.a1*C1_sym - aLin*C2_sym)*R_sym
        C1dot = (self.a1*R_sym - self.d1[i] - self.aP*P_sym/(1+self.aP*self.hP*(C1_sym+C2_sym)))*C1_sym
        C2dot = (aLin*R_sym - self.d2[j] - self.aP*P_sym/(1+self.aP*self.hP*(C1_sym+C2_sym)))*C2_sym
        Pdot = (self.aP*(C1_sym+C2_sym)/(1+self.aP*self.hP*(C1_sym+C2_sym)) - self.dP)*P_sym
        
        # Jacobian
        mymatrix = sp.Matrix([Rdot, C1dot, C2dot, Pdot])
        jacobian = mymatrix.jacobian([R_sym, C1_sym, C2_sym, P_sym])
        jacobian_func = sp.lambdify([[R_sym, C1_sym, C2_sym, P_sym]], jacobian, 'numpy')
        
        # Eigenvalues at equilibrium
        eigenvalues = np.linalg.eigvals(jacobian_func([R, C1, C2, P]))
        
        return all(eigenvalues.real < 0)
    
    def store_result(self, i, j, result):
        """Store all equilibrium densities and stability."""
        self.coexistence_matrix[i, j] = result['coexistence']
        self.stability_matrix[i, j] = result['stability']
        self.R_star[i, j] = result['R']
        self.C1_star[i, j] = result['C1']
        self.C2_star[i, j] = result['C2']
        self.P_star[i, j] = result['P']
    
    def cleanup(self):
        """No files to cleanup for analytical model."""
        pass
    
    def get_results(self):
        """Return coexistence, predator equilibrium, and stability."""
        return self.coexistence_matrix, self.P_star, self.stability_matrix
```

---

## Example Implementation: LinSat

This model has a **different iteration pattern** (backwards):

```python
from base_model import CoexistenceModel
from ode import predator_prey
from utils import make_filename, simulate_and_save
import numpy as np
import os

class LinSatModel(CoexistenceModel):
    """
    Basic relative nonlinearity model (no predator).
    Uses invasion analysis with backwards iteration.
    """
    
    def get_required_params(self):
        return {'a1', 'a2', 'h2', 'resolution'}
    
    def initialize_matrices(self):
        """Initialize invasion rate matrices."""
        self.invasionrate_C1 = np.zeros([self.resolution, self.resolution])
        self.invasionrate_C2 = np.zeros([self.resolution, self.resolution])
    
    def get_iteration_order(self):
        """Backwards iteration for C1 invasion analysis."""
        return [(i, j) for i in range(self.resolution-1, -1, -1) for j in range(self.resolution)]
    
    def get_initial_density(self):
        """Two-species system."""
        return [0.01, 0.01]
    
    def compute(self):
        """Custom workflow: first compute C2 invasion, then C1 invasion."""
        # Step 1: C2 can invade C1 (analytical)
        self._compute_C2_invasion()
        
        # Step 2: C1 can invade C2 (simulation, needs special iteration)
        self._compute_C1_invasion()
        
        # Step 3: Combine
        coexistence = self.invasionrate_C1 * self.invasionrate_C2 * 2
        
        self.cleanup()
        return coexistence
    
    def _compute_C2_invasion(self):
        """Analytical calculation of C2 invasion."""
        for i in range(self.resolution):
            R_Eq = self.d1[i] / self.a1
            self.invasionrate_C2[i, :] = self.a2*R_Eq/(1+self.a2*self.h2*R_Eq) - self.d2
        
        self.invasionrate_C2[self.invasionrate_C2 > 0] = 1
        self.invasionrate_C2[self.invasionrate_C2 < 0] = 0
    
    def _compute_C1_invasion(self):
        """Simulation-based C1 invasion with backwards iteration."""
        initial_density = [0.01, 0.01]
        
        for i in range(self.resolution-1, -1, -1):
            # Simulate R-C2 system
            tend = 100000
            tstep = 0.1
            time_array = np.arange(0, tend, tstep)
            
            rc_params = {'a': self.a2, 'h': self.h2, 'd': self.d2[i]}
            filename = make_filename('results/timeseries/RC/timeseries_RC', rc_params)
            predator_prey_partial = lambda time, density: predator_prey(density, time, rc_params)
            
            density_timeseries = simulate_and_save(
                filename=filename,
                ode_func=predator_prey_partial,
                x0=initial_density,
                t=time_array,
                params=rc_params
            )
            
            # Check for issues
            if np.any(np.isnan(density_timeseries)) or np.any(density_timeseries < 0):
                self.invasionrate_C1[:, i] = np.nan
                initial_density = [0.001, 0.001]
            else:
                average_R = np.mean(density_timeseries[0, :])
                self.invasionrate_C1[:, i] = self.a1 * average_R - self.d1
                initial_density = density_timeseries[:, -1] * 1.01
        
        self.invasionrate_C1[self.invasionrate_C1 > 0] = 1
        self.invasionrate_C1[self.invasionrate_C1 < 0] = 0
    
    def compute_grid_point(self, i, j):
        """Not used - custom compute() method instead."""
        pass
    
    def store_result(self, i, j, result):
        """Not used - custom compute() method instead."""
        pass
    
    def cleanup(self):
        """Delete RC timeseries files."""
        rc_params = {'a': self.a2, 'h': self.h2, 'd': self.d2[0]}
        filename = make_filename('results/timeseries/RC/timeseries_RC', rc_params)
        filenameprefix = filename.rsplit('_d', 1)[0]
        
        try:
            for file in os.listdir(os.path.dirname(filenameprefix)):
                if file.startswith(os.path.basename(filenameprefix)):
                    os.remove(os.path.join(os.path.dirname(filenameprefix), file))
        except FileNotFoundError:
            pass
    
    def get_results(self):
        """Already returned in custom compute()."""
        pass
```

---

## Benefits of This Approach

### 1. **Reduced Code Duplication**
- Grid creation: 1 place instead of 3
- Continuation logic: 1 place
- Progress reporting: 1 place
- File cleanup pattern: 1 place

### 2. **Easier to Add New Models**
Just inherit and implement 5-6 methods instead of writing 100+ lines from scratch.

### 3. **Consistent Interface**
```python
# All models work the same way in main.py:
model = LinSatPredModel(params)
results = model.compute()

# vs current approach:
results = lin_sat_pred(params)
```

### 4. **Easier Testing**
Can mock individual methods to test components independently.

### 5. **Better Maintainability**
- Bug fix in grid creation? Fix once in base class
- Want to add progress bar? Update base class only
- Need to change continuation strategy? Override one method

---

## Migration Path

### Phase 1: Create BaseModel (2-3 hours)
1. Create `src/models/base.py` with `CoexistenceModel` class
2. Implement common methods
3. Write unit tests for base class

### Phase 2: Refactor One Model (2 hours)
1. Start with simplest model (maybe `lin_lin_pred` since it's analytical)
2. Create `src/models/lin_lin_pred.py` inheriting from base
3. Test that it produces identical results to original

### Phase 3: Refactor Remaining Models (3-4 hours)
1. Refactor `lin_sat_pred`
2. Refactor `lin_sat` (most complex due to custom workflow)
3. Update `main.py` to use new models

### Phase 4: Cleanup (1 hour)
1. Delete old `coexistence.py`
2. Update imports
3. Add documentation

---

## Questions to Consider

1. **How flexible should BaseModel be?**
   - Very flexible (more abstract methods) = more boilerplate per model
   - Less flexible (more built-in) = easier to use but less customizable

2. **Should we support the "general" model easily?**
   - Could make `h1` and `hP` parameters in base class
   - Would make it trivial to implement general model

3. **What about the continuation strategy?**
   - Current: Each model stores `initial_density` for next iteration
   - Could: Make continuation a first-class concept in base class

4. **Testing strategy?**
   - Should we create fixtures with small grids (e.g., 5x5) for fast tests?
   - Compare new vs old implementation outputs?

---

## What do you think?

Does this approach make sense for your use case? Some questions:

1. **Is this level of abstraction comfortable for you?** (Python inheritance, abstract methods, etc.)
2. **Would you prefer to start with just extracting common utilities first?** (smaller step)
3. **Are there other patterns in the models I missed?**
4. **Do you want me to implement one complete example to see it in action?**

Let me know what direction you'd like to go!
