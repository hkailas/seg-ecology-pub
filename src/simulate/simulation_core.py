# === src/simulate/simulation_core.py ===
"""
Core simulation functions and classes.
"""

import numpy as np
import time
from scipy.integrate import solve_ivp
from ..models.eco_evo_model import ode_model1, ode_model3
from ..config.parameters import SIMULATION_PARAMS

# Simulation parameters
t_start = SIMULATION_PARAMS['t_start']
t_max = SIMULATION_PARAMS['t_max']

eq_tsteps=SIMULATION_PARAMS['eq_tsteps']
eq_thresh=SIMULATION_PARAMS['eq_thresh']
chunk_size = SIMULATION_PARAMS['chunk_size']
verbose = SIMULATION_PARAMS['verbose']
evo_dyn = SIMULATION_PARAMS['evo_dyn']
min_thresh = SIMULATION_PARAMS['min_thresh']
abs_thresh = SIMULATION_PARAMS['abs_thresh']
method = SIMULATION_PARAMS.get('method', 'RK45')
maxstep = SIMULATION_PARAMS.get('maxstep', np.inf)
model_name = SIMULATION_PARAMS.get('model_name', 'model1')

# Model selection dictionary
ODE_MODELS = {
    'model1': ode_model1,
    'model3': ode_model3,
}


class chunk_ode_sol:
    """
    Custom class to mimic the behavior of SciPy's OdeResult returned by solve_ivp.
    """

    def __init__(self, t, y, sol=None, equilibrium=False):
        """
        Initialize a chuck_ode_sol instance.
        
        Parameters:
        -----------
        t : array_like
            Time points at which the solution was evaluated
        y : array_like
            Values of the solution at the time points in t. Shape: (n_state_vars, len(t))
        sol : callable, optional
            Function that evaluates the solution at arbitrary time points
        equilibrium : bool, optional
            Whether the system reached equilibrium during the integration
        """
        self.t = t
        self.y = y
        self.sol = sol
        self.equilibrium = equilibrium
        self.num_species = len(np.where(y[:len(y)//2] > 1e-6)[0]) 

def check_equilibrium(sol, eq_thresh, eq_tsteps, abs_thresh=abs_thresh, pop_thresh=1e-4):
    """
    Check if the system has reached equilibrium during the last eq_tsteps time units.
    
    For each species, compute the frequency (population / total population) and
    check that the absolute differences between consecutive time samples (sampled at unit intervals)
    are all below eq_thresh. In addition, check that the absolute differences between first 
    and last time points is less than abs_thresh. 
    
    Args:
        sol: A solve_ivp solution object with dense_output=True.
        eq_thresh (float): Threshold for frequency change.
        eq_tsteps (float): Time window over which frequencies must remain stable.
        abs_thresh (float): Threshold for absolute change in populations and traits between first and last time points.
        pop_thresh (float): Minimum population threshold for considering a species.

    Returns:
        bool: True if equilibrium condition is met, False otherwise.
        int: Number of species above the population threshold.
    """
    t_end = sol.t[-1]
    t_window_start = t_end - eq_tsteps
    # If not enough time has elapsed, cannot check equilibrium yet.
    if t_window_start < sol.t[0]:
        return False
    # Sample at integer time points over the window.
    sample_times = np.arange(t_window_start, t_end + 1)
    sol_values = sol.sol(sample_times)  # shape: (state_dim, number of samples)
    
    # Extract populations 
    num_pops = sol_values.shape[0] // 2

    # Extract trait values and population sizes
    x_list = sol_values[:num_pops]  # Extract all populations
    u_list = sol_values[num_pops:]  # Extract all traits

    # Check if the absolute differences between first and last time points are below the abs_threshold
    abs_flag = np.all(np.abs(x_list[:, 0] - x_list[:, -1]) <= abs_thresh) and np.all(np.abs(u_list[:, 0] - u_list[:, -1]) <= abs_thresh)

    # If populations are below the population threshold (extinct), we consider it as equilibrium. 
    if np.all(x_list[:, -1] < pop_thresh):
        return np.True_, 0
    sum_x = np.sum(x_list, axis=0)
    frequencies = x_list / sum_x  # shape: (num_pops, number of samples)
    
    # Compute absolute differences between consecutive samples.
    diffs_x = np.abs(np.diff(frequencies, axis=1))
    diffs_u = np.abs(np.diff(u_list, axis=1))

    # Check if all frequency and trait differences are below the threshold.
    return np.all(diffs_x < eq_thresh) and np.all(diffs_u < eq_thresh) and abs_flag, len(np.where(x_list[:,-1] >= pop_thresh)[0])


def run_simulation(initial_conditions, r_max, K, g, alpha, beta, beta_bully, sigma_a_squared, E, d, sigma, include_E, evo_dyn=True, min_thresh=min_thresh, t_start=t_start, t_max=t_max, eq_thresh=eq_thresh, abs_thresh=abs_thresh, eq_tsteps=eq_tsteps, chunk_size=chunk_size, verbose=verbose, store_sol=False, maxstep=maxstep, method=method, model_name=model_name, stop_at_equilibrium=True, fd=True):
    """
    Runs the ODE simulation using derivative-based equilibrium detection.

    fd=False replaces trait-dependent competition with uniform competition.

    When ``stop_at_equilibrium`` is false, equilibrium detection is disabled and
    integration continues to ``t_max``. The default preserves the historical
    early-stopping behavior.
    """
    if chunk_size < eq_tsteps:
        raise ValueError("chunk_size must be at least as large as eq_tsteps for equilibrium detection.")

    # Try to get logger, but don't fail if it doesn't exist
    try:
        from ..utils.mp_logging import get_current_worker_logger
        logger = get_current_worker_logger()
        if logger is None:
            # Fallback approaches
            from ..utils.logging_config import get_worker_logger
            logger = get_worker_logger()
            if logger is None:
                from ..utils.logging_config import get_chunk_logger
                logger = get_chunk_logger()
    except Exception:
        logger = None

    ode_func = ODE_MODELS.get(model_name)
    if ode_func is None: raise ValueError(f"Unknown model: {model_name}")

    # Extract Initial Conditions
    u0 = np.array(initial_conditions['traits'])
    x0 = np.array(initial_conditions['populations'])
    num_pops = len(x0)
    
    # Construct state vector
    y0 = np.concatenate([x0, u0])
    
    t_current = t_start
    y_current = y0.copy()
    
    # parameter tuple for Numba function
    args = (r_max, K, g, alpha, beta, beta_bully, sigma_a_squared, E, d, sigma, include_E, num_pops, evo_dyn, min_thresh, fd)
    
    chunk_count = 0
    equilibrium_reached = False
    
    # Storage
    t_full = [t_start] if store_sol else None
    y_full = [y0.reshape(-1,1)] if store_sol else None
    d_output_funcs = []    # To store dense output functions from each chunk.
    chunk_intervals = []   # To store the (t_start, t_end) for each chunk.
    while t_current < t_max:
        chunk_count += 1
        t_chunk_end = min(t_current + chunk_size, t_max)
        
        # Integration Step
        sol = solve_ivp(
            fun=lambda t, y: ode_func(t, y, *args),
            t_span=[t_current, t_chunk_end],
            y0=y_current,
            method=method,
            dense_output=True,
            max_step=maxstep,
            rtol=1e-9, atol=1e-11,first_step=1e-6 
        )
        
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        y_end = sol.y[:, -1]
        t_end = sol.t[-1]
        
        if store_sol:
            t_full.extend(sol.t[1:])
            y_full.append(sol.y[:, 1:])
            d_output_funcs.append(sol.sol)
            chunk_intervals.append((t_current, t_end))
        else:
            t_full = sol.t
            y_full = sol.y

        # EQUILIBRIUM CHECK
        if stop_at_equilibrium:
            equilibrium_result = check_equilibrium(
                sol,
                eq_thresh=eq_thresh,
                eq_tsteps=eq_tsteps,
                abs_thresh=abs_thresh,
            )
            if equilibrium_result[0]:
                y_current = y_end
                t_current = t_end
                break
        else:
            equilibrium_result = (
                False,
                len(np.where(y_end[:num_pops] >= 1e-4)[0]),
            )
            
        y_current = y_end
        t_current = t_end

    # Concatenate all chunks 
    if store_sol:
        t_full = np.array(t_full)
        y_full = np.hstack(y_full)
        # Global dense output function that uses the stored dense outputs.
        def global_dense_output(t_query, d_output_funcs=d_output_funcs, chunk_intervals=chunk_intervals):
            """
            Evaluate the solution at times t_query by calling the appropriate dense output functions
            from each integration chunk.

            Args:
                t_query (float or array_like): Time(s) at which to evaluate the solution.

            Returns:
                ndarray: Array of solution values corresponding to t_query.
            """
            t_query = np.atleast_1d(t_query)
            state_dim = y_full.shape[0]
            y_out = np.empty((state_dim, t_query.size))
            for (t_start_chunk, t_end_chunk), d_func in zip(chunk_intervals, d_output_funcs):
                indices = np.where((t_query >= t_start_chunk) & (t_query <= t_end_chunk))[0]
                if indices.size > 0:
                    y_out[:, indices] = d_func(t_query[indices])
            if y_out.shape[1] == 1:
                return y_out[:, 0]
            return y_out

        # Store the full solution and the global dense output function.
        solution = chunk_ode_sol(t=t_full, y=y_full, sol=global_dense_output)
    else:
        solution = chunk_ode_sol(t=t_full, y=y_full, sol=sol.sol)    
    
    # Set equilibrium flag
    solution.equilibrium = equilibrium_result[0]
    solution.num_species = equilibrium_result[1]

    return solution


def find_Nstar_given_u(u, E, params, init_pops, **sim_kwargs):
    """
    Find N* given u (that does not evolve). Evo_dyn is set to false. 
    """
    # Try to get logger
    try:
        from ..utils.mp_logging import get_current_worker_logger
        logger = get_current_worker_logger()
        if logger is None:
            # Fallback approaches
            from ..utils.logging_config import get_worker_logger
            logger = get_worker_logger()
            if logger is None:
                from ..utils.logging_config import get_chunk_logger
                logger = get_chunk_logger()
    except Exception:
        logger = None
    
    start_time = time.time()
    
    if logger:
        try:
            logger.debug(f"find_Nstar_given_u called with u={u.tolist()}, E={float(E):.4f}, init_pops={init_pops}")
        except Exception:
            logger.debug(f"find_Nstar_given_u called with u={u}, E={E}, init_pops={init_pops}")
    
    params = dict(params, E=E)

    # Set initial conditions
    IC = {'populations': init_pops,
           'traits': u.tolist()}

    sim_kwargs['evo_dyn'] = False  # Disable evolution dynamics for this simulation
    
    try:
        # Run simulation
        sol = run_simulation(IC, **params, **sim_kwargs)
        
        # Extract last populations
        y_eq = sol.y[:, -1]
        n = len(y_eq)//2
        N_star = y_eq[:n]
        
        elapsed_time = time.time() - start_time
        
        if logger:
            logger.debug(f"find_Nstar_given_u completed in {elapsed_time:.4f}s, "
                        f"N_star_sum={N_star.sum():.4f}, final_t={sol.t[-1]:.4f}")
        
        return N_star, sol.equilibrium
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        if logger:
            logger.error(f"find_Nstar_given_u failed after {elapsed_time:.4f}s: {str(e)}")
        raise
    
def species_br(E, params, IC, **sim_kwargs):
    # Ensure run_simulation is called with evo_dyn=True
    sim_kwargs['evo_dyn'] = True
    params = dict(params, E=E)
    sol = run_simulation(IC, **params, **sim_kwargs)
    y_eq = sol.y if len(sol.y.shape)==1 else sol.y[:, -1]
    n = len(y_eq)//2
    N_star = y_eq[:n]
    u_star = y_eq[n:]
    return N_star, u_star, sol.equilibrium
