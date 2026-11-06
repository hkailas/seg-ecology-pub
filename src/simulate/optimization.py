"""
Optimization functions for finding best responses and equilibria.
"""

import multiprocessing as mp
import inspect
import platform
import signal
import threading
import queue
import warnings

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from ..config.parameters import (
    SIMULATION_PARAMS,
    MODEL_PARAMS,
    ECONOMIC_PARAMS,
)
from .simulation_core import find_Nstar_given_u, run_simulation, species_br
from src.result_data_structures.simulation_result_data_types import SimulationResult
from ..models.economic_model import Q_leader, Q


def _fixed_trait_payoff_worker(args):
    """Evaluate one fixed-trait ecological equilibrium and leader payoff."""
    (
        E,
        u_fixed,
        initial_populations,
        model_params,
        economic_params,
        simulation_params,
        configured_num_pops,
    ) = args
    N_star, equilibrium = find_Nstar_given_u(
        u_fixed,
        float(E),
        model_params,
        initial_populations,
        **simulation_params,
    )
    N_star = np.asarray(N_star, dtype=float)
    if not bool(equilibrium):
        return N_star, False, np.nan

    payoff = Q_leader(
        float(E),
        economic_params["L"],
        economic_params["p"],
        economic_params["delta"],
        economic_params["alpha_leader"],
        float(np.sum(N_star)),
        N_star=np.atleast_2d(N_star),
        u_star=np.atleast_2d(u_fixed),
        c1=economic_params.get("c1", 0),
        c4=economic_params.get("c4", 0),
        c6=economic_params.get("c6", 0),
        configured_num_pops=configured_num_pops,
    )
    return N_star, True, float(np.asarray(payoff).squeeze())

def evaluate_fixed_trait_payoff_curve(
    E_values,
    u_fixed,
    initial_populations,
    model_params,
    economic_params,
    simulation_params,
    *,
    configured_num_pops,
    num_cpus=1,
):
    """Evaluate leader payoff across E with traits fixed and ecology re-solved."""
    E_values = np.asarray(E_values, dtype=float)
    u_fixed = np.asarray(u_fixed, dtype=float)
    initial_populations = np.asarray(initial_populations, dtype=float)
    if E_values.ndim != 1:
        raise ValueError("E_values must be one-dimensional.")
    if u_fixed.ndim != 1 or initial_populations.ndim != 1:
        raise ValueError(
            "Fixed traits and initial populations must be one-dimensional."
        )
    if u_fixed.shape != initial_populations.shape:
        raise ValueError(
            "Fixed traits and initial populations must have matching shapes."
        )
    if num_cpus < 1:
        raise ValueError("num_cpus must be at least 1.")

    accepted_run_arguments = set(inspect.signature(run_simulation).parameters)
    reserved_arguments = {"initial_conditions", "E", *model_params}
    simulation_kwargs = {
        name: value
        for name, value in simulation_params.items()
        if name in accepted_run_arguments and name not in reserved_arguments
    }
    tasks = [
        (
            float(E),
            u_fixed,
            initial_populations,
            dict(model_params),
            dict(economic_params),
            simulation_kwargs,
            configured_num_pops,
        )
        for E in E_values
    ]
    if not tasks:
        rows = []
    elif num_cpus == 1:
        rows = [_fixed_trait_payoff_worker(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=num_cpus) as pool:
            rows = pool.map(_fixed_trait_payoff_worker, tasks)

    if rows:
        N_star = np.vstack([row[0] for row in rows])
        equilibrium = np.asarray([row[1] for row in rows], dtype=bool)
        payoff = np.asarray([row[2] for row in rows], dtype=float)
    else:
        N_star = np.empty((0, len(u_fixed)), dtype=float)
        equilibrium = np.empty(0, dtype=bool)
        payoff = np.empty(0, dtype=float)
    return {
        "E": E_values.copy(),
        "payoff": payoff,
        "N_star": N_star,
        "equilibrium": equilibrium,
        "u_fixed": u_fixed.copy(),
        "initial_populations": initial_populations.copy(),
    }

def timeout_handler(signum, frame):
    """Signal handler for timeout (Unix/Linux only)"""
    raise TimeoutError("Function execution timed out")

def _timeout_with_signal(args, timeout_seconds):
    """
    Signal-based timeout implementation (Linux/Unix/macOS).
    More efficient and can truly interrupt computations.
    """
    
    E_val, IC = args
    params = MODEL_PARAMS
    
    # Set up timeout signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        # Call the potentially long-running function
        N_star, u_star, equilibrium = species_br(E_val, params, IC)
        
        # Cancel the alarm if we finished successfully
        signal.alarm(0)
        
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        
        return SimulationResult(N_star, u_star, equilibrium)
        
    except TimeoutError:
        # Cancel the alarm
        signal.alarm(0)
        
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        
        # Return identifiable dummy result with timeout information
        timeout_result = {
            'timeout': True, 
            'E': E_val, 
            'IC': IC, 
            'timeout_seconds': timeout_seconds,
            'timeout_method': 'signal'
        }
        return SimulationResult(None, None, timeout_result)
        
    except RuntimeError as e:
        # Cancel the alarm and restore handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
        # Integration errors should be returned as error results for healing
        error_result = {
            'error': True,
            'E': E_val,
            'IC': IC,
            'error_type': 'RuntimeError',
            'message': str(e),
            'timeout_method': 'signal'
        }
        return SimulationResult(None, None, error_result)
        
    except Exception as e:
        # Cancel the alarm and restore handler in case of other exceptions
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
        # Re-raise the original exception
        raise e

def _timeout_with_threading(args, timeout_seconds):
    """
    Threading-based timeout implementation (Windows/fallback).
    Compatible with all platforms but cannot force-kill threads.
    """
    
    E_val, IC = args
    params = MODEL_PARAMS
    
    # Create a queue to store the result
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def worker():
        """Worker function that runs species_br in a separate thread."""
        try:
            N_star, u_star, equilibrium = species_br(E_val, params, IC)
            result_queue.put(SimulationResult(N_star, u_star, equilibrium))
        except RuntimeError as e:
            # Integration errors should be returned as error results for healing
            error_result = {
                'error': True,
                'E': E_val,
                'IC': IC,
                'error_type': 'RuntimeError',
                'message': str(e),
                'timeout_method': 'threading'
            }
            result_queue.put(SimulationResult(None, None, error_result))
        except Exception as e:
            # Other exceptions are put in exception_queue to be raised
            exception_queue.put(e)
    
    # Start the worker thread
    thread = threading.Thread(target=worker)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    
    # Wait for the thread to complete or timeout
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Timeout occurred - thread is still running
        # Note: We can't force-kill the thread, but it will die when the process ends
        timeout_result = {
            'timeout': True, 
            'E': E_val, 
            'IC': IC, 
            'timeout_seconds': timeout_seconds,
            'timeout_method': 'threading'
        }
        return SimulationResult(None, None, timeout_result)
    else:
        # Thread completed - check for results or exceptions
        try:
            # Check if there was an exception
            if not exception_queue.empty():
                exception = exception_queue.get_nowait()
                raise exception
            
            # Get the result
            if not result_queue.empty():
                return result_queue.get_nowait()
            else:
                # Thread finished but no result (shouldn't happen)
                error_result = {
                    'error': True, 
                    'E': E_val, 
                    'IC': IC, 
                    'message': 'Thread completed without result',
                    'timeout_method': 'threading'
                }
                return SimulationResult(None, None, error_result)
                
        except queue.Empty:
            # No result available (shouldn't happen if thread finished)
            error_result = {
                'error': True, 
                'E': E_val, 
                'IC': IC, 
                'message': 'No result from completed thread',
                'timeout_method': 'threading'
            }
            return SimulationResult(None, None, error_result)

def find_best_E_given_u_IC(
    u,
    initial_populations,
    model_params,
    economic_params,
    simulation_params,
    *,
    configured_num_pops,
    bounds=(0.0, 1.0),
    xatol=1e-5,
):
    """Strictly optimize the leader response with species traits held fixed.

    Configuration is supplied explicitly. Optimization and final ecological
    equilibrium failures are raised so callers can record them unambiguously.
    """
    u = np.asarray(u, dtype=float)
    initial_populations = np.asarray(initial_populations, dtype=float)
    if u.ndim != 1 or initial_populations.ndim != 1:
        raise ValueError("u and initial_populations must be one-dimensional.")
    if len(u) != len(initial_populations):
        raise ValueError("u and initial_populations must have equal lengths.")
    # Larger configured systems may be collapsed into at most four trait groups.
    if configured_num_pops <= 4 and len(u) != configured_num_pops:
        raise ValueError(
            "u and initial_populations must match configured_num_pops."
        )
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(initial_populations)):
        raise ValueError("u and initial_populations must contain finite values.")

    lower, upper = (float(value) for value in bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must contain two finite, increasing values.")
    if xatol <= 0:
        raise ValueError("xatol must be positive.")

    run_parameters = inspect.signature(run_simulation).parameters
    simulation_kwargs = {
        key: value
        for key, value in simulation_params.items()
        if key in run_parameters
        and key != "initial_conditions"
        and key not in model_params
    }

    def ecological_equilibrium(E):
        return find_Nstar_given_u(
            u,
            float(E),
            model_params,
            initial_populations.tolist(),
            **simulation_kwargs,
        )

    def objective_function(E):
        N_star, equilibrium = ecological_equilibrium(E)
        if not bool(equilibrium):
            return np.inf

        N_star = np.asarray(N_star, dtype=float)
        payoff = Q_leader(
            float(E),
            economic_params["L"],
            economic_params["p"],
            economic_params["delta"],
            economic_params["alpha_leader"],
            float(np.sum(N_star)),
            N_star=np.atleast_2d(N_star),
            u_star=np.atleast_2d(u),
            c1=economic_params.get("c1", 0),
            c4=economic_params.get("c4", 0),
            c6=economic_params.get("c6", 0),
            configured_num_pops=configured_num_pops,
        )
        payoff_scalar = float(np.asarray(payoff).squeeze())
        if not np.isfinite(payoff_scalar):
            raise RuntimeError("Leader payoff is not finite.")
        return -payoff_scalar

    optimization_result = minimize_scalar(
        objective_function,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": float(xatol)},
    )
    if not optimization_result.success or not np.isfinite(optimization_result.fun):
        message = getattr(optimization_result, "message", "unknown failure")
        raise RuntimeError(f"Leader optimization failed: {message}")

    E_star = float(optimization_result.x)
    Q_star = -float(optimization_result.fun)
    N_star, equilibrium = ecological_equilibrium(E_star)
    if not bool(equilibrium):
        raise RuntimeError("Ecological equilibrium failed at optimized E.")

    return E_star, Q_star, True, np.asarray(N_star, dtype=float)

def find_best_u_best_N_given_E_IC(args, timeout_seconds=None):
    """
    Process a single E, IC pair to find optimal u and N with platform-adaptive timeout protection.

    Automatically selects the best timeout method based on platform:
    - Linux/Unix/macOS: Signal-based timeout (more efficient, true interruption)
    - Windows: Threading-based timeout (compatible fallback)

    Parameters:
    -----------
    args : tuple
        (E_val, IC) pair
    timeout_seconds : int, optional
        Maximum time in seconds to allow for computation. If None, uses value from SIMULATION_PARAMS.

    Returns:
    --------
    SimulationResult
        - If successful: ``SimulationResult(N_star, u_star, equilibrium)``
        - If timeout: a result with ``N_star`` and ``u_star`` set to ``None``
          and timeout metadata in ``equilibrium``.
    """
    
    # Use timeout from config if not specified
    if timeout_seconds is None:
        timeout_seconds = SIMULATION_PARAMS.get('timeout_seconds', 3600)
    
    # Platform-adaptive timeout method selection
    use_signal_timeout = (
        hasattr(signal, 'SIGALRM') and 
        platform.system() != 'Windows'
    )
    
    if use_signal_timeout:
        # Use efficient signal-based timeout on Unix-like systems
        try:
            return _timeout_with_signal(args, timeout_seconds)
        except AttributeError:
            # Fallback to threading if signal setup fails
            return _timeout_with_threading(args, timeout_seconds)
    else:
        # Use threading-based timeout on Windows or when signals unavailable
        return _timeout_with_threading(args, timeout_seconds)

def payoff_at_species_br(
    E_values_order,
    N_star_list,
    u_star_list,
    equilibrium_list,
    bifurcation_labels,
    *,
    configured_num_pops,
    economic_params=None,
):
    """Evaluate leader payoff for every species-BR initial condition.

    Population, trait, E, branch, equilibrium, and payoff arrays remain aligned
    with the input records. Non-equilibrium records receive a NaN payoff so
    they can be retained without being treated as valid species best responses.
    """
    E_values = np.asarray(E_values_order, dtype=float)
    equilibrium = np.asarray(equilibrium_list, dtype=bool)
    branches = np.asarray(bifurcation_labels, dtype=object)
    lengths = {
        len(E_values),
        len(N_star_list),
        len(u_star_list),
        len(equilibrium),
        len(branches),
    }
    if len(lengths) != 1:
        raise ValueError(
            "E values, populations, traits, equilibrium flags, and branch "
            "labels must have matching lengths."
        )

    try:
        if len(E_values):
            N_star_all = np.vstack(
                [np.asarray(value, dtype=float) for value in N_star_list]
            )
            u_star_all = np.vstack(
                [np.asarray(value, dtype=float) for value in u_star_list]
            )
        else:
            N_star_all = np.empty((0, 0), dtype=float)
            u_star_all = np.empty((0, 0), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Population and trait records must each have consistent shapes."
        ) from error
    if N_star_all.shape != u_star_all.shape:
        raise ValueError(
            "Population and trait arrays must have matching shapes."
        )

    payoffs = np.full(len(E_values), np.nan, dtype=float)
    non_equilibrium_count = int(np.count_nonzero(~equilibrium))
    if non_equilibrium_count:
        warnings.warn(
            f"{non_equilibrium_count} species-BR initial conditions did not "
            "reach equilibrium; their payoffs were set to NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    equilibrium_indices = np.flatnonzero(equilibrium)
    if len(equilibrium_indices):
        N_star = N_star_all[equilibrium_indices]
        u_star = u_star_all[equilibrium_indices]

        economic = ECONOMIC_PARAMS if economic_params is None else economic_params
        equilibrium_payoffs = Q_leader(
            E_values[equilibrium_indices],
            economic["L"],
            economic["p"],
            economic["delta"],
            economic["alpha_leader"],
            np.sum(N_star, axis=1),
            N_star=N_star,
            u_star=u_star,
            c1=economic["c1"],
            c4=economic["c4"],
            c6=economic["c6"],
            configured_num_pops=configured_num_pops,
        )
        payoffs[equilibrium_indices] = np.asarray(
            equilibrium_payoffs,
            dtype=float,
        )

    finite_payoff_indices = np.flatnonzero(np.isfinite(payoffs))
    if len(finite_payoff_indices):
        finite_payoffs = payoffs[finite_payoff_indices]
        stackelberg_index = int(
            finite_payoff_indices[int(np.argmax(finite_payoffs))]
        )
        E_stackelberg = float(E_values[stackelberg_index])
        payoff_stackelberg = float(payoffs[stackelberg_index])
        N_star_stackelberg = N_star_all[stackelberg_index].copy()
        u_star_stackelberg = u_star_all[stackelberg_index].copy()
        branch_stackelberg = str(branches[stackelberg_index])
    else:
        stackelberg_index = None
        E_stackelberg = None
        payoff_stackelberg = None
        N_star_stackelberg = None
        u_star_stackelberg = None
        branch_stackelberg = None

    return {
        "E": E_values,
        "payoff": payoffs,
        "branch": branches,
        "equilibrium": equilibrium,
        "N_star": N_star_all,
        "u_star": u_star_all,
        "E_stackelberg": E_stackelberg,
        "payoff_stackelberg": payoff_stackelberg,
        "N_star_stackelberg": N_star_stackelberg,
        "u_star_stackelberg": u_star_stackelberg,
        "branch_stackelberg": branch_stackelberg,
        "stackelberg_record_index": stackelberg_index,
    }
