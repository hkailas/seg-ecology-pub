"""
Compute boundary trait values (W+ and W-) using the Lambert W function.

The growth function G_i has roots at specific trait values when v = u, 
assuming very small population (logistic term ≈ 1). This module finds 
these boundary values which define the valid trait interval for simulations.
"""

import numpy as np
from scipy.special import lambertw
from src.config.parameters import MODEL_PARAMS


def compute_boundary_traits(E):
    """
    Compute the boundary trait values W+ and W- for a given extraction rate E.
    
    The equation for the roots using the Lambert W function is:
    u = (alpha/E) - beta - (1/g) * W(-alpha*d*g/(E*r_max) * exp((alpha*g/E) - beta*g))
    
    
    Returns
    -------
    W_minus : float
        Lower boundary trait value (using W_{-1} branch)
    W_plus : float
        Upper boundary trait value (using W_0 branch)
    
    Notes
    -----
    The Lambert W function has two real branches for arguments in [-1/e, 0):
    - W_0 (k=0): principal branch, returns values in [-1, ∞)
    - W_{-1} (k=-1): secondary branch, returns values in (-∞, -1]
    
    These correspond to the two roots of the growth function G = 0.
    """
    # Extract parameters
    alpha = MODEL_PARAMS['alpha']
    beta = MODEL_PARAMS['beta']
    g = MODEL_PARAMS['g']
    d = MODEL_PARAMS['d']
    r_max = MODEL_PARAMS['r_max']
    
    # Validate E
    if E <= 0:
        raise ValueError(f"Extraction rate E must be positive, got {E}")
    
    # Compute the argument of the Lambert W function
    # arg = -alpha*d*g/(E*r_max) * exp((alpha*g/E) - beta*g)
    exponent = (alpha * g / E) - (beta * g)
    coefficient = -(alpha * d * g) / (E * r_max)
    lambert_arg = coefficient * np.exp(exponent)
    
    # Check if argument is in valid range for real branches [-1/e, 0)
    if lambert_arg < -1/np.e or lambert_arg >= 0:
        raise ValueError(
            f"Lambert W argument {lambert_arg:.6e} is outside valid range "
            f"[-1/e, 0) for E={E}. No real roots exist."
        )
    
    W_0 = lambertw(lambert_arg, k=0).real
    W_minus_1 = lambertw(lambert_arg, k=-1).real
    
    # Compute the trait values
    common_term = (alpha / E) - beta
    W_plus = common_term - (1 / g) * W_0
    W_minus = common_term - (1 / g) * W_minus_1
    
    return W_minus, W_plus


def get_trait_bounds(E):
    """
    Get the valid trait interval [W_minus, W_plus] for simulations.
    
    Parameters
    ----------
    E : float
        Extraction rate
    
    Returns
    -------
    bounds : tuple
        (W_minus, W_plus) defining the valid trait interval
    """
    W_minus, W_plus = compute_boundary_traits(E)
    return (W_minus, W_plus)
