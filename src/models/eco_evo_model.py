"""
Core mathematical model functions and classes.
"""

import numpy as np
from numba import njit


@njit
def E_c(u, alpha, beta):
    return alpha / (beta + u)

@njit
def alpha_func(v, u_i, sigma_a_squared, beta_bully, fd=True):
    """
    Competition kernel based on traits; fd=False gives uniform competition.
    """
    if not fd:
        return 1.0
    return 1.0 + np.exp(-((v - u_i + beta_bully)**2) / (2.0 * sigma_a_squared)) - np.exp(-beta_bully**2 / (2.0 * sigma_a_squared))
    

@njit
def d_alpha_dv(v, u_i, beta_bully, sigma_a_squared, fd=True):
    """
    Derivative of the competition kernel with respect to v (zero if fd=False).
    """
    if not fd:
        return 0.0
    term1 = -(v - u_i + beta_bully) / sigma_a_squared
    term2 = np.exp(-((v - u_i + beta_bully)**2) / (2.0 * sigma_a_squared))
    return term1 * term2


@njit
def d2_alpha_dv2(v, u_i, beta_bully, sigma_a_squared, fd=True):
    """
    Second derivative of the competition kernel (zero if fd=False).
    """
    if not fd:
        return 0.0
    diff = v - u_i + beta_bully
    term1 = -(1.0 / sigma_a_squared)
    term2 = (1.0 - (diff**2) / sigma_a_squared)
    term3 = np.exp(-(diff**2) / (2.0 * sigma_a_squared))
    return term1 * term2 * term3


# -----------------------------------------------------------------------------
# MODEL 1 SPECIFIC FUNCTIONS
# -----------------------------------------------------------------------------


@njit
def r(u, r_max, g):
    return r_max * np.exp(-g * u)

@njit
def G(v_j, u_list, x_list, r_max, K, g, alpha, beta, beta_bully, sigma_a_squared, E, d, include_E, fd=True):
    """
    Fitness generating function for population j.
    """
    # Calculate alpha values for all u_list values at once
    alpha_values = np.array([alpha_func(v_j, u_i, sigma_a_squared, beta_bully, fd) for u_i in u_list])
    # print(alpha_values)
    # Calculate competition by vector multiplication
    A = np.sum(x_list * alpha_values)

    if include_E:
        return r(v_j, r_max, g) * (1 - A/K) * (E/E_c(v_j, alpha, beta) - 1) - d
    else:
        return r(v_j, r_max, g) * (1 - A/K) - d

@njit
def dGdv(v_j, u_list, x_list, r_max, K, g, alpha, beta, beta_bully, sigma_a_squared, E, include_E, v_j_index, fd=True):
    """
    Partial derivative of G with respect to v_j for population j.
    """
    r_vj = r(v_j, r_max, g)
    alpha_values = np.array([alpha_func(v_j, u_i, sigma_a_squared, beta_bully, fd) for u_i in u_list])
    A = np.sum(x_list * alpha_values)

    d_alpha_values = np.array([d_alpha_dv(v_j, u_i, beta_bully, sigma_a_squared, fd) for u_i in u_list])
    # Exclude self-interaction by masking out the v_j_index element
    # d_alpha_values_masked = d_alpha_values.copy()
    # d_alpha_values_masked[v_j_index] = 0.0
    # B = np.sum(x_list * d_alpha_values_masked)
    B = np.sum(x_list * d_alpha_values)
    # print('B:', B)
    # print('d_masked:', d_alpha_values_masked)
    # print('B1:', B1)
    # print('d_alpha_values:', d_alpha_values)
    # print('x_list:', x_list)

    E_cu = E_c(v_j, alpha, beta)
    D = (E / E_cu - 1.0)

    if include_E:
        term1 = -g * r_vj * (1 - A/ K) * D
        term2 = r_vj * (-B/K) * D
        # term2a = r_vj*(-B1/K)*D
        term3 = r_vj * (1 - A/ K) * (E/alpha)
        # print('masked:', term1 + term2 + term3)
        # print('unmasked:', term1 + term2a + term3)
    else:
        term1 = -g * r_vj * (1 - A/ K)
        term2 = r_vj * (-B/K)
        term3 = 0.0

    return term1 + term2 + term3

@njit
def d2Gdv2(v_j, u_list, x_list, r_max, K, g, alpha, beta, beta_bully, sigma_a_squared, E, include_E, fd=True):
    """
    Second partial derivative of G with respect to v_j for population j.
    
    This function computes the second derivative of the fitness generating function
    with respect to trait v_j, which is useful for analyzing evolutionary stability
    and trait dynamics.
    
    Args:
        v_j: Trait value at which to evaluate the second derivative
        u_list: Array of trait values for all populations
        x_list: Array of population sizes for all populations
        r_max: Maximum growth rate parameter
        K: Carrying capacity
        g: Growth rate decay parameter
        alpha: Energy parameter
        beta: Energy parameter
        beta_bully: Bullying parameter in competition kernel
        sigma_a_squared: Variance in competition kernel
        E: Environmental energy level
    
    Returns:
        float: Second derivative d²G/dv² evaluated at v_j
    """
    r_vj = r(v_j, r_max, g)
    E_cu = E_c(v_j, alpha, beta)
    
    # Compute alpha and its derivatives
    alpha_values = np.array([alpha_func(v_j, u_i, sigma_a_squared, beta_bully, fd) for u_i in u_list])
    d_alpha_values = np.array([d_alpha_dv(v_j, u_i, beta_bully, sigma_a_squared, fd) for u_i in u_list])
    d2_alpha_values = np.array([d2_alpha_dv2(v_j, u_i, beta_bully, sigma_a_squared, fd) for u_i in u_list])
    
    # Compute sums
    A = np.sum(x_list * alpha_values)
    B = np.sum(x_list * d_alpha_values)
    C = np.sum(x_list * d2_alpha_values)
    D = (E / E_cu - 1.0)
    
    if include_E:
        # Term 1
        term1 = r_vj * D * (g**2 * (1.0 - A / K) + 2.0 * g * B/K - C/K)
        # Term 2
        term2 = r_vj * (-E/alpha) * (g * (1.0 - A / K) + B / K)
    else:
        term1 = r_vj * (g**2 * (1.0 - A / K) + 2.0 * g * B/K - C/K)
        term2 = 0.0        
    return term1 + term2

@njit
def ode_model1(t, y, r_max, K, g, alpha, beta, beta_bully, sigma_a_squared, E, d, sigma, include_E, num_pops, evo_dyn=True, min_thresh=None, fd=True):
    """
    System of ODEs for a polymorphic model with N populations.
    """
    dydt = np.zeros(2 * num_pops)

    # Extract trait values and population sizes
    x_list = y[:num_pops]  # Extract all populations
    u_list = y[num_pops:]  # Extract all traits
    
    # Set populations below the minimum threshold to zero
    if min_thresh is not None:
        x_list[x_list < min_thresh] = 0.0
    
    # Compute G for all populations
    G_list = np.array([G(v_j, u_list, x_list, r_max, K, g, alpha, beta, beta_bully, sigma_a_squared, E, d, include_E, fd) for v_j in u_list]).flatten()
        
    # Compute population size changes
    dxdt = x_list * G_list
    
    if evo_dyn:
        dGdv_list = []
        for v_j_index, v_j in enumerate(u_list):
            # Compute trait derivatives
            dGdv_list.append(dGdv(v_j, u_list, x_list, r_max, K, g, alpha, beta, beta_bully, sigma_a_squared, E, include_E, v_j_index, fd))
            
        dGdv_list = np.array(dGdv_list)
        dudt = sigma * dGdv_list
    else:
        dudt = np.zeros(num_pops)
    
    if min_thresh is not None:
        dudt[x_list < min_thresh] = 0.0 # The trait stops evolving if the population is below the minimum threshold

    # Update dydt with computed derivatives
    dydt[:num_pops] = dxdt
    dydt[num_pops:] = dudt
    return dydt

# -----------------------------------------------------------------------------
# MODEL 3 SPECIFIC FUNCTIONS 
# -----------------------------------------------------------------------------

@njit
def K_func(v, K_max, g):
    """
    Carrying capacity for Model 3: Cost is in K.
    K(v) = K_max * exp(-g * v)
    """
    return K_max * np.exp(-g * v)

@njit
def G_model3(v_j, u_list, x_list, r_max, K_max, g, alpha, beta, beta_bully, sigma_a_squared, E, d, include_E, fd=True):
    """
    G function for Model 3.
    """
    # Calculate alpha values
    alpha_values = np.array([alpha_func(v_j, u_i, sigma_a_squared, beta_bully, fd) for u_i in u_list])
    
    # Competition
    A = np.sum(x_list * alpha_values)
    
    # Calculate trait-dependent K
    Kv = K_func(v_j, K_max, g)
    
    # Calculate Energy term (E - E_c)/E_c
    E_cu = E_c(v_j, alpha, beta)
    D = (E / E_cu) - 1.0
    # print(A/Kv, D)
    if include_E: 
        return r_max * (1.0 - A / Kv) * D -d
    else:
        return r_max * (1.0 - A / Kv) -d
    
@njit
def dGdv_model3(v_j, u_list, x_list, r_max, K_max, g, alpha, beta, beta_bully, sigma_a_squared, E, include_E, fd=True):
    """
    Partial derivative of G with respect to v for Model 3.
    """
    # Pre-compute terms
    Kv = K_func(v_j, K_max, g)
    E_cu = E_c(v_j, alpha, beta)
    
    alpha_values = np.array([alpha_func(v_j, u_i, sigma_a_squared, beta_bully, fd) for u_i in u_list])
    d_alpha_values = np.array([d_alpha_dv(v_j, u_i, beta_bully, sigma_a_squared, fd) for u_i in u_list])
    
    A = np.sum(x_list * alpha_values)
    B = np.sum(x_list * d_alpha_values)
    D = (E / E_cu) - 1.0

    if include_E:
        term1 = r_max * (-D * (B + g*A)/ Kv)
        term2 = r_max * (1.0 - A / Kv) * (E / alpha)
    else:
        term1 = r_max * (-(B + g*A)/ Kv)
        term2 = 0.0

    return term1 + term2
    

@njit
def d2Gdv2_model3(v_j, u_list, x_list, r_max, K_max, g, alpha, beta, beta_bully, sigma_a_squared, E, include_E, fd=True):
    """
    Second partial derivative of G with respect to v for Model 3.
    """
    Kv = K_func(v_j, K_max, g)
    E_cu = E_c(v_j, alpha, beta)
    
    alpha_values = np.array([alpha_func(v_j, u_i, sigma_a_squared, beta_bully, fd) for u_i in u_list])
    d_alpha_values = np.array([d_alpha_dv(v_j, u_i, beta_bully, sigma_a_squared, fd) for u_i in u_list])
    d2_alpha_values = np.array([d2_alpha_dv2(v_j, u_i, beta_bully, sigma_a_squared, fd) for u_i in u_list])
    
    A = np.sum(x_list * alpha_values)
    B = np.sum(x_list * d_alpha_values)
    C = np.sum(x_list * d2_alpha_values)
    D = (E / E_cu) - 1.0

    if include_E:
        term1 = 2.0 * (E/alpha) * (B + g * A)
        term2 = D * (g**2 * A + 2.0 * g * B + C)
    else:
        term1 = 0.0
        term2 = g**2 * A + 2.0 * g * B + C
    return - (r_max/Kv) * (term1 + term2)


@njit
def ode_model3(t, y, r_max, K_max, g, alpha, beta, beta_bully, sigma_a_squared, E, d, sigma, include_E, num_pops, evo_dyn=True, min_thresh=None, fd=True):
    """
    System of ODEs for Model 3 (Cost in K).
    """
    dydt = np.zeros(2 * num_pops)
    x_list = y[:num_pops]
    u_list = y[num_pops:]
    
    if min_thresh is not None:
        x_list[x_list < min_thresh] = 0.0
    
    # Compute G for all populations using Model 3
    G_list = np.array([G_model3(v_j, u_list, x_list, r_max, K_max, g, alpha, beta, beta_bully, sigma_a_squared, E, d, include_E, fd) for v_j in u_list]).flatten()
        
    dxdt = x_list * G_list
    
    if evo_dyn:
        # Compute trait derivatives using Model 3
        dGdv_list = np.array([dGdv_model3(v_j, u_list, x_list, r_max, K_max, g, alpha, beta, beta_bully, sigma_a_squared, E, include_E, fd) for v_j in u_list])
        dudt = sigma * dGdv_list
        # Enforce non-negative trait constraint
        dudt[(u_list <= 0.0) & (dudt < 0.0)] = 0.0
    else:
        dudt = np.zeros(num_pops)
    
    if min_thresh is not None:
        dudt[x_list < min_thresh] = 0.0

    dydt[:num_pops] = dxdt
    dydt[num_pops:] = dudt
    
    return dydt


class EcoEvoModel:
    def __init__(self, params):
        self.r_max              = params['r_max']
        self.K                  = params['K']
        self.g                  = params['g']
        self.alpha              = params['alpha']
        self.beta               = params['beta']
        self.beta_bully         = params['beta_bully']
        self.sigma_a_squared    = params['sigma_a_squared']
        self.E                  = params['E']
        self.d                  = params.get('d', 0)
        self.sigma              = params['sigma']
        self.include_E          = params.get('include_E', True)
        self.fd                 = params.get('fd', True)

    def derivatives(self, t, y):
        raise NotImplementedError("Use a subclass of EcoEvoModel")


class Model1(EcoEvoModel):

    def derivatives(self, t, y):
        arr = np.array(y, dtype=np.float64)
        out = ode_model1(t, arr, self.r_max, self.K, self.g, 
                               self.alpha, self.beta, self.beta_bully,
                               self.sigma_a_squared, self.E, self.d,
                               self.sigma, self.include_E, len(y)//2, fd=self.fd)
        return out.tolist()

class Model3(EcoEvoModel):
    """
    Wrapper for Model 3 (Cost in K).
    """
    def derivatives(self, t, y):
        arr = np.array(y, dtype=np.float64)
        # Note: self.K is passed as K_max
        out = ode_model3(t, arr, self.r_max, self.K, self.g, 
                               self.alpha, self.beta, self.beta_bully,
                               self.sigma_a_squared, self.E, self.d,
                               self.sigma, self.include_E, len(y)//2, fd=self.fd)
        return out.tolist()