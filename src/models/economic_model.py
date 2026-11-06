"""
Economic model functions for leader payoff calculations.
"""

import numpy as np
from ..simulate.simulation_core import species_br

def Q_leader(
    E,
    L,
    p,
    delta,
    alpha_leader,
    N_total,
    N_star=None,
    u_star=None,
    c1=0,
    c4=0,
    c6=0,
    epsilon=1e-10,
    *,
    configured_num_pops,
):
    """
    Returns leader payoff given N_total and trait values.
    
    Parameters:
    -----------
    E : float
        Exploitation level
    L : float
        Linear payoff coefficient
    p : float
        Price parameter
    delta : float
        Discount rate
    alpha_leader : float
        Leader's catch efficiency
    N_total : float or array
        Total population(s) or array of populations
    u_values : array, optional
        Trait values for each species
    c1 : float, optional
        Coefficient for squared mean trait term
    c4 : float, optional
        Coefficient for trait variance term V_u = sum_i p_i*(u_i - u_mean)^2
    c6 : float, optional
        Coefficient for richness term R = sum_i ln(epsilon + N_i)
    epsilon : float, optional
        Small positive value for numerical stability in log calculations
    configured_num_pops : int
        Number of populations configured for the simulation. This is distinct
        from the number of surviving populations and from padded vector width.
    
    Returns:
    --------
    float
        Leader payoff value
    """
    # Base payoff terms
    payoff = L*(1 - E) + p/delta*(1 - np.exp(-alpha_leader*N_total))
    # print(N_star)
    
    # Add trait-based terms if u_values provided
    if u_star is not None and N_star is not None:
        u_star = np.array(u_star)
        N_star = np.array(N_star)
        alive = N_star > 1e-4
        effective_N_star = np.where(alive, N_star, 0.0)
        effective_u_star = np.where(alive, u_star, 0.0)

        # Apply c1/u only to single-species simulations
        # c1 / trait value
        if c1 != 0 and configured_num_pops == 1:
            trait = effective_u_star[..., 0]
            single_population_alive = alive[..., 0]
            addition = np.divide(
                c1,
                np.maximum(trait, epsilon),
                out=np.zeros_like(trait, dtype=float),
                where=single_population_alive,
            )
            payoff += addition
        
        # Weighted variance of traits
        # c4 * [weighted variance of traits]
        # V_u(E) = sum_i p_i*(u_i - u_mean)^2 = sum_i p_i*u_i^2 - (sum_i p_i*u_i)^2 
        # where p_i = N_i/N_total and u_mean = sum_i p_i*u_i
        if c4 != 0:
            effective_N_total = np.sum(
                effective_N_star,
                axis=-1,
                keepdims=True,
            )
            props = np.divide(
                effective_N_star,
                np.maximum(effective_N_total, epsilon),
                out=np.zeros_like(effective_N_star, dtype=float),
                where=effective_N_total > 0,
            )
            mean_u = np.sum(
                props * effective_u_star,
                axis=-1,
                keepdims=True,
            )
            difference_term = effective_u_star - mean_u
            variance = np.sum(props * difference_term**2, axis=-1)  # V_u(E)
            addition = c4 * variance
            if isinstance(addition, np.ndarray) and addition.size == 1:
                addition = addition.item()
            payoff += addition
        
        # Species richness
        # c6 * [richness] = c6 * sum_i ln(epsilon + N_i)
        # R(E) = sum_i ln(epsilon + N_i)
        if c6 != 0:
            num_alive = np.count_nonzero(alive, axis=-1)
            num_lost = configured_num_pops - num_alive
            if np.any(num_lost < 0):
                raise ValueError(
                    "Number of living unique populations cannot exceed "
                    "configured_num_pops."
                )
            living_richness = np.sum(
                np.where(
                    alive,
                    np.log(epsilon + effective_N_star),
                    0.0,
                ),
                axis=-1,
            )
            richness = living_richness + num_lost * np.log(epsilon)
            if isinstance(richness, np.ndarray) and richness.size == 1:
                richness = richness.item()
            payoff += c6 * richness

    return payoff


def Q(
    E,
    params,
    IC,
    L,
    p,
    delta,
    alpha_leader,
    c1=0,
    c4=0,
    c6=0,
    epsilon=1e-10,
    *,
    configured_num_pops,
    **sim_kwargs,
):
    """
    Calculate leader payoff for given E using species best response.
    
    Parameters:
    -----------
    E : float
        Exploitation level
    params : dict
        Model parameters
    IC : dict or array
        Initial conditions
    L : float
        Linear payoff coefficient
    p : float
        Price parameter
    delta : float
        Discount rate
    alpha_leader : float
        Leader's catch efficiency
    c1 : float, optional
        Coefficient for squared mean trait term
    c4 : float, optional
        Coefficient for trait variance term
    c6 : float, optional
        Coefficient for richness term
    epsilon : float, optional
        Small positive value for numerical stability
    configured_num_pops : int
        Number of populations configured for the simulation.
    **sim_kwargs : dict
        Additional simulation keyword arguments
    
    Returns:
    --------
    float
        Leader payoff value
    """
    
    def total_N_and_u(E, params, IC, weights=None, **sim_kwargs):
        N_star, u_star = species_br(E, params, IC, **sim_kwargs)
        # weighted sum of N_star
        if weights is not None:
            N_star = np.array([N_star[i] * weights[i] for i in range(len(N_star))])
        return N_star, u_star
    
    N_star, u_star = total_N_and_u(E, params, IC, weights=None, **sim_kwargs)
    N_total = np.sum(N_star)
    return Q_leader(E, L, p, delta, alpha_leader, N_total, 
                   N_star=N_star, u_star=u_star, c1=c1, c4=c4, c6=c6,
                   epsilon=epsilon,
                   configured_num_pops=configured_num_pops)
