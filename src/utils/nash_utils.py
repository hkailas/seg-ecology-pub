"""
Utility functions for Nash equilibrium analysis.
"""

import numpy as np


def sort_nash_equilibria(u, N, trait_tol=1e-3, min_thresh=1e-4):
    """
    Create canonical representation of Nash equilibrium by:
    1. Filtering out extinct species (N < min_thresh)
    2. Sorting by trait values
    3. Grouping species with similar traits (within trait_tol)
    
    This makes equilibria comparable regardless of:
    - Species ordering (same species, different order)
    - Near-identical traits (traits within trait_tol are merged)
    
    Args:
        u (np.ndarray): Array of trait values
        N (np.ndarray): Array of population sizes
        trait_tol (float): Tolerance for grouping similar traits
        min_thresh (float): Threshold below which populations are considered extinct
        
    Returns:
        tuple: (sorted_traits, sorted_populations) - canonical representation
            Empty arrays if all populations are extinct
    
    Example:
        >>> u1, N1 = np.array([10, 15]), np.array([50, 30])
        >>> u2, N2 = np.array([15, 10]), np.array([30, 50])
        >>> sort_nash_equilibria(u1, N1) == sort_nash_equilibria(u2, N2)
        True
    """
    # Filter out extinct populations
    alive_mask = N > min_thresh
    u_alive = u[alive_mask]
    N_alive = N[alive_mask]
    
    if len(u_alive) == 0:
        return np.array([]), np.array([])
    
    # Sort by trait values
    sort_idx = np.argsort(u_alive)
    u_sorted = u_alive[sort_idx]
    N_sorted = N_alive[sort_idx]
    
    # Group species with similar traits
    canonical_u = []
    canonical_N = []
    
    i = 0
    while i < len(u_sorted):
        # Find all species within trait_tol of current species
        group_mask = np.abs(u_sorted - u_sorted[i]) < trait_tol
        
        # Average trait and sum populations for this group
        group_u = np.mean(u_sorted[group_mask])
        group_N = np.sum(N_sorted[group_mask])
        
        canonical_u.append(group_u)
        canonical_N.append(group_N)
        
        # Skip to next ungrouped species
        i = np.where(group_mask)[0][-1] + 1
    
    return np.array(canonical_u), np.array(canonical_N)

