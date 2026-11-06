"""
Utility functions for data processing and manipulation.
"""

import numpy as np


def gen_ICs(
    num_pops,
    min_pop=10,
    max_pop=500,
    min_trait=0,
    max_trait=10,
    fixed_traits=None,
    seed=None,
    rng=None,
):
    """
    Generate a list of initial conditions for the model.
    
    Args:
        num_pops (int): Number of populations.
        min_pop (int): Minimum population size.
        max_pop (int): Maximum population size.
        min_trait (float): Minimum trait value.
        max_trait (float): Maximum trait value.
        fixed_traits (list, optional): Fixed trait values to use.
        seed (int, optional): Random seed for reproducibility.
        rng (numpy.random.Generator, optional): Explicit random-number generator.
            Exactly one of ``seed`` or ``rng`` must be supplied.
    
    Returns:
        dict: Dictionary containing populations and traits for initial condition.
    """
    if seed is None and rng is None:
        raise ValueError("Either seed or rng is required for reproducibility")
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both")

    if rng is None:
        rng = np.random.default_rng(seed)

    populations = rng.integers(min_pop, max_pop, size=num_pops).tolist()
    traits = (
        fixed_traits
        if fixed_traits is not None
        else rng.uniform(min_trait, max_trait, size=num_pops).tolist()
    )
    IC = {'populations': populations,
           'traits': traits}
    return IC

def split_grid_indices(total_points, num_chunks):
    """
    Split a grid of points into chunks for parallel processing.
    
    Args:
        total_points (int): Total number of points to split.
        num_chunks (int): Number of chunks to create.
    
    Returns:
        list: List of (start, end) tuples for each chunk.
    """
    base_chunk_size = total_points // num_chunks
    remainder = total_points % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + base_chunk_size + (1 if i < remainder else 0)
        chunks.append((start, end))
        start = end
    return chunks

def combine_species_br_chunk_results(output_dir):
    """Combine species-BR chunks via the workflow-level combiner.

    This compatibility wrapper reads the expected chunk count from
    ``run_parameters.json``. New code should import
    :func:`src.workflows.combine_species_br.combine_species_br_chunk_results`
    directly and provide an explicit policy for missing chunks and incomplete
    initial conditions.
    """
    from src.workflows.combine_species_br import combine_species_br_chunk_results as combine

    return combine(output_dir=output_dir)

def collapse_species_to_unique_traits(N_row, u_row, min_thresh=1e-3,
                                      u_thresh=1e-3):
    """
    Collapse an 8-species (N, u) pair into a 4-element vector.

    Species with N <= min_thresh are dead.  Among survivors, species whose
    traits differ by less than u_thresh are merged (populations summed,
    traits averaged).  The result is zero-padded to exactly 4 elements.

    Returns
    -------
    N_collapsed : ndarray, shape (4,)
    u_collapsed : ndarray, shape (4,)
    n_unique    : int   — number of surviving trait groups
    """
    alive = N_row > min_thresh
    N_alive = N_row[alive]
    u_alive = u_row[alive]

    if len(N_alive) == 0:
        return np.zeros(4), np.zeros(4), 0

    # Sort by trait value for greedy grouping
    order = np.argsort(u_alive)
    N_sorted = N_alive[order]
    u_sorted = u_alive[order]

    # Greedy sequential merge
    groups_N = [N_sorted[0]]
    groups_u_sum = [u_sorted[0] * N_sorted[0]]  # weighted sum for mean
    groups_count = [N_sorted[0]]                 # weight = population

    for i in range(1, len(u_sorted)):
        if abs(u_sorted[i] - (groups_u_sum[-1] / groups_count[-1])) < u_thresh:
            # Same trait group — merge
            groups_N[-1] += N_sorted[i]
            groups_u_sum[-1] += u_sorted[i] * N_sorted[i]
            groups_count[-1] += N_sorted[i]
        else:
            # New trait group
            groups_N.append(N_sorted[i])
            groups_u_sum.append(u_sorted[i] * N_sorted[i])
            groups_count.append(N_sorted[i])

    n_unique = len(groups_N)
    # Compute population-weighted mean trait per group
    u_groups = [gs / gc for gs, gc in zip(groups_u_sum, groups_count)]

    # Zero-pad to 4 elements
    N_collapsed = np.zeros(4)
    u_collapsed = np.zeros(4)
    for i in range(min(n_unique, 4)):
        N_collapsed[i] = groups_N[i]
        u_collapsed[i] = u_groups[i]

    return N_collapsed, u_collapsed, n_unique
