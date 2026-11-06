"""Species best-response simulation workflow."""

import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import tqdm

from src.simulate.optimization import find_best_u_best_N_given_E_IC
from src.result_data_structures.simulation_result_data_types import is_failed_result
from src.models.eco_evo_model import K_func
from tools.LambertW import compute_boundary_traits
from tools.data_utils import gen_ICs, split_grid_indices


def run_species_br(args, config, output_dir):
    """Run the species best-response workflow for one grid chunk."""
    simulation_params = config['SIMULATION_PARAMS']
    model_params = config['MODEL_PARAMS']
    ic_params = config['IC_PARAMS']
    e_grid = config['E_GRID']

    num_pops = ic_params['num_pops']
    num_ICs_E = ic_params['num_ICs_E']
    chunk_seed = ic_params['base_seed'] + args.chunk_id

    chunks_spbr = split_grid_indices(len(e_grid), args.num_chunks)
    if args.chunk_id >= args.num_chunks or args.chunk_id < 0:
        raise ValueError("chunk_id must be between 0 and NUM_CHUNKS-1")

    idx_start, idx_end = chunks_spbr[args.chunk_id]
    E_chunk = e_grid[idx_start:idx_end]

    def generate_initial_condition(seed, min_trait, max_trait):
        ic_rng = np.random.default_rng(seed)
        IC = gen_ICs(
            num_pops,
            min_pop=ic_params['min_pop'],
            max_pop=ic_params['max_pop'],
            min_trait=min_trait,
            max_trait=max_trait,
            rng=ic_rng,
        )
        if simulation_params['model_name'] == 'model3':
            max_population = int(
                K_func(
                    max(IC['traits']),
                    model_params['K'],
                    model_params['g'],
                ) / 2
            )
            IC['populations'] = ic_rng.integers(
                ic_params['min_pop'],
                max_population,
                size=num_pops,
            ).tolist()
        return IC

    chunk_rng = np.random.default_rng(chunk_seed)
    seeds_for_E_vals = chunk_rng.integers(0, 100_000_000, size=len(E_chunk))
    print(f"Initial round: {num_ICs_E} ICs for {len(E_chunk)} E values (yet to filter by valid traits)")
    E_IC_pairs = []
    E_IC_pairs_failed = []
    E_skipped = []
    for i, E in enumerate(E_chunk):
        if simulation_params['model_name'] == 'model1':
            try:
                u_ub_temp, u_lb_temp = compute_boundary_traits(E)
                u_ub = min(u_ub_temp + 0.1, ic_params['max_trait'])
                u_lb = max(u_lb_temp - 0.1, ic_params['min_trait'])
            except ValueError:
                E_skipped.append(E)
                continue
        else:
            u_ub, u_lb = ic_params['max_trait'], ic_params['min_trait']

        E_rng = np.random.default_rng(seeds_for_E_vals[i])
        seeds_for_ICs = E_rng.integers(0, 1_000_000, size=num_ICs_E)
        for ic_seed in seeds_for_ICs:
            IC = generate_initial_condition(ic_seed, u_lb, u_ub)
            E_IC_pairs.append((E, IC))

    start_time = time.time()
    with mp.Pool(args.num_cpus) as pool:
        results_species_br = list(tqdm.tqdm(
            pool.imap(find_best_u_best_N_given_E_IC, E_IC_pairs),
            total=len(E_IC_pairs),
            desc=f"Chunk {args.chunk_id} Species BR"
        ))

    N_star_list = [result.N_star for result in results_species_br]
    u_star_list = [result.u_star for result in results_species_br]
    equilibrium_list = [result.equilibrium for result in results_species_br]

    all_results = results_species_br.copy()
    all_results_E_values = [E for E, _ in E_IC_pairs]
    final_E_metadata = {}

    for E in E_chunk:
        final_E_metadata[E] = {
            'successful_count': 0,
            'total_attempts': 0,
            'final_success': False,
            'iterations_completed': 0
        }

    for i, (E, IC) in enumerate(E_IC_pairs):
        result = all_results[i]
        final_E_metadata[E]['total_attempts'] += 1

        if not is_failed_result(result):
            final_E_metadata[E]['successful_count'] += 1
        elif E not in E_skipped:
            E_IC_pairs_failed.append((E, IC))

    initial_results_filename = f"initial_results_species_br_chunk_{args.chunk_id}.pkl"
    initial_complete_E_values = 0
    initial_incomplete_E_values = []
    for E in E_chunk:
        if final_E_metadata[E]['successful_count'] >= num_ICs_E:
            initial_complete_E_values += 1
            final_E_metadata[E]['final_success'] = True
        else:
            initial_incomplete_E_values.append({
                'E': E,
                'successful_count': final_E_metadata[E]['successful_count'],
                'missing_count': num_ICs_E - final_E_metadata[E]['successful_count']
            })

    with open(os.path.join(output_dir, initial_results_filename), 'wb') as f:
        pickle.dump({
            'N_star': N_star_list,
            'u_star': u_star_list,
            'equilibrium': equilibrium_list,
            'E_values_order': [E for E, _ in E_IC_pairs],
            'E_chunk': E_chunk,
            'num_ICs_E': num_ICs_E,
            'chunk_seed': chunk_seed,
            'chunk_id': args.chunk_id,
            'idx_start': idx_start,
            'idx_end': idx_end,
            'initial_complete_E_count': initial_complete_E_values,
            'initial_incomplete_E_count': len(initial_incomplete_E_values),
            'initial_incomplete_E_values': initial_incomplete_E_values,
            'E_IC_pairs_failed': E_IC_pairs_failed,
            'E_metadata': final_E_metadata.copy(),
            'stage': 'initial_run'
        }, f)

    completed_iterations = 0
    for iteration in range(1, args.max_attempts + 1):
        E_values_needing_healing = []
        total_missing_solutions = 0

        for E in E_chunk:
            successful_count = final_E_metadata[E]['successful_count']
            if successful_count < num_ICs_E:
                if E in E_skipped:
                    continue
                missing_count = num_ICs_E - successful_count
                E_values_needing_healing.append((E, missing_count))
                total_missing_solutions += missing_count
            else:
                final_E_metadata[E]['final_success'] = True

        if not E_values_needing_healing:
            print(f"All E values completed successfully after {iteration - 1} healing iterations")
            completed_iterations = iteration - 1
            break

        print(f"Healing iteration {iteration}: Processing {len(E_values_needing_healing)} E values, generating {total_missing_solutions} new ICs")

        healing_E_IC_pairs = []
        for E, missing_count in E_values_needing_healing:
            E_index = list(E_chunk).index(E)
            healing_chunk_seed = chunk_seed + iteration * 1000
            E_rng = np.random.default_rng(healing_chunk_seed + E_index)
            healing_seeds_for_ICs = E_rng.integers(0, 1_000_000, size=missing_count)
            if simulation_params['model_name'] == 'model1':
                u_ub_temp, u_lb_temp = compute_boundary_traits(E)
                u_ub = min(u_ub_temp + 0.1, ic_params['max_trait'])
                u_lb = max(u_lb_temp - 0.1, ic_params['min_trait'])
            else:
                u_ub, u_lb = ic_params['max_trait'], ic_params['min_trait']
            for ic_seed in healing_seeds_for_ICs:
                IC = generate_initial_condition(ic_seed, u_lb, u_ub)
                healing_E_IC_pairs.append((E, IC))

        with mp.Pool(args.num_cpus) as pool:
            healing_results = list(tqdm.tqdm(
                pool.imap(find_best_u_best_N_given_E_IC, healing_E_IC_pairs),
                total=len(healing_E_IC_pairs),
                desc=f"Healing iteration {iteration}"
            ))

        for i, result in enumerate(healing_results):
            E, IC = healing_E_IC_pairs[i]
            final_E_metadata[E]['total_attempts'] += 1
            final_E_metadata[E]['iterations_completed'] = iteration

            if not is_failed_result(result):
                final_E_metadata[E]['successful_count'] += 1
            else:
                E_IC_pairs_failed.append((E, IC))
            all_results.append(result)
            all_results_E_values.append(E)

        healing_results_filename = f"healing_iter_{iteration}_species_br_chunk_{args.chunk_id}.pkl"
        current_complete_E_values = 0
        current_incomplete_E_values = []
        for E in E_chunk:
            if final_E_metadata[E]['successful_count'] >= num_ICs_E:
                current_complete_E_values += 1
            else:
                current_incomplete_E_values.append({
                    'E': E,
                    'successful_count': final_E_metadata[E]['successful_count'],
                    'missing_count': num_ICs_E - final_E_metadata[E]['successful_count']
                })

        current_final_results = []
        current_final_E_values = []
        for E in E_chunk:
            E_results_collected = 0
            target_results = min(final_E_metadata[E]['successful_count'], num_ICs_E)
            for i, result in enumerate(all_results):
                result_E = all_results_E_values[i]
                if (
                    result_E == E
                    and not is_failed_result(result)
                    and E_results_collected < target_results
                ):
                    current_final_results.append(result)
                    current_final_E_values.append(E)
                    E_results_collected += 1

        current_N_star = [result.N_star for result in current_final_results]
        current_u_star = [result.u_star for result in current_final_results]
        current_equilibrium = [result.equilibrium for result in current_final_results]

        with open(os.path.join(output_dir, healing_results_filename), 'wb') as f:
            pickle.dump({
                'N_star': current_N_star,
                'u_star': current_u_star,
                'equilibrium': current_equilibrium,
                'E_values_order': current_final_E_values,
                'E_chunk': E_chunk,
                'num_ICs_E': num_ICs_E,
                'chunk_seed': chunk_seed,
                'chunk_id': args.chunk_id,
                'idx_start': idx_start,
                'idx_end': idx_end,
                'iteration': iteration,
                'complete_E_count': current_complete_E_values,
                'incomplete_E_count': len(current_incomplete_E_values),
                'incomplete_E_values': current_incomplete_E_values,
                'E_IC_pairs_failed': E_IC_pairs_failed,
                'E_metadata': final_E_metadata.copy(),
                'stage': f'healing_iteration_{iteration}'
            }, f)

        completed_iterations = iteration

    for E in E_chunk:
        if final_E_metadata[E]['successful_count'] >= num_ICs_E:
            final_E_metadata[E]['final_success'] = True

    final_N_star_list = []
    final_u_star_list = []
    final_equilibrium_list = []
    final_list_E_values = []

    for E in E_chunk:
        E_results_collected = 0
        target_results = min(final_E_metadata[E]['successful_count'], num_ICs_E)
        for i, result in enumerate(all_results):
            result_E = all_results_E_values[i]
            if (
                result_E == E
                and not is_failed_result(result)
                and E_results_collected < target_results
            ):
                final_N_star_list.append(result.N_star)
                final_u_star_list.append(result.u_star)
                final_equilibrium_list.append(result.equilibrium)
                final_list_E_values.append(E)
                E_results_collected += 1

    incomplete_E_values = []
    for E in E_chunk:
        if final_E_metadata[E]['successful_count'] < num_ICs_E:
            if E in E_skipped:
                continue
            incomplete_E_values.append({
                'E': E,
                'successful_count': final_E_metadata[E]['successful_count'],
                'target_count': num_ICs_E,
                'missing_count': num_ICs_E - final_E_metadata[E]['successful_count'],
                'total_attempts': final_E_metadata[E]['total_attempts'],
                'iterations_completed': final_E_metadata[E]['iterations_completed']
            })

    successful_cases = len(final_N_star_list)

    if incomplete_E_values:
        incomplete_E_list = [case['E'] for case in incomplete_E_values]
        print(f"{len(incomplete_E_list)} Incomplete E values. These are:", incomplete_E_list)
        incomplete_filename = f"incomplete_E_values_chunk_{args.chunk_id}.pkl"
        with open(os.path.join(output_dir, incomplete_filename), 'wb') as f:
            pickle.dump(incomplete_E_values, f)
    else:
        print(f"All {len(E_chunk)} E values achieved {num_ICs_E} successful solutions")

    filename = f"results_species_br_chunk_{args.chunk_id}.pkl"
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump({
            'N_star': final_N_star_list,
            'u_star': final_u_star_list,
            'equilibrium': final_equilibrium_list,
            'E_values_order': final_list_E_values,
            'E_chunk': E_chunk,
            'num_ICs_E': num_ICs_E,
            'chunk_seed': chunk_seed,
            'chunk_id': args.chunk_id,
            'idx_start': idx_start,
            'idx_end': idx_end,
            'incomplete_E_count': len(incomplete_E_values),
            'E_IC_pairs_failed': E_IC_pairs_failed,
            'successful_count': successful_cases,
            'max_attempts': args.max_attempts,
            'E_metadata': final_E_metadata,
            'incomplete_E_values': incomplete_E_values,
            'E_skipped': E_skipped,
            'healing_enabled': True
        }, f)

    if len(incomplete_E_values) == 0:
        print("Cleaning up initial results file...")
        initial_results_filename = f"initial_results_species_br_chunk_{args.chunk_id}.pkl"
        initial_results_path = os.path.join(output_dir, initial_results_filename)
        if os.path.exists(initial_results_path):
            try:
                os.remove(initial_results_path)
            except OSError as e:
                print(f"Could not delete initial results file {initial_results_filename}: {e}")
        if completed_iterations > 0:
            print("Cleaning up healing iteration files...")
            for iter_num in range(1, completed_iterations + 1):
                healing_file = f"healing_iter_{iter_num}_species_br_chunk_{args.chunk_id}.pkl"
                healing_file_path = os.path.join(output_dir, healing_file)
                if os.path.exists(healing_file_path):
                    try:
                        os.remove(healing_file_path)
                        print(f"Deleted healing iteration file: {healing_file}")
                    except OSError as e:
                        print(f"Could not delete healing iteration file {healing_file}: {e}")
        print("Cleanup completed")
    elif completed_iterations > 0:
        print(f"Keeping {completed_iterations} healing iteration files due to incomplete E values")

    total_time = time.time() - start_time
    print(f"Species best response chunk {args.chunk_id} processing time: {total_time:.2f} seconds")
    print(f"Chunk {args.chunk_id} done: processed E values {idx_start} to {idx_end - 1}")
