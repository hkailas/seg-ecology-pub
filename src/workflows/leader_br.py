"""Single-species leader best responses for fixed traits from ``U_GRID``."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import tqdm

from src.result_data_structures.single_species_leader_br_result_data_types import (
    SingleSpeciesLeaderBrChunk,
    SingleSpeciesLeaderBrOptimizationResult,
    SingleSpeciesLeaderBrResultRecord,
    SingleSpeciesLeaderBrWorkItem,
)
from tools.data_utils import gen_ICs, split_grid_indices


_optimizer: Callable | None = None
_worker_model_params: dict | None = None
_worker_economic_params: dict | None = None
_worker_simulation_params: dict | None = None


def validate_leader_br_arguments(args) -> None:
    """Validate chunk and multiprocessing arguments."""
    if args.num_chunks < 1:
        raise ValueError("num_chunks must be at least 1.")
    if args.chunk_id < 0 or args.chunk_id >= args.num_chunks:
        raise ValueError("chunk_id must be between 0 and num_chunks - 1.")
    if args.num_cpus < 1:
        raise ValueError("num_cpus must be at least 1.")


def validate_leader_br_config(config: dict) -> tuple[np.ndarray, int, int]:
    """Validate and normalize the supported single-species configuration."""
    simulation_params = config["SIMULATION_PARAMS"]
    ic_params = config["IC_PARAMS"]

    if simulation_params.get("model_name") != "model1":
        raise ValueError("Single-species leader BR supports only model1 presets.")
    if ic_params.get("num_pops") != 1:
        raise ValueError("Single-species leader BR requires IC_PARAMS.num_pops == 1.")

    u_grid = np.asarray(config["U_GRID"], dtype=float)
    if u_grid.ndim != 2 or u_grid.shape[1] != 1 or len(u_grid) == 0:
        raise ValueError("U_GRID must be a non-empty array with shape (n, 1).")
    if not np.all(np.isfinite(u_grid)):
        raise ValueError("U_GRID must contain only finite values.")

    num_ics = ic_params.get("num_ICs")
    if isinstance(num_ics, bool) or not isinstance(num_ics, (int, np.integer)):
        raise TypeError("IC_PARAMS.num_ICs must be an integer.")
    if int(num_ics) < 1:
        raise ValueError("IC_PARAMS.num_ICs must be positive.")

    base_seed = ic_params.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError("IC_PARAMS.base_seed must be an integer.")
    base_seed = int(base_seed)

    for key in ("min_pop", "max_pop"):
        value = ic_params.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"IC_PARAMS.{key} must be an integer.")
    if int(ic_params["max_pop"]) <= int(ic_params["min_pop"]):
        raise ValueError("IC_PARAMS.max_pop must be greater than min_pop.")

    preset = config.get("PRESET_NAME")
    if not isinstance(preset, str) or not preset.strip():
        raise ValueError("PRESET_NAME must be a non-empty string.")

    return u_grid, int(num_ics), base_seed


def build_leader_br_work_items(
    config: dict,
    chunk_id: int,
    num_chunks: int,
) -> tuple[list[SingleSpeciesLeaderBrWorkItem], int, int, int, np.ndarray]:
    """Build trait-specific ICs using the species-BR seed hierarchy."""
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1.")
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError("chunk_id must be between 0 and num_chunks - 1.")
    u_grid, num_ics, base_seed = validate_leader_br_config(config)
    chunks = split_grid_indices(len(u_grid), num_chunks)
    idx_start, idx_end = chunks[chunk_id]
    u_chunk = u_grid[idx_start:idx_end].copy()
    chunk_seed = base_seed + chunk_id

    chunk_rng = np.random.default_rng(chunk_seed)
    seeds_for_u_values = chunk_rng.integers(
        0,
        100_000_000,
        size=len(u_chunk),
    )
    ic_params = config["IC_PARAMS"]
    work_items: list[SingleSpeciesLeaderBrWorkItem] = []

    for local_u_index, (u, u_seed) in enumerate(zip(u_chunk, seeds_for_u_values)):
        u_rng = np.random.default_rng(u_seed)
        ic_seeds = u_rng.integers(0, 1_000_000, size=num_ics)
        for ic_index, ic_seed in enumerate(ic_seeds):
            ic_rng = np.random.default_rng(ic_seed)
            initial_condition = gen_ICs(
                1,
                min_pop=int(ic_params["min_pop"]),
                max_pop=int(ic_params["max_pop"]),
                fixed_traits=u.tolist(),
                rng=ic_rng,
            )
            work_items.append(
                SingleSpeciesLeaderBrWorkItem(
                    u_index=idx_start + local_u_index,
                    ic_index=ic_index,
                    ic_seed=int(ic_seed),
                    u=np.asarray(initial_condition["traits"], dtype=float),
                    initial_populations=np.asarray(
                        initial_condition["populations"],
                        dtype=float,
                    ),
                )
            )

    return work_items, idx_start, idx_end, chunk_seed, u_chunk


def _pool_init(
    preset_name: str,
    model_params: dict,
    economic_params: dict,
    simulation_params: dict,
) -> None:
    """Initialize configuration and optimizer state in a spawned worker."""
    global _optimizer
    global _worker_model_params
    global _worker_economic_params
    global _worker_simulation_params

    os.environ["SEG_PRESET"] = preset_name
    from src.simulate.optimization import (
        find_best_E_given_u_IC,
    )

    _optimizer = find_best_E_given_u_IC
    _worker_model_params = model_params
    _worker_economic_params = economic_params
    _worker_simulation_params = simulation_params


def _worker_leader_br(
    work_item: SingleSpeciesLeaderBrWorkItem,
) -> SingleSpeciesLeaderBrResultRecord:
    """Optimize the leader response for one fixed trait and population IC."""
    if (
        _optimizer is None
        or _worker_model_params is None
        or _worker_economic_params is None
        or _worker_simulation_params is None
    ):
        raise RuntimeError("Leader-BR worker has not been initialized.")

    try:
        e_star, q_star, equilibrium, n_star = _optimizer(
            work_item.u,
            work_item.initial_populations,
            _worker_model_params,
            _worker_economic_params,
            _worker_simulation_params,
            configured_num_pops=1,
        )
        n_star = np.asarray(n_star, dtype=float)
        if not bool(equilibrium):
            raise RuntimeError("Ecological equilibrium failed at optimized E.")
        if not np.isfinite(e_star) or not np.isfinite(q_star):
            raise RuntimeError("Optimizer returned a non-finite optimum.")
        if n_star.shape != (1,) or not np.all(np.isfinite(n_star)):
            raise RuntimeError("Optimizer returned an invalid population equilibrium.")
        result = SingleSpeciesLeaderBrOptimizationResult(
            E_star=float(e_star),
            Q_star=float(q_star),
            equilibrium=True,
            N_star=n_star,
        )
    except Exception as error:
        result = SingleSpeciesLeaderBrOptimizationResult(
            E_star=np.nan,
            Q_star=np.nan,
            equilibrium=False,
            N_star=None,
            error_message=f"{type(error).__name__}: {error}",
        )
    return SingleSpeciesLeaderBrResultRecord(work_item=work_item, result=result)


def run_leader_br(
    args,
    config: dict,
    output_dir: str | Path,
) -> SingleSpeciesLeaderBrChunk:
    """Run and persist one chunk of fixed-trait leader best responses."""
    validate_leader_br_arguments(args)
    _, num_ics, base_seed = validate_leader_br_config(config)
    work_items, idx_start, idx_end, chunk_seed, u_chunk = (
        build_leader_br_work_items(config, args.chunk_id, args.num_chunks)
    )
    preset_name = config["PRESET_NAME"]

    print(
        f"Leader BR chunk {args.chunk_id}/{args.num_chunks}: "
        f"U_GRID indices {idx_start}-{idx_end - 1}, "
        f"{len(work_items)} trait/IC pairs"
    )
    start_time = time.time()
    with mp.Pool(
        args.num_cpus,
        initializer=_pool_init,
        initargs=(
            preset_name,
            config["MODEL_PARAMS"],
            config["ECONOMIC_PARAMS"],
            config["SIMULATION_PARAMS"],
        ),
    ) as pool:
        records = list(
            tqdm.tqdm(
                pool.imap(_worker_leader_br, work_items),
                total=len(work_items),
                desc=f"Leader BR chunk {args.chunk_id}",
            )
        )

    elapsed = time.time() - start_time
    failed = sum(record.result.error_message is not None for record in records)
    print(
        f"Leader BR chunk {args.chunk_id} completed in {elapsed:.1f}s: "
        f"{len(records) - failed}/{len(records)} successful"
    )

    chunk = SingleSpeciesLeaderBrChunk(
        chunk_id=args.chunk_id,
        num_chunks=args.num_chunks,
        idx_start=idx_start,
        idx_end=idx_end,
        preset=preset_name,
        base_seed=base_seed,
        chunk_seed=chunk_seed,
        num_ics=num_ics,
        u_chunk=u_chunk,
        records=records,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chunk_path = output_path / f"results_leader_br_chunk_{args.chunk_id}.pkl"
    with chunk_path.open("wb") as file:
        pickle.dump(chunk, file)
    print(f"Saved: {chunk_path}")
    return chunk
