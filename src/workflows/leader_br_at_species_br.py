"""Leader best-response simulations at species best-response equilibria.

This workflow copies the execution-specific logic from
``compute_leader_br_at_species_br.py`` without changing that analysis script.
For every distinct, non-extinct species best-response branch at an ``E`` value,
it fixes the equilibrium traits and maximises the leader payoff over ``E``.
"""

import multiprocessing as mp
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tqdm

from src.result_data_structures.leader_br_at_species_br_result_data_types import (
    LeaderBrChunk,
    LeaderBrOptimizationResult,
    LeaderBrResultRecord,
    LeaderBrWorkItem,
)
from tools.data_utils import collapse_species_to_unique_traits, split_grid_indices
from src.utils.result_paths import result_root


MIN_THRESH = 1e-4

BRANCH_ORDER = ["branch_0", "branch_1", "branch_2", "branch_3"]
BRANCH_DISPLAY = {
    "branch_0": "1 species",
    "branch_1": "2 species",
    "branch_2": "3 species",
    "branch_3": "4 species",
    "extinct": "Extinct",
}

MASTER_ORIGINAL = "results_species_br.pkl"

# These are populated in the parent process and separately in spawned workers.
find_best_E_given_u_IC = None
_worker_model_params = None
_worker_economic_params = None
_worker_simulation_params = None
_worker_num_pops = None


def _project_root() -> Path:
    """Return the repository root from this workflow module."""
    return Path(__file__).resolve().parents[2]


def _load_model(
    preset_name: str,
    model_params: dict,
    economic_params: dict,
    simulation_params: dict,
    num_pops: int,
) -> None:
    """Load the requested preset and the leader best-response optimiser."""
    global find_best_E_given_u_IC
    global _worker_model_params
    global _worker_economic_params
    global _worker_simulation_params
    global _worker_num_pops

    os.environ["SEG_PRESET"] = preset_name
    from src.simulate.optimization import find_best_E_given_u_IC as optimiser

    find_best_E_given_u_IC = optimiser
    _worker_model_params = model_params
    _worker_economic_params = economic_params
    _worker_simulation_params = simulation_params
    _worker_num_pops = num_pops


def _pool_init(
    preset_name: str,
    model_params: dict,
    economic_params: dict,
    simulation_params: dict,
    num_pops: int,
) -> None:
    """Initialise model state in each Windows-spawned worker process."""
    _load_model(
        preset_name,
        model_params,
        economic_params,
        simulation_params,
        num_pops,
    )


def load_species_br_master(data_dir: Path) -> dict:
    """Load complete species-BR results with combined branch labels."""
    original_path = data_dir / MASTER_ORIGINAL
    print(f"  Loading species-BR master: {MASTER_ORIGINAL}")
    with original_path.open("rb") as file:
        species_br = pickle.load(file)
    if "bifurcation_labels" not in species_br:
        raise ValueError(
            f"{original_path} has no bifurcation_labels. Re-run the species-BR "
            "combiner after all E values have all requested IC results."
        )
    return species_br


def preprocess_master_data(
    species_br: dict,
    min_thresh: float = MIN_THRESH,
    u_thresh: float = 1e-3,
    consistency_thresh: float = 1e-1,
) -> tuple[list[LeaderBrWorkItem], dict]:
    """Collapse, validate, and deduplicate species-BR equilibria by branch."""
    e_grid = np.asarray(species_br["E_grid"])
    num_ics_e = species_br["num_ICs_E"]
    n_e = len(e_grid)

    n_star = np.array([np.atleast_1d(value) for value in species_br["N_star"]])
    u_star = np.array([np.atleast_1d(value) for value in species_br["u_star"]])
    equilibrium = np.asarray(species_br["equilibrium"])
    bif_labels = np.asarray(species_br["bifurcation_labels"])

    num_species_raw = n_star.shape[1]
    n_star = n_star.reshape(n_e, num_ics_e, num_species_raw)
    u_star = u_star.reshape(n_e, num_ics_e, num_species_raw)
    equilibrium = equilibrium.reshape(n_e, num_ics_e)
    bif_labels = bif_labels.reshape(n_e, num_ics_e)

    total_pairs = n_e * num_ics_e
    n_not_converged = 0
    n_extinct = 0
    n_collapsed = 0
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)

    for e_idx in range(n_e):
        for ic_idx in range(num_ics_e):
            if not bool(equilibrium[e_idx, ic_idx]):
                n_not_converged += 1
                continue

            branch = str(bif_labels[e_idx, ic_idx])
            if branch == "extinct":
                n_extinct += 1
                continue

            n_collapsed_row, u_collapsed_row, n_unique = (
                collapse_species_to_unique_traits(
                    n_star[e_idx, ic_idx],
                    u_star[e_idx, ic_idx],
                    min_thresh=min_thresh,
                    u_thresh=u_thresh,
                )
            )
            if n_unique == 0:
                n_extinct += 1
                continue

            n_collapsed += 1
            grouped[(e_idx, branch)].append(
                {
                    "N": n_collapsed_row,
                    "u": u_collapsed_row,
                    "ic_idx": ic_idx,
                    "n_unique": n_unique,
                }
            )

    work_list = []
    discrepancies = []
    discrepant_keys = set()
    branch_counts_before = Counter()
    branch_counts_after = Counter()

    for (e_idx, branch), entries in grouped.items():
        branch_counts_before[branch] += len(entries)
        reference = entries[0]

        for entry in entries[1:]:
            max_delta_u = float(np.max(np.abs(reference["u"] - entry["u"])))
            max_delta_n = float(np.max(np.abs(reference["N"] - entry["N"])))
            if max_delta_u <= consistency_thresh and max_delta_n <= consistency_thresh:
                continue

            discrepancy = {
                "E": float(e_grid[e_idx]),
                "E_idx": e_idx,
                "branch": branch,
                "ref_ic": reference["ic_idx"],
                "bad_ic": entry["ic_idx"],
                "max_delta_u": max_delta_u,
                "max_delta_N": max_delta_n,
            }
            discrepancies.append(discrepancy)
            discrepant_keys.add((e_idx, branch))

            if len(discrepant_keys) > 1:
                raise ValueError(
                    f"Discrepancy #{len(discrepant_keys)} on branch {branch!r} "
                    f"at E={e_grid[e_idx]:.6f}. "
                    "See the first discrepancy in the workflow output."
                )

            print(
                f"\n  [WARN] Discrepancy at E={e_grid[e_idx]:.6f}, "
                f"branch={branch!r}; using IC {reference['ic_idx']} as representative."
            )

        branch_counts_after[branch] += 1
        work_list.append(
            LeaderBrWorkItem(
                E=float(e_grid[e_idx]),
                u_full=reference["u"].astype(float),
                N_full=reference["N"].astype(float),
                branch=branch,
                E_idx=e_idx,
                ic_idx=reference["ic_idx"],
            )
        )

    work_list.sort(
        key=lambda item: (
            BRANCH_ORDER.index(item.branch)
            if item.branch in BRANCH_ORDER
            else len(BRANCH_ORDER),
            item.E,
        )
    )

    summary = {branch: branch_counts_after.get(branch, 0) for branch in BRANCH_ORDER}
    summary["total"] = len(work_list)

    print("\n  --- Preprocessing statistics ---")
    print(f"  Raw (E, IC) pairs:          {total_pairs}")
    print(f"    Excluded non-converged:    {n_not_converged}")
    print(f"    Excluded extinct:          {n_extinct}")
    print(f"    Collapsed to 4-element:    {n_collapsed}")
    print(f"  Unique (E, branch) groups:   {len(grouped)}")
    print(f"  After dedup (unique per E,branch): {len(work_list)}")
    print("\n  Per-branch breakdown (before -> after dedup):")
    for branch in BRANCH_ORDER:
        print(
            f"    {BRANCH_DISPLAY[branch]:>12}: "
            f"{branch_counts_before.get(branch, 0):>6} -> "
            f"{branch_counts_after.get(branch, 0):>6}"
        )
    if discrepancies:
        print(f"\n  Discrepancies found: {len(discrepancies)}")

    return work_list, summary


def _worker_leader_br(work_item: LeaderBrWorkItem) -> LeaderBrResultRecord:
    """Optimise the leader response for one species-BR equilibrium point."""
    try:
        e_opt, q_opt, equilibrium, n_star_leader = find_best_E_given_u_IC(
            work_item.u_full,
            work_item.N_full,
            _worker_model_params,
            _worker_economic_params,
            _worker_simulation_params,
            configured_num_pops=_worker_num_pops,
        )
    except Exception as error:
        print(
            f"  [WARN] E={work_item.E:.5f} "
            f"branch={work_item.branch}: {error}"
        )
        e_opt, q_opt, equilibrium, n_star_leader = np.nan, np.nan, False, None

    result = LeaderBrOptimizationResult(
        E_star=float(e_opt) if np.isfinite(e_opt) else np.nan,
        Q_star=float(q_opt) if np.isfinite(q_opt) else np.nan,
        eq_status=bool(equilibrium),
        N_star_leader=(
            np.asarray(n_star_leader) if n_star_leader is not None else None
        ),
    )
    return LeaderBrResultRecord(work_item=work_item, result=result)


def run_leader_br_at_species_br(
    args,
    config: dict,
    output_dir: str | Path,
    species_results_path: str | Path | None = None,
) -> LeaderBrChunk:
    """Run and save one chunk of leader best responses at species-BR points."""
    if args.num_chunks < 1:
        raise ValueError("num_chunks must be at least 1")
    if args.chunk_id < 0 or args.chunk_id >= args.num_chunks:
        raise ValueError("chunk_id must be between 0 and num_chunks - 1")
    if args.num_cpus < 1:
        raise ValueError("num_cpus must be at least 1")

    preset_name = config["PRESET_NAME"]
    _load_model(
        preset_name,
        config["MODEL_PARAMS"],
        config["ECONOMIC_PARAMS"],
        config["SIMULATION_PARAMS"],
        config["IC_PARAMS"]["num_pops"],
    )

    data_dir = (Path(species_results_path).resolve().parent if species_results_path is not None
                else result_root(_project_root()) / config["OUTPUT_CONFIG"]["date"]
                / config["OUTPUT_CONFIG"]["compute_dir"])
    species_br = load_species_br_master(data_dir)
    work_list, summary = preprocess_master_data(species_br)

    print(f"  Work list: {summary['total']} unique (E, branch) points")
    for branch in BRANCH_ORDER:
        print(f"    {BRANCH_DISPLAY[branch]:>12}: {summary[branch]}")

    chunks = split_grid_indices(len(work_list), args.num_chunks)
    idx_start, idx_end = chunks[args.chunk_id]
    work_chunk = work_list[idx_start:idx_end]
    print(
        f"\n  Chunk {args.chunk_id}/{args.num_chunks}: "
        f"items {idx_start}-{idx_end - 1} ({len(work_chunk)} items)"
    )

    print("\n" + "=" * 70)
    print("  Leader BR: optimise E from each species equilibrium (N*, u*)")
    print("=" * 70)
    start_time = time.time()
    with mp.Pool(
        args.num_cpus,
        initializer=_pool_init,
        initargs=(
            preset_name,
            config["MODEL_PARAMS"],
            config["ECONOMIC_PARAMS"],
            config["SIMULATION_PARAMS"],
            config["IC_PARAMS"]["num_pops"],
        ),
    ) as pool:
        records = list(
            tqdm.tqdm(
                pool.imap(_worker_leader_br, work_chunk),
                total=len(work_chunk),
                desc=f"Leader BR chunk {args.chunk_id}",
            )
        )

    elapsed = time.time() - start_time
    n_valid = sum(np.isfinite(record.result.E_star) for record in records)
    n_equilibria = sum(record.result.eq_status for record in records)
    print(
        f"\n  Leader BR done in {elapsed:.1f}s: "
        f"{n_valid}/{len(records)} valid, {n_equilibria} equilibria"
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chunk_path = output_path / f"results_leader_br_at_spbr_chunk_{args.chunk_id}.pkl"
    chunk = LeaderBrChunk(
        chunk_id=args.chunk_id,
        num_chunks=args.num_chunks,
        idx_start=idx_start,
        idx_end=idx_end,
        preset=preset_name,
        records=records,
    )
    with chunk_path.open("wb") as file:
        pickle.dump(chunk, file)
    print(f"\n  Chunk saved -> {chunk_path}")
    return chunk
