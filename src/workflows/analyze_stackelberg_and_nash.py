"""Analyze species-BR payoffs and Nash points from combined simulation data."""

from collections import defaultdict
from pathlib import Path
import pickle
import tempfile
from typing import Any

import numpy as np

from src.result_data_structures.analysis_result_data_types import AnalysisResult
from src.result_data_structures.leader_br_at_species_br_result_data_types import (
    LeaderBrCombinedResult,
    LeaderBrResultRecord,
)
from src.result_data_structures.single_species_leader_br_result_data_types import (
    SingleSpeciesLeaderBrCombinedResult,
    SingleSpeciesLeaderBrResultRecord,
)
from src.simulate.optimization import (
    evaluate_fixed_trait_payoff_curve,
    payoff_at_species_br,
)
from src.workflows.combine_leader_br import (
    collapse_equivalent_E_solutions,
    find_u_indices_with_multiple_E_solutions,
    multiple_E_solution_warning,
)


BRANCH_ORDER = ("branch_0", "branch_1", "branch_2", "branch_3")
RESULT_FILENAME = "results_stack_nash.pkl"


class StackNashAnalysisError(RuntimeError):
    """Base exception for invalid inputs or failed analysis persistence."""


class InvalidAnalysisInputError(StackNashAnalysisError):
    """Raised when a combined result does not have the required structure."""


def load_species_br_results(path: str | Path) -> dict[str, Any]:
    """Load and validate the complete combined species-BR result."""
    path = Path(path)
    try:
        with path.open("rb") as file:
            data = pickle.load(file)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise InvalidAnalysisInputError(
            f"Could not load species-BR results from {path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise InvalidAnalysisInputError(
            f"{path} does not contain a combined species-BR dictionary."
        )

    required_fields = {
        "N_star",
        "u_star",
        "equilibrium",
        "E_values_order",
        "bifurcation_labels",
        "combination_status",
    }
    missing_fields = sorted(required_fields - set(data))
    if missing_fields:
        raise InvalidAnalysisInputError(
            f"{path} is missing required fields: {', '.join(missing_fields)}."
        )
    if data["combination_status"] != "complete":
        raise InvalidAnalysisInputError(
            "Stackelberg-payoff analysis requires complete species-BR results."
        )

    try:
        lengths = {
            len(data["N_star"]),
            len(data["u_star"]),
            len(data["equilibrium"]),
            len(data["E_values_order"]),
            len(data["bifurcation_labels"]),
        }
    except TypeError as error:
        raise InvalidAnalysisInputError(
            f"{path} contains a non-sequence species-BR field."
        ) from error
    if len(lengths) != 1:
        raise InvalidAnalysisInputError(
            f"{path} has misaligned species-BR result fields."
        )

    invalid_flags = [
        value
        for value in data["equilibrium"]
        if not isinstance(value, (bool, np.bool_))
    ]
    if invalid_flags:
        raise InvalidAnalysisInputError(
            f"{path} contains non-boolean equilibrium values."
        )
    return data


def load_leader_br_results(
    path: str | Path,
    expected_preset: str,
) -> LeaderBrCombinedResult | SingleSpeciesLeaderBrCombinedResult:
    """Load and validate the combined leader-BR result."""
    path = Path(path)
    try:
        with path.open("rb") as file:
            data = pickle.load(file)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise InvalidAnalysisInputError(
            f"Could not load leader-BR results from {path}: {error}"
        ) from error

    if not isinstance(
        data,
        (LeaderBrCombinedResult, SingleSpeciesLeaderBrCombinedResult),
    ):
        raise InvalidAnalysisInputError(
            f"{path} does not contain a supported combined leader-BR result."
        )
    if data.preset != expected_preset:
        raise InvalidAnalysisInputError(
            f"Leader-BR results use preset {data.preset!r}, but the configured "
            f"preset is {expected_preset!r}."
        )
    expected_record_type = (
        LeaderBrResultRecord
        if isinstance(data, LeaderBrCombinedResult)
        else SingleSpeciesLeaderBrResultRecord
    )
    if any(not isinstance(record, expected_record_type) for record in data.records):
        raise InvalidAnalysisInputError(
            f"{path} contains an invalid leader-BR result record."
        )
    if isinstance(data, SingleSpeciesLeaderBrCombinedResult):
        multiple_solution_indices = find_u_indices_with_multiple_E_solutions(
            data.records
        )
        if multiple_solution_indices:
            print(multiple_E_solution_warning(len(multiple_solution_indices)))
            raise InvalidAnalysisInputError(
                "Analysis stopped because the leader-BR result contains "
                "multiple E*(u) solutions."
            )
    return data


def assemble_branch_results(
    records: list[LeaderBrResultRecord],
) -> dict[str, dict[str, np.ndarray]]:
    """Implement Phase C by sorting leader responses and computing residuals."""
    records_by_branch: dict[str, list[LeaderBrResultRecord]] = defaultdict(list)
    for record in records:
        records_by_branch[record.work_item.branch].append(record)

    known_branches = [
        branch for branch in BRANCH_ORDER if branch in records_by_branch
    ]
    other_branches = sorted(set(records_by_branch) - set(BRANCH_ORDER))
    branch_data = {}

    for branch in known_branches + other_branches:
        rows = sorted(records_by_branch[branch], key=lambda row: row.work_item.E)
        E = np.asarray([row.work_item.E for row in rows], dtype=float)
        E_star = np.asarray([row.result.E_star for row in rows], dtype=float)
        branch_data[branch] = {
            "E": E,
            "E_star_eq": E_star,
            "Q_star_eq": np.asarray(
                [row.result.Q_star for row in rows],
                dtype=float,
            ),
            "eq_eq": np.asarray(
                [row.result.eq_status for row in rows],
                dtype=bool,
            ),
            "u_full": np.asarray([row.work_item.u_full for row in rows]),
            "N_full": np.asarray([row.work_item.N_full for row in rows]),
            "N_star_leader": np.asarray(
                [row.result.N_star_leader for row in rows],
                dtype=object,
            ),
            "E_idx": np.asarray(
                [row.work_item.E_idx for row in rows],
                dtype=int,
            ),
            "ic_idx": np.asarray(
                [row.work_item.ic_idx for row in rows],
                dtype=int,
            ),
            "residual_eq": E_star - E,
        }

    return branch_data


def assemble_grid_leader_branch_results(
    records: list[SingleSpeciesLeaderBrResultRecord],
    species_results: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Interpolate a single-species E-star(u) curve onto species-BR traits."""
    representative_records = collapse_equivalent_E_solutions(records)
    successful_records = [
        record
        for record in representative_records
        if record.result.error_message is None
        and record.result.equilibrium
        and np.isfinite(record.result.E_star)
    ]
    if not successful_records:
        raise InvalidAnalysisInputError(
            "Leader-BR results contain no successful E*(u) values."
        )

    successful_records.sort(key=lambda record: float(record.work_item.u[0]))
    u_curve = np.asarray(
        [record.work_item.u[0] for record in successful_records],
        dtype=float,
    )
    if len(np.unique(u_curve)) != len(u_curve):
        raise InvalidAnalysisInputError(
            "Leader-BR results contain duplicate U_GRID values."
        )
    E_star_curve = np.asarray(
        [record.result.E_star for record in successful_records],
        dtype=float,
    )
    Q_star_curve = np.asarray(
        [record.result.Q_star for record in successful_records],
        dtype=float,
    )
    N_star_curve = np.asarray(
        [np.asarray(record.result.N_star, dtype=float)[0] for record in successful_records],
        dtype=float,
    )

    grouped: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    source_rows = zip(
        species_results["E_values_order"],
        species_results["N_star"],
        species_results["u_star"],
        species_results["equilibrium"],
        species_results["bifurcation_labels"],
    )
    for source_index, (E, N_full, u_full, equilibrium, branch) in enumerate(
        source_rows
    ):
        branch = str(branch)
        if not bool(equilibrium) or branch == "extinct":
            continue
        N_full = np.asarray(N_full, dtype=float)
        u_full = np.asarray(u_full, dtype=float)
        if N_full.ndim != 1 or u_full.ndim != 1 or not len(u_full):
            raise InvalidAnalysisInputError(
                "Species-BR population and trait records must be vectors."
            )

        u_value = float(u_full[0])
        single_species_N = N_full[:1].copy()
        single_species_u = u_full[:1].copy()
        if u_value < u_curve[0] or u_value > u_curve[-1]:
            E_star = np.nan
            Q_star = np.nan
            N_star_leader = np.nan
        else:
            E_star = float(np.interp(u_value, u_curve, E_star_curve))
            Q_star = float(np.interp(u_value, u_curve, Q_star_curve))
            N_star_leader = float(np.interp(u_value, u_curve, N_star_curve))

        E = float(E)
        grouped[branch].setdefault(
            E,
            {
                "E": E,
                "E_star": E_star,
                "Q_star": Q_star,
                "u_full": single_species_u,
                "N_full": single_species_N,
                "N_star_leader": np.asarray([N_star_leader]),
                "source_index": source_index,
            },
        )

    known_branches = [branch for branch in BRANCH_ORDER if branch in grouped]
    other_branches = sorted(set(grouped) - set(BRANCH_ORDER))
    branch_data = {}
    for branch in known_branches + other_branches:
        rows = [grouped[branch][E] for E in sorted(grouped[branch])]
        E = np.asarray([row["E"] for row in rows], dtype=float)
        E_star = np.asarray([row["E_star"] for row in rows], dtype=float)
        branch_data[branch] = {
            "E": E,
            "E_star_eq": E_star,
            "Q_star_eq": np.asarray([row["Q_star"] for row in rows]),
            "eq_eq": np.isfinite(E_star),
            "u_full": np.asarray([row["u_full"] for row in rows]),
            "N_full": np.asarray([row["N_full"] for row in rows]),
            "N_star_leader": np.asarray(
                [row["N_star_leader"] for row in rows]
            ),
            "E_idx": np.asarray([row["source_index"] for row in rows], dtype=int),
            "ic_idx": np.zeros(len(rows), dtype=int),
            "residual_eq": E_star - E,
        }
    return branch_data


def find_zero_crossings(
    E: np.ndarray,
    residual: np.ndarray,
) -> list[dict[str, float | int]]:
    """Find exact zeros and sign-changing intervals in a residual curve."""
    E = np.asarray(E, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if E.ndim != 1 or residual.ndim != 1 or len(E) != len(residual):
        raise ValueError("E and residual must be aligned one-dimensional arrays.")

    crossings = []
    for index in range(len(residual)):
        if not np.isfinite(residual[index]):
            continue
        if residual[index] == 0:
            crossings.append(
                {
                    "x": float(E[index]),
                    "idx_l": index,
                    "idx_r": index,
                    "frac": 0.0,
                }
            )
            continue
        if index == len(residual) - 1 or not np.isfinite(residual[index + 1]):
            continue
        if residual[index] * residual[index + 1] < 0:
            fraction = -residual[index] / (
                residual[index + 1] - residual[index]
            )
            crossings.append(
                {
                    "x": float(E[index] + fraction * (E[index + 1] - E[index])),
                    "idx_l": index,
                    "idx_r": index + 1,
                    "frac": float(fraction),
                }
            )
    return crossings


def find_nash(
    branch_data: dict[str, dict[str, np.ndarray]],
    interpolate: bool = False,
    residual_threshold: float | None = 1e-4,
) -> list[dict[str, Any]]:
    """Implement Phase D by finding zeros of E-star minus E per branch."""
    if residual_threshold is not None and residual_threshold < 0:
        raise ValueError("residual_threshold must be non-negative or None.")

    nash_points = []
    for branch, data in branch_data.items():
        E = data["E"]
        E_star = data["E_star_eq"]
        residual = data["residual_eq"]
        crossings = find_zero_crossings(E, residual)

        candidates = []
        for crossing in crossings:
            left = int(crossing["idx_l"])
            right = int(crossing["idx_r"])
            fraction = float(crossing["frac"])

            if interpolate:
                E_nash = float(crossing["x"])
                E_star_nash = float(
                    E_star[left] + fraction * (E_star[right] - E_star[left])
                )
                N_nash = (
                    data["N_full"][left]
                    + fraction
                    * (data["N_full"][right] - data["N_full"][left])
                )
                u_nash = (
                    data["u_full"][left]
                    + fraction
                    * (data["u_full"][right] - data["u_full"][left])
                )
            else:
                index = (
                    left
                    if abs(residual[left]) <= abs(residual[right])
                    else right
                )
                E_nash = float(E[index])
                E_star_nash = float(E_star[index])
                N_nash = data["N_full"][index].copy()
                u_nash = data["u_full"][index].copy()

            candidates.append(
                {
                    "branch": branch,
                    "E_nash": E_nash,
                    "E_star": E_star_nash,
                    "r_E": float(E_star_nash - E_nash),
                    "N_nash": N_nash,
                    "u_nash": u_nash,
                    "interpolated": interpolate,
                }
            )

        if interpolate or residual_threshold is None:
            nash_points.extend(candidates)
            continue

        passing = [
            candidate
            for candidate in candidates
            if abs(candidate["r_E"]) < residual_threshold
        ]
        if passing:
            nash_points.append(min(passing, key=lambda item: abs(item["r_E"])))

    return nash_points


def print_nash_table(nash_points: list[dict[str, Any]]) -> None:
    """Print the Phase D Nash results."""
    if not nash_points:
        print("\n  No Nash points found.\n")
        return

    print("\n" + "=" * 100)
    print("  Nash Equilibria -- E*_leader(E) - E = 0")
    print("=" * 100)
    print(
        f"{'Branch':<12} {'Method':<14} {'E_nash':>8} {'E*':>8} "
        f"{'r_E':>10}   {'u*(E_nash)'}"
    )
    print("-" * 100)
    for point in sorted(nash_points, key=lambda item: (item["branch"], item["E_nash"])):
        traits = np.array2string(
            point["u_nash"],
            precision=4,
            separator=", ",
            formatter={"float_kind": lambda value: f"{value:.4f}"},
        )
        method = "interpolated" if point["interpolated"] else "grid"
        print(
            f"{point['branch']:<12} {method:<14} {point['E_nash']:>8.5f} "
            f"{point['E_star']:>8.5f} {point['r_E']:>+10.6f}   {traits}"
        )
    print("=" * 100)
    print(f"  Total: {len(nash_points)} point(s)\n")


def compute_nash_payoff_curves(
    nash_points: list[dict[str, Any]],
    species_E_values: Any,
    config: dict[str, Any],
    *,
    num_cpus: int = 1,
) -> list[dict[str, Any]]:
    """Compute one fixed-trait payoff curve for every detected Nash point."""
    if num_cpus < 1:
        raise ValueError("num_cpus must be at least 1.")
    base_E = np.asarray(species_E_values, dtype=float)
    base_E = np.unique(base_E[np.isfinite(base_E)])
    curves = []
    for point in nash_points:
        E_nash = float(point["E_nash"])
        curve_E = np.unique(np.append(base_E, E_nash))
        curve = evaluate_fixed_trait_payoff_curve(
            curve_E,
            point["u_nash"],
            point["N_nash"],
            config["MODEL_PARAMS"],
            config["ECONOMIC_PARAMS"],
            config["SIMULATION_PARAMS"],
            configured_num_pops=config["IC_PARAMS"]["num_pops"],
            num_cpus=num_cpus,
        )
        nash_indices = np.flatnonzero(curve["E"] == E_nash)
        if len(nash_indices) != 1:
            raise RuntimeError(
                f"Fixed-trait curve does not contain E_nash={E_nash:.12g}."
            )
        nash_index = int(nash_indices[0])
        curves.append(
            {
                **curve,
                "source": "detected_nash",
                "branch": str(point["branch"]),
                "E_nash": E_nash,
                "payoff_nash": float(curve["payoff"][nash_index]),
            }
        )
    return curves


def _save_result_atomically(result: AnalysisResult, output_path: Path) -> None:
    """Save the analysis without exposing a partially written pickle."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            pickle.dump(result, temporary_file)
        temporary_path.replace(output_path)
    except OSError as error:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise StackNashAnalysisError(
            f"Could not save analysis result to {output_path}: {error}"
        ) from error


def run_stack_nash_analysis(
    config: dict[str, Any],
    species_results_path: str | Path,
    leader_results_path: str | Path,
    output_dir: str | Path,
    interpolate_nash: bool = False,
    nash_residual_threshold: float | None = 1e-4,
    payoff_curve_num_cpus: int = 1,
) -> AnalysisResult:
    """Compute species-BR payoffs, Phase C branch data, and Phase D Nash points."""
    if payoff_curve_num_cpus < 1:
        raise ValueError("payoff_curve_num_cpus must be at least 1.")

    species_results_path = Path(species_results_path)
    leader_results_path = Path(leader_results_path)
    output_dir = Path(output_dir)

    species_results = load_species_br_results(species_results_path)
    leader_results = load_leader_br_results(
        leader_results_path,
        expected_preset=config["PRESET_NAME"],
    )

    print("\n" + "=" * 70)
    print("  Species-BR leader payoff evaluation")
    print("=" * 70)
    payoff_data = payoff_at_species_br(
        species_results["E_values_order"],
        species_results["N_star"],
        species_results["u_star"],
        species_results["equilibrium"],
        species_results["bifurcation_labels"],
        configured_num_pops=config["IC_PARAMS"]["num_pops"],
        economic_params=config["ECONOMIC_PARAMS"],
    )

    print("\n" + "=" * 70)
    print("  Phase C: Assemble results and compute residuals")
    print("=" * 70)
    if isinstance(leader_results, SingleSpeciesLeaderBrCombinedResult):
        if config["IC_PARAMS"]["num_pops"] != 1:
            raise InvalidAnalysisInputError(
                "Fixed-trait leader-BR analysis requires one configured population."
            )
        branch_data = assemble_grid_leader_branch_results(
            leader_results.records,
            species_results,
        )
    else:
        branch_data = assemble_branch_results(leader_results.records)
    for branch, data in branch_data.items():
        valid_count = int(np.count_nonzero(np.isfinite(data["E_star_eq"])))
        print(
            f"  {branch:>12}: {len(data['E'])} points, "
            f"{valid_count} valid E*"
        )

    print("\n" + "=" * 70)
    print("  Phase D: Find Nash equilibria")
    print("=" * 70)
    nash_points = find_nash(
        branch_data,
        interpolate=interpolate_nash,
        residual_threshold=nash_residual_threshold,
    )
    print_nash_table(nash_points)

    print("\n" + "=" * 70)
    print("  Fixed-trait Nash payoff curves")
    print("=" * 70)
    nash_payoff_curves = compute_nash_payoff_curves(
        nash_points,
        payoff_data["E"],
        config,
        num_cpus=payoff_curve_num_cpus,
    )
    print(f"  Computed {len(nash_payoff_curves)} Nash payoff curve(s)")

    result = AnalysisResult(
        preset=config["PRESET_NAME"],
        species_br_payoff_data=payoff_data,
        branch_data=branch_data,
        nash_points=nash_points,
        nash_payoff_curves=nash_payoff_curves,
        nash_interpolation=interpolate_nash,
        nash_residual_threshold=nash_residual_threshold,
        species_results_file=str(species_results_path.resolve()),
        leader_results_file=str(leader_results_path.resolve()),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / RESULT_FILENAME
    _save_result_atomically(result, output_path)
    print(f"Saved Stackelberg-payoff and Nash analysis to {output_path}")
    return result
