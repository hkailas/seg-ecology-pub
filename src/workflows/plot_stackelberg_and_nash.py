"""Load and validate persisted analysis and best-response curves for paper plots."""

from pathlib import Path
import pickle
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
from src.workflows.combine_leader_br import (
    collapse_equivalent_E_solutions,
    find_u_indices_with_multiple_E_solutions,
    multiple_E_solution_warning,
)


_REQUIRED_PAYOFF_FIELDS = {
    "E",
    "payoff",
    "branch",
    "equilibrium",
    "N_star",
    "u_star",
    "E_stackelberg",
    "N_star_stackelberg",
    "u_star_stackelberg",
    "branch_stackelberg",
    "payoff_stackelberg",
}


class StackNashPlottingError(RuntimeError):
    """Raised when analysis plot inputs or outputs are invalid."""


def load_stack_nash_analysis(
    path: str | Path,
    expected_preset: str | None = None,
) -> AnalysisResult:
    """Load and validate the analysis result needed by the plotting workflow."""
    path = Path(path)
    try:
        with path.open("rb") as file:
            result = pickle.load(file)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise StackNashPlottingError(
            f"Could not load Stackelberg/Nash analysis from {path}: {error}"
        ) from error

    if not isinstance(result, AnalysisResult):
        raise StackNashPlottingError(
            f"{path} does not contain an AnalysisResult."
        )
    if not hasattr(result, "nash_payoff_curves"):
        raise StackNashPlottingError(
            "Analysis is missing nash_payoff_curves. Re-run "
            "main_analyze_stackelberg_and_nash.py to regenerate it."
        )
    if not isinstance(result.nash_payoff_curves, list):
        raise StackNashPlottingError(
            "Analysis nash_payoff_curves is not a list. Re-run "
            "main_analyze_stackelberg_and_nash.py to regenerate it."
        )
    if expected_preset is not None and result.preset != expected_preset:
        raise StackNashPlottingError(
            f"Analysis uses preset {result.preset!r}, but the configured preset "
            f"is {expected_preset!r}."
        )

    payoff_data = result.species_br_payoff_data
    if not isinstance(payoff_data, dict):
        raise StackNashPlottingError(
            "Analysis species_br_payoff_data is not a dictionary."
        )
    missing_fields = sorted(_REQUIRED_PAYOFF_FIELDS - set(payoff_data))
    if missing_fields:
        raise StackNashPlottingError(
            "Analysis is missing plot fields: "
            f"{', '.join(missing_fields)}. Re-run "
            "main_analyze_stackelberg_and_nash.py to regenerate it."
        )
    return result

def load_single_species_leader_br_curve(
    path: str | Path,
    expected_preset: str | None = None,
) -> dict[str, np.ndarray | str]:
    """Load leader-BR coordinates for a single-species response plot."""
    path = Path(path)
    try:
        with path.open("rb") as file:
            result = pickle.load(file)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise StackNashPlottingError(
            f"Could not load leader-BR results from {path}: {error}"
        ) from error

    if isinstance(result, SingleSpeciesLeaderBrCombinedResult):
        expected_record_type = SingleSpeciesLeaderBrResultRecord
        source = "leader_br"
        if any(
            not isinstance(record, expected_record_type)
            for record in result.records
        ):
            raise StackNashPlottingError(
                f"{path} contains an invalid leader-BR record."
            )
        multiple_indices = find_u_indices_with_multiple_E_solutions(result.records)
        if multiple_indices:
            warning = multiple_E_solution_warning(len(multiple_indices))
            raise StackNashPlottingError(warning)
        records = collapse_equivalent_E_solutions(result.records)
        coordinates = [
            (float(record.result.E_star), float(record.work_item.u[0]))
            for record in records
            if record.result.error_message is None
            and record.result.equilibrium
            and np.isfinite(record.result.E_star)
        ]
    elif isinstance(result, LeaderBrCombinedResult):
        expected_record_type = LeaderBrResultRecord
        source = "leader_br_at_spbr"
        if any(
            not isinstance(record, expected_record_type)
            for record in result.records
        ):
            raise StackNashPlottingError(
                f"{path} contains an invalid leader-BR record."
            )
        coordinates = [
            (float(record.result.E_star), float(record.work_item.u_full[0]))
            for record in result.records
            if record.result.eq_status and np.isfinite(record.result.E_star)
        ]
    else:
        raise StackNashPlottingError(
            f"{path} does not contain a supported combined leader-BR result."
        )

    if expected_preset is not None and result.preset != expected_preset:
        raise StackNashPlottingError(
            f"Leader-BR results use preset {result.preset!r}, but the configured "
            f"preset is {expected_preset!r}."
        )
    if not coordinates:
        raise StackNashPlottingError(
            f"{path} contains no successful leader best-response points."
        )

    coordinates.sort(key=lambda coordinate: coordinate[1])
    return {
        "E": np.asarray([coordinate[0] for coordinate in coordinates]),
        "u": np.asarray([coordinate[1] for coordinate in coordinates]),
        "source": source,
    }


def single_species_response_curve(
    payoff_data: dict[str, Any],
    population_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one ordered species-BR trait per retained E value."""
    E = np.asarray(payoff_data["E"], dtype=float)
    N_star = np.asarray(payoff_data["N_star"], dtype=float)
    u_star = np.asarray(payoff_data["u_star"], dtype=float)
    branch = np.asarray(payoff_data["branch"], dtype=object)
    equilibrium = np.asarray(payoff_data["equilibrium"], dtype=bool)
    if N_star.ndim != 2 or u_star.ndim != 2 or N_star.shape != u_star.shape:
        raise StackNashPlottingError(
            "Single-species population and trait arrays must have matching shapes."
        )
    if N_star.shape[1] < 1 or not (
        len(E) == len(N_star) == len(branch) == len(equilibrium)
    ):
        raise StackNashPlottingError(
            "Single-species best-response fields are not aligned."
        )

    retained = (
        equilibrium
        & (branch != "extinct")
        & np.isfinite(E)
        & np.isfinite(N_star[:, 0])
        & np.isfinite(u_star[:, 0])
        & (N_star[:, 0] >= population_threshold)
    )
    first_trait_by_E: dict[float, float] = {}
    for E_value, u_value in zip(E[retained], u_star[retained, 0]):
        first_trait_by_E.setdefault(float(E_value), float(u_value))
    if not first_trait_by_E:
        raise StackNashPlottingError(
            "Analysis contains no successful single-species best-response points."
        )
    ordered_E = np.asarray(sorted(first_trait_by_E), dtype=float)
    ordered_u = np.asarray([first_trait_by_E[value] for value in ordered_E])
    return ordered_E, ordered_u

