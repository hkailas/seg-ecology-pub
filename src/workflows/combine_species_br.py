"""Combine persisted species best-response chunk results.

Run after the individual chunk jobs have finished, for example as a scheduler
dependency job:

    python -m src.workflows.combine_species_br --output-dir <path> --num-chunks 70
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from src.result_data_structures.simulation_result_data_types import (
    SimulationResult,
    is_failed_result,
)
from src.result_data_structures.species_br_result_data_types import (
    SpeciesBrChunk,
    SpeciesBrResultRecord,
)
from tools.data_utils import collapse_species_to_unique_traits


_CHUNK_FILENAME_PATTERN = re.compile(r"results_species_br_chunk_(\d+)\.pkl")
_MISSING_CHUNK_POLICIES = ("stop", "combine-available")
_INCOMPLETE_IC_POLICIES = ("stop", "common-count", "zero-fill")
_MIN_POPULATION_THRESHOLD = 1e-4
_TRAIT_GROUP_THRESHOLD = 1e-3
_COLLAPSED_SPECIES_COUNT = 4
_BRANCH_BY_UNIQUE_TRAITS = {
    0: "extinct",
    1: "branch_0",
    2: "branch_1",
    3: "branch_2",
    4: "branch_3",
}


class SpeciesBrCombinationError(RuntimeError):
    """Base exception for species best-response chunk combination errors."""


class MissingChunkResultsError(SpeciesBrCombinationError):
    """Raised when one or more expected chunk files are unavailable."""


class IncompleteSpeciesBrResultsError(SpeciesBrCombinationError):
    """Raised when one or more E values have too few successful ICs."""


class InvalidSpeciesBrChunkError(SpeciesBrCombinationError):
    """Raised when a chunk file does not match the expected result schema."""


def _energy_key(E: Any) -> float:
    """Return a stable key for matching energy values across pickle fields."""
    return float(E)


def _read_num_chunks_from_metadata(output_dir: Path) -> int:
    """Read the expected chunk count from the run metadata when available."""
    metadata_path = output_dir / "run_parameters.json"
    if not metadata_path.exists():
        raise SpeciesBrCombinationError(
            "num_chunks was not supplied and run_parameters.json is unavailable. "
            "Provide --num-chunks explicitly."
        )

    try:
        with metadata_path.open("r") as file:
            metadata = json.load(file)
        return int(metadata["run_config"]["num_chunks"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SpeciesBrCombinationError(
            "Could not determine num_chunks from run_parameters.json. "
            "Provide --num-chunks explicitly."
        ) from error


def _discover_chunk_files(output_dir: Path) -> dict[int, Path]:
    """Return chunk files keyed by their parsed chunk ID."""
    if not output_dir.is_dir():
        raise SpeciesBrCombinationError(
            f"Output directory does not exist: {output_dir}"
        )

    chunk_files: dict[int, Path] = {}
    for path in output_dir.iterdir():
        if not path.is_file():
            continue

        match = _CHUNK_FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue

        chunk_id = int(match.group(1))
        if chunk_id in chunk_files:
            raise InvalidSpeciesBrChunkError(
                f"Duplicate chunk ID {chunk_id} found in {output_dir}."
            )
        chunk_files[chunk_id] = path

    return chunk_files


def _validate_chunk_payload(data: Any, path: Path, expected_chunk_id: int) -> SpeciesBrChunk:
    """Validate one chunk pickle and reconstruct its SimulationResult records."""
    if not isinstance(data, dict):
        raise InvalidSpeciesBrChunkError(f"{path.name} does not contain a dictionary.")

    required_keys = {
        "N_star",
        "u_star",
        "equilibrium",
        "E_values_order",
        "E_chunk",
        "num_ICs_E",
        "chunk_id",
        "idx_start",
        "idx_end",
        "E_skipped",
        "incomplete_E_values",
        "E_metadata",
    }
    missing_keys = sorted(required_keys.difference(data))
    if missing_keys:
        raise InvalidSpeciesBrChunkError(
            f"{path.name} is missing required keys: {', '.join(missing_keys)}."
        )

    try:
        chunk_id = int(data["chunk_id"])
        idx_start = int(data["idx_start"])
        idx_end = int(data["idx_end"])
        num_ICs_E = int(data["num_ICs_E"])
    except (TypeError, ValueError) as error:
        raise InvalidSpeciesBrChunkError(
            f"{path.name} has invalid chunk metadata."
        ) from error

    if chunk_id != expected_chunk_id:
        raise InvalidSpeciesBrChunkError(
            f"{path.name} contains chunk_id={chunk_id}, expected {expected_chunk_id}."
        )
    if idx_start < 0 or idx_end <= idx_start:
        raise InvalidSpeciesBrChunkError(
            f"{path.name} has invalid index range [{idx_start}, {idx_end})."
        )
    if num_ICs_E <= 0:
        raise InvalidSpeciesBrChunkError(
            f"{path.name} has invalid num_ICs_E={num_ICs_E}."
        )

    E_chunk = np.asarray(data["E_chunk"])
    if E_chunk.ndim != 1 or len(E_chunk) != idx_end - idx_start:
        raise InvalidSpeciesBrChunkError(
            f"{path.name} has an E_chunk incompatible with its index range."
        )

    N_star = data["N_star"]
    u_star = data["u_star"]
    equilibrium = data["equilibrium"]
    E_values_order = data["E_values_order"]
    try:
        result_lengths = {
            len(N_star),
            len(u_star),
            len(equilibrium),
            len(E_values_order),
        }
    except TypeError as error:
        raise InvalidSpeciesBrChunkError(
            f"{path.name} has non-sequence result fields."
        ) from error

    if len(result_lengths) != 1:
        raise InvalidSpeciesBrChunkError(
            f"{path.name} has mismatched N_star, u_star, equilibrium, "
            "and E_values_order lengths."
        )

    E_chunk_by_key: dict[float, float] = {}
    for E in E_chunk:
        key = _energy_key(E)
        if key in E_chunk_by_key:
            raise InvalidSpeciesBrChunkError(
                f"{path.name} contains duplicate E values in E_chunk."
            )
        E_chunk_by_key[key] = float(E)

    records = []
    for N_value, u_value, equilibrium_value, E in zip(
        N_star,
        u_star,
        equilibrium,
        E_values_order,
    ):
        key = _energy_key(E)
        if key not in E_chunk_by_key:
            raise InvalidSpeciesBrChunkError(
                f"{path.name} contains a result for E={E}, which is outside E_chunk."
            )
        records.append(
            SpeciesBrResultRecord(
                E=E_chunk_by_key[key],
                result=SimulationResult(N_value, u_value, equilibrium_value),
            )
        )

    skipped_E_values = [float(E) for E in data["E_skipped"]]
    skipped_E_keys = {_energy_key(E) for E in skipped_E_values}
    if not skipped_E_keys.issubset(E_chunk_by_key):
        raise InvalidSpeciesBrChunkError(
            f"{path.name} contains skipped E values outside E_chunk."
        )

    records_by_E: dict[float, list[SpeciesBrResultRecord]] = defaultdict(list)
    for record in records:
        records_by_E[_energy_key(record.E)].append(record)

    for skipped_E_key in skipped_E_keys:
        if records_by_E[skipped_E_key]:
            raise InvalidSpeciesBrChunkError(
                f"{path.name} contains results for an E value marked as skipped."
            )

    for E_key, E_records in records_by_E.items():
        successful_count = sum(
            not is_failed_result(record.result) for record in E_records
        )
        if successful_count > num_ICs_E:
            raise InvalidSpeciesBrChunkError(
                f"{path.name} has more than num_ICs_E results for E={E_key}."
            )

    e_metadata = data["E_metadata"]
    if not isinstance(e_metadata, dict):
        raise InvalidSpeciesBrChunkError(f"{path.name} has invalid E_metadata.")

    metadata = {
        "E_metadata": e_metadata,
        "E_IC_pairs_failed": data.get("E_IC_pairs_failed", []),
        "max_attempts": data.get("max_attempts"),
        "healing_enabled": data.get("healing_enabled"),
        "successful_count": data.get("successful_count"),
    }
    return SpeciesBrChunk(
        source_path=path,
        chunk_id=chunk_id,
        idx_start=idx_start,
        idx_end=idx_end,
        E_chunk=E_chunk,
        num_ICs_E=num_ICs_E,
        records=records,
        skipped_E_values=skipped_E_values,
        incomplete_E_values=list(data["incomplete_E_values"]),
        metadata=metadata,
    )


def _load_chunk(path: Path, expected_chunk_id: int) -> SpeciesBrChunk:
    """Load and validate one persisted chunk file."""
    try:
        with path.open("rb") as file:
            data = pickle.load(file)
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        raise InvalidSpeciesBrChunkError(
            f"Could not load chunk file {path}: {error}"
        ) from error

    return _validate_chunk_payload(data, path, expected_chunk_id)


def _validate_chunk_ranges(chunks: list[SpeciesBrChunk], require_contiguous: bool) -> None:
    """Ensure selected chunks are ordered, non-overlapping, and optionally contiguous."""
    previous_chunk: SpeciesBrChunk | None = None
    seen_E_values: set[float] = set()
    for chunk in chunks:
        if previous_chunk is not None:
            if chunk.idx_start < previous_chunk.idx_end:
                raise InvalidSpeciesBrChunkError(
                    "Selected chunks have overlapping index ranges: "
                    f"{previous_chunk.chunk_id} and {chunk.chunk_id}."
                )
            if require_contiguous and chunk.idx_start != previous_chunk.idx_end:
                raise InvalidSpeciesBrChunkError(
                    "Expected complete chunk set has a gap between ranges "
                    f"{previous_chunk.chunk_id} and {chunk.chunk_id}."
                )

        for E in chunk.E_chunk:
            E_key = _energy_key(E)
            if E_key in seen_E_values:
                raise InvalidSpeciesBrChunkError(
                    f"Selected chunks contain duplicate E={E_key}."
                )
            seen_E_values.add(E_key)
        previous_chunk = chunk


def _resolve_num_pops(output_dir: Path, chunks: list[SpeciesBrChunk]) -> int:
    """Determine the number of populations required for synthetic records."""
    metadata_path = output_dir / "run_parameters.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r") as file:
                metadata = json.load(file)
            num_pops = int(metadata["IC_PARAMS"]["num_pops"])
            if num_pops > 0:
                return num_pops
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    for chunk in chunks:
        for record in chunk.records:
            if is_failed_result(record.result) or record.result.N_star is None:
                continue
            value = np.asarray(record.result.N_star)
            if value.size > 0:
                return int(value.size)

    raise SpeciesBrCombinationError(
        "Could not determine num_pops for synthetic records. "
        "Ensure run_parameters.json is present or provide at least one valid result."
    )


def _collect_records_by_E(
    chunks: list[SpeciesBrChunk],
) -> tuple[dict[float, list[SpeciesBrResultRecord]], int]:
    """Group usable records by E and count handled failures excluded from output."""
    records_by_E: dict[float, list[SpeciesBrResultRecord]] = defaultdict(list)
    failed_result_count = 0
    for chunk in chunks:
        for record in chunk.records:
            if is_failed_result(record.result):
                failed_result_count += 1
                continue
            records_by_E[_energy_key(record.E)].append(record)
    return records_by_E, failed_result_count


def _incomplete_E_values(
    E_grid: np.ndarray,
    skipped_E_keys: set[float],
    records_by_E: dict[float, list[SpeciesBrResultRecord]],
    target_num_ICs: int,
) -> list[dict[str, Any]]:
    """Describe each non-skipped E lacking the required number of records."""
    incomplete = []
    for E in E_grid:
        E_key = _energy_key(E)
        if E_key in skipped_E_keys:
            continue

        successful_count = len(records_by_E[E_key])
        if successful_count != target_num_ICs:
            incomplete.append({
                "E": float(E),
                "successful_count": successful_count,
                "target_count": target_num_ICs,
                "missing_count": max(target_num_ICs - successful_count, 0),
            })
    return incomplete


def _synthetic_record(
    E: float,
    num_pops: int,
    equilibrium: bool,
    reason: str,
) -> SpeciesBrResultRecord:
    """Create a zero-valued synthetic result for a known missing case."""
    return SpeciesBrResultRecord(
        E=E,
        result=SimulationResult(
            N_star=np.zeros(num_pops),
            u_star=np.zeros(num_pops),
            equilibrium=equilibrium,
        ),
        is_synthetic=True,
        synthetic_reason=reason,
    )


def _build_combined_records(
    E_grid: np.ndarray,
    skipped_E_keys: set[float],
    records_by_E: dict[float, list[SpeciesBrResultRecord]],
    original_num_ICs: int,
    incomplete_policy: str,
    incomplete_E_values: list[dict[str, Any]],
    num_pops: int,
) -> tuple[list[SpeciesBrResultRecord], int, int]:
    """Normalize records to the selected incomplete-IC policy."""
    effective_num_ICs = original_num_ICs
    incomplete_keys = {_energy_key(item["E"]) for item in incomplete_E_values}

    if incomplete_E_values and incomplete_policy == "common-count":
        available_counts = [
            len(records_by_E[_energy_key(E)])
            for E in E_grid
            if _energy_key(E) not in skipped_E_keys
        ]
        if not available_counts or min(available_counts) == 0:
            raise IncompleteSpeciesBrResultsError(
                "Cannot create a common-count result because at least one "
                "non-skipped E has zero successful initial conditions."
            )
        effective_num_ICs = min(available_counts)

    combined_records = []
    synthetic_count = 0
    for E in E_grid:
        E_float = float(E)
        E_key = _energy_key(E)
        if E_key in skipped_E_keys:
            for _ in range(effective_num_ICs):
                combined_records.append(
                    _synthetic_record(
                        E_float,
                        num_pops,
                        equilibrium=True,
                        reason="non_viable_E",
                    )
                )
                synthetic_count += 1
            continue

        E_records = records_by_E[E_key]
        if incomplete_E_values and incomplete_policy == "common-count":
            combined_records.extend(E_records[:effective_num_ICs])
            continue

        combined_records.extend(E_records)
        if E_key in incomplete_keys and incomplete_policy == "zero-fill":
            missing_count = original_num_ICs - len(E_records)
            for _ in range(missing_count):
                combined_records.append(
                    _synthetic_record(
                        E_float,
                        num_pops,
                        equilibrium=False,
                        reason="incomplete_E",
                    )
                )
                synthetic_count += 1

    return combined_records, effective_num_ICs, synthetic_count


def _combined_E_metadata(chunks: list[SpeciesBrChunk]) -> dict[float, Any]:
    """Merge per-E metadata from selected chunk files."""
    combined_metadata = {}
    for chunk in chunks:
        for E, metadata in chunk.metadata["E_metadata"].items():
            E_key = _energy_key(E)
            if E_key in combined_metadata:
                raise InvalidSpeciesBrChunkError(
                    f"Duplicate E_metadata entry for E={E_key}."
                )
            combined_metadata[E_key] = metadata
    return combined_metadata


def _collapse_species_br_records(
    records: list[SpeciesBrResultRecord],
    min_population_threshold: float = _MIN_POPULATION_THRESHOLD,
    trait_group_threshold: float = _TRAIT_GROUP_THRESHOLD,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Collapse every species-BR record to four unique-trait group slots."""
    collapsed_N_star = []
    collapsed_u_star = []
    unique_trait_counts = []
    for record in records:
        result = record.result
        collapsed = collapse_species_to_unique_traits(
            np.asarray(result.N_star),
            np.asarray(result.u_star),
            min_thresh=min_population_threshold,
            u_thresh=trait_group_threshold,
        )
        N_collapsed, u_collapsed, num_unique_traits = collapsed
        collapsed_N_star.append(N_collapsed)
        collapsed_u_star.append(u_collapsed)
        unique_trait_counts.append(num_unique_traits)
    return collapsed_N_star, collapsed_u_star, unique_trait_counts


def _bifurcation_labels(
    records: list[SpeciesBrResultRecord],
    unique_trait_counts: list[int] | None = None,
) -> list[str]:
    """Classify complete species-BR records by surviving trait-group count."""
    if unique_trait_counts is None:
        _, _, unique_trait_counts = _collapse_species_br_records(records)
    if len(records) != len(unique_trait_counts):
        raise ValueError(
            "records and unique_trait_counts must have matching lengths."
        )

    labels = []
    for record, num_unique_traits in zip(records, unique_trait_counts):
        if not bool(record.result.equilibrium):
            labels.append("extinct")
            continue
        labels.append(
            _BRANCH_BY_UNIQUE_TRAITS.get(num_unique_traits, "extinct")
        )
    return labels


def _warn_bifurcation_labels_omitted() -> None:
    """Print a prominent warning for partial outputs without branch labels."""
    border = "!" * 78
    print(f"\n{border}")
    print("MAJOR WARNING: BIFURCATION BRANCH LABELS WERE NOT ADDED")
    print("The combined species-BR result contains incomplete simulation data.")
    print("Complete every E value with all requested ICs before adding labels.")
    print(f"{border}\n")


def _save_pickle_atomically(payload: dict[str, Any], output_path: Path) -> None:
    """Write a pickle without exposing a partially written final file."""
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
            pickle.dump(payload, temporary_file)
        temporary_path.replace(output_path)
    except OSError as error:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise SpeciesBrCombinationError(
            f"Could not save combined result to {output_path}: {error}"
        ) from error


def combine_species_br_chunk_results(
    output_dir: str | Path,
    num_chunks: int | None = None,
    missing_chunks: str = "stop",
    incomplete_ics: str = "stop",
) -> dict[str, Any]:
    """Validate, combine, normalize, and save species best-response chunks.

    A complete input produces ``results_species_br.pkl``. Inputs combined with
    missing chunks or an incomplete-IC policy produce
    ``results_species_br_partial.pkl`` instead. Persisted population and trait
    records are collapsed and zero-padded to four unique-trait group slots.
    """
    if missing_chunks not in _MISSING_CHUNK_POLICIES:
        raise ValueError(
            f"missing_chunks must be one of {_MISSING_CHUNK_POLICIES}."
        )
    if incomplete_ics not in _INCOMPLETE_IC_POLICIES:
        raise ValueError(
            f"incomplete_ics must be one of {_INCOMPLETE_IC_POLICIES}."
        )

    output_dir = Path(output_dir)
    if num_chunks is None:
        num_chunks = _read_num_chunks_from_metadata(output_dir)
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive.")

    discovered_files = _discover_chunk_files(output_dir)
    expected_chunk_ids = set(range(num_chunks))
    discovered_chunk_ids = set(discovered_files)
    unexpected_chunk_ids = sorted(discovered_chunk_ids.difference(expected_chunk_ids))
    if unexpected_chunk_ids:
        raise InvalidSpeciesBrChunkError(
            "Found chunk files outside the expected range: "
            f"{unexpected_chunk_ids}."
        )

    missing_chunk_ids = sorted(expected_chunk_ids.difference(discovered_chunk_ids))
    if missing_chunk_ids and missing_chunks == "stop":
        raise MissingChunkResultsError(
            "Not all chunk results are available. Missing chunk IDs: "
            f"{missing_chunk_ids}. Re-run with --missing-chunks "
            "combine-available to save a partial result from available chunks."
        )

    selected_chunk_ids = sorted(discovered_chunk_ids)
    if not selected_chunk_ids:
        raise MissingChunkResultsError("No species best-response chunk files were found.")

    chunks = [
        _load_chunk(discovered_files[chunk_id], chunk_id)
        for chunk_id in selected_chunk_ids
    ]
    chunks.sort(key=lambda chunk: chunk.idx_start)
    _validate_chunk_ranges(
        chunks,
        require_contiguous=not missing_chunk_ids,
    )

    num_ICs_per_chunk = {chunk.num_ICs_E for chunk in chunks}
    if len(num_ICs_per_chunk) != 1:
        raise InvalidSpeciesBrChunkError(
            "Selected chunks do not agree on num_ICs_E."
        )
    original_num_ICs = num_ICs_per_chunk.pop()

    E_grid = np.concatenate([chunk.E_chunk for chunk in chunks])
    skipped_E_values = [
        E
        for chunk in chunks
        for E in chunk.skipped_E_values
    ]
    skipped_E_keys = {_energy_key(E) for E in skipped_E_values}
    records_by_E, failed_result_count = _collect_records_by_E(chunks)
    incomplete_E_values = _incomplete_E_values(
        E_grid,
        skipped_E_keys,
        records_by_E,
        original_num_ICs,
    )

    if incomplete_E_values and incomplete_ics == "stop":
        incomplete_summary = ", ".join(
            f"E={item['E']} ({item['successful_count']}/{item['target_count']})"
            for item in incomplete_E_values
        )
        raise IncompleteSpeciesBrResultsError(
            "Some E values have incomplete initial conditions: "
            f"{incomplete_summary}. Re-run with --incomplete-ics common-count "
            "or --incomplete-ics zero-fill to save a partial result."
        )

    num_pops = _resolve_num_pops(output_dir, chunks)
    combined_records, effective_num_ICs, synthetic_result_count = _build_combined_records(
        E_grid,
        skipped_E_keys,
        records_by_E,
        original_num_ICs,
        incomplete_ics,
        incomplete_E_values,
        num_pops,
    )
    collapsed_N_star, collapsed_u_star, unique_trait_counts = (
        _collapse_species_br_records(combined_records)
    )

    is_partial = bool(missing_chunk_ids or incomplete_E_values)
    output_filename = (
        "results_species_br_partial.pkl"
        if is_partial
        else "results_species_br.pkl"
    )
    output_path = output_dir / output_filename

    payload = {
        "N_star": collapsed_N_star,
        "u_star": collapsed_u_star,
        "equilibrium": [record.result.equilibrium for record in combined_records],
        "E_values_order": [record.E for record in combined_records],
        "E_grid": E_grid,
        "num_ICs_E": effective_num_ICs,
        "original_num_ICs_E": original_num_ICs,
        "E_metadata": _combined_E_metadata(chunks),
        "E_skipped": skipped_E_values,
        "incomplete_E_values": incomplete_E_values,
        "combination_status": "partial" if is_partial else "complete",
        "source_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "source_chunk_files": [chunk.source_path.name for chunk in chunks],
        "expected_num_chunks": num_chunks,
        "missing_chunk_ids": missing_chunk_ids,
        "incomplete_IC_policy": incomplete_ics,
        "missing_chunk_policy": missing_chunks,
        "synthetic_result_count": synthetic_result_count,
        "synthetic_non_viable_E_count": sum(
            record.synthetic_reason == "non_viable_E"
            for record in combined_records
        ),
        "synthetic_incomplete_E_count": sum(
            record.synthetic_reason == "incomplete_E"
            for record in combined_records
        ),
        "failed_result_count_excluded": failed_result_count,
        "species_vectors_collapsed": True,
        "collapsed_species_count": _COLLAPSED_SPECIES_COUNT,
        "collapse_min_population_threshold": _MIN_POPULATION_THRESHOLD,
        "collapse_trait_group_threshold": _TRAIT_GROUP_THRESHOLD,
        "combined_at": datetime.now().isoformat(),
    }
    if is_partial:
        _warn_bifurcation_labels_omitted()
    else:
        payload["bifurcation_labels"] = _bifurcation_labels(
            combined_records,
            unique_trait_counts,
        )

    _save_pickle_atomically(payload, output_path)
    print(
        f"Saved {payload['combination_status']} species best-response result "
        f"to {output_path}"
    )
    return payload


def main() -> None:
    """Run the chunk combiner from the command line."""
    parser = argparse.ArgumentParser(
        description="Combine species best-response chunk result files."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing results_species_br_chunk_<id>.pkl files.",
    )
    parser.add_argument(
        "--num-chunks",
        required=True,
        type=int,
        help="Total number of chunk jobs expected for this run.",
    )
    parser.add_argument(
        "--missing-chunks",
        choices=_MISSING_CHUNK_POLICIES,
        default="stop",
        help="Policy when one or more expected chunk files are missing.",
    )
    parser.add_argument(
        "--incomplete-ics",
        choices=_INCOMPLETE_IC_POLICIES,
        default="stop",
        help="Policy when an included non-skipped E has too few ICs.",
    )
    args = parser.parse_args()

    try:
        combine_species_br_chunk_results(
            output_dir=args.output_dir,
            num_chunks=args.num_chunks,
            missing_chunks=args.missing_chunks,
            incomplete_ics=args.incomplete_ics,
        )
    except SpeciesBrCombinationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
