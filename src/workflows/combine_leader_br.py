"""Validate and combine single-species fixed-trait leader-BR chunks."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any

import numpy as np

from src.result_data_structures.single_species_leader_br_result_data_types import (
    SingleSpeciesLeaderBrChunk,
    SingleSpeciesLeaderBrCombinedResult,
    SingleSpeciesLeaderBrOptimizationResult,
    SingleSpeciesLeaderBrResultRecord,
    SingleSpeciesLeaderBrWorkItem,
)
from tools.data_utils import split_grid_indices


_CHUNK_FILENAME_PATTERN = re.compile(r"results_leader_br_chunk_(\d+)\.pkl")
_COMBINED_FILENAME = "results_leader_br.pkl"
_SUPPORTED_SCHEMA_VERSION = 1
_MULTIPLE_SOLUTION_THRESHOLD = 0.01


class SingleSpeciesLeaderBrCombinationError(RuntimeError):
    """Base exception for fixed-trait leader-BR combination errors."""


class MissingSingleSpeciesLeaderBrChunksError(
    SingleSpeciesLeaderBrCombinationError
):
    """Raised when an expected chunk is absent."""


class InvalidSingleSpeciesLeaderBrChunkError(
    SingleSpeciesLeaderBrCombinationError
):
    """Raised when a chunk is inconsistent with its run."""


def _read_num_chunks_from_metadata(output_dir: Path) -> int:
    """Read the expected chunk count from ``run_parameters.json``."""
    metadata_path = output_dir / "run_parameters.json"
    if not metadata_path.is_file():
        raise SingleSpeciesLeaderBrCombinationError(
            "num_chunks was not supplied and run_parameters.json is unavailable."
        )
    try:
        with metadata_path.open("r") as file:
            metadata = json.load(file)
        return int(metadata["run_config"]["num_chunks"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SingleSpeciesLeaderBrCombinationError(
            "Could not determine num_chunks from run_parameters.json."
        ) from error


def _discover_chunk_files(output_dir: Path) -> dict[int, Path]:
    """Return matching chunk paths keyed by filename chunk ID."""
    if not output_dir.is_dir():
        raise SingleSpeciesLeaderBrCombinationError(
            f"Output directory does not exist: {output_dir}"
        )
    files: dict[int, Path] = {}
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = _CHUNK_FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        chunk_id = int(match.group(1))
        if chunk_id in files:
            raise InvalidSingleSpeciesLeaderBrChunkError(
                f"Duplicate chunk ID {chunk_id} found in {output_dir}."
            )
        files[chunk_id] = path
    return files


def _invalid(path: Path, message: str) -> None:
    raise InvalidSingleSpeciesLeaderBrChunkError(f"{path.name}: {message}")


def _finite_scalar(value: Any) -> bool:
    """Return whether a value is a finite numeric scalar."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return array.ndim == 0 and bool(np.isfinite(array))


def _validate_record(
    record: Any,
    path: Path,
    record_index: int,
    expected_u_index: int,
    expected_ic_index: int,
    expected_u: np.ndarray,
) -> None:
    """Validate one nested work-item/result record and its ordering."""
    if not isinstance(record, SingleSpeciesLeaderBrResultRecord):
        _invalid(path, f"record {record_index} has an invalid record type")
    work_item = record.work_item
    result = record.result
    if not isinstance(work_item, SingleSpeciesLeaderBrWorkItem):
        _invalid(path, f"record {record_index} has an invalid work item")
    if not isinstance(result, SingleSpeciesLeaderBrOptimizationResult):
        _invalid(path, f"record {record_index} has an invalid result")

    if work_item.u_index != expected_u_index:
        _invalid(path, f"record {record_index} has an unexpected u_index")
    if work_item.ic_index != expected_ic_index:
        _invalid(path, f"record {record_index} has an unexpected ic_index")
    if not isinstance(work_item.ic_seed, (int, np.integer)):
        _invalid(path, f"record {record_index} has a non-integer IC seed")

    try:
        u = np.asarray(work_item.u, dtype=float)
        populations = np.asarray(work_item.initial_populations, dtype=float)
    except (TypeError, ValueError):
        _invalid(path, f"record {record_index} has non-numeric work-item arrays")
    if u.shape != (1,) or not np.array_equal(u, expected_u):
        _invalid(path, f"record {record_index} does not match its U_GRID value")
    if populations.shape != (1,) or not np.all(np.isfinite(populations)):
        _invalid(path, f"record {record_index} has invalid initial populations")

    if result.error_message is None:
        if not result.equilibrium:
            _invalid(path, f"record {record_index} is unsuccessful without an error")
        if not _finite_scalar(result.E_star) or not 0.0 <= float(result.E_star) <= 1.0:
            _invalid(path, f"record {record_index} has invalid E_star")
        if not _finite_scalar(result.Q_star):
            _invalid(path, f"record {record_index} has invalid Q_star")
        try:
            n_star = np.asarray(result.N_star, dtype=float)
        except (TypeError, ValueError):
            _invalid(path, f"record {record_index} has non-numeric N_star")
        if n_star.shape != (1,) or not np.all(np.isfinite(n_star)):
            _invalid(path, f"record {record_index} has invalid N_star")
    else:
        if not isinstance(result.error_message, str) or not result.error_message:
            _invalid(path, f"record {record_index} has an invalid error message")
        try:
            failed_scalars_are_nan = bool(
                np.asarray(result.E_star).ndim == 0
                and np.asarray(result.Q_star).ndim == 0
                and np.isnan(result.E_star)
                and np.isnan(result.Q_star)
            )
        except TypeError:
            failed_scalars_are_nan = False
        if result.equilibrium or not failed_scalars_are_nan or result.N_star is not None:
            _invalid(path, f"record {record_index} has inconsistent failure fields")


def _load_and_validate_chunk(
    path: Path,
    expected_chunk_id: int,
    expected_num_chunks: int,
    expected_u_grid: np.ndarray,
) -> SingleSpeciesLeaderBrChunk:
    """Load one chunk and validate its complete schema and grid coverage."""
    try:
        with path.open("rb") as file:
            chunk = pickle.load(file)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise InvalidSingleSpeciesLeaderBrChunkError(
            f"Could not load {path.name}: {error}"
        ) from error

    if not isinstance(chunk, SingleSpeciesLeaderBrChunk):
        _invalid(path, "does not contain a SingleSpeciesLeaderBrChunk")
    if chunk.schema_version != _SUPPORTED_SCHEMA_VERSION:
        _invalid(path, f"unsupported schema version {chunk.schema_version}")
    if chunk.chunk_id != expected_chunk_id:
        _invalid(path, f"contains chunk_id={chunk.chunk_id}")
    if chunk.num_chunks != expected_num_chunks:
        _invalid(path, f"contains num_chunks={chunk.num_chunks}")
    if not isinstance(chunk.preset, str) or not chunk.preset:
        _invalid(path, "preset must be a non-empty string")
    if not isinstance(chunk.num_ics, (int, np.integer)) or chunk.num_ics < 1:
        _invalid(path, "num_ics must be positive")
    if not isinstance(chunk.base_seed, (int, np.integer)) or not isinstance(
        chunk.chunk_seed,
        (int, np.integer),
    ):
        _invalid(path, "base_seed and chunk_seed must be integers")
    if chunk.chunk_seed != chunk.base_seed + chunk.chunk_id:
        _invalid(path, "chunk_seed does not equal base_seed + chunk_id")

    expected_start, expected_end = split_grid_indices(
        len(expected_u_grid),
        expected_num_chunks,
    )[expected_chunk_id]
    if (chunk.idx_start, chunk.idx_end) != (expected_start, expected_end):
        _invalid(path, "index range does not match the configured U_GRID partition")
    expected_u_chunk = expected_u_grid[expected_start:expected_end]
    try:
        u_chunk = np.asarray(chunk.u_chunk, dtype=float)
    except (TypeError, ValueError):
        _invalid(path, "u_chunk must be numeric")
    if u_chunk.shape != expected_u_chunk.shape or not np.array_equal(
        u_chunk,
        expected_u_chunk,
    ):
        _invalid(path, "u_chunk does not match the configured U_GRID")

    expected_records = len(expected_u_chunk) * chunk.num_ics
    if len(chunk.records) != expected_records:
        _invalid(path, f"contains {len(chunk.records)} records; expected {expected_records}")
    for record_index, record in enumerate(chunk.records):
        local_u_index, expected_ic_index = divmod(record_index, chunk.num_ics)
        _validate_record(
            record,
            path,
            record_index,
            expected_start + local_u_index,
            expected_ic_index,
            expected_u_chunk[local_u_index],
        )
    return chunk


def _save_pickle_atomically(
    result: SingleSpeciesLeaderBrCombinedResult,
    output_path: Path,
) -> None:
    """Save the combined result without exposing a partial final file."""
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
        raise SingleSpeciesLeaderBrCombinationError(
            f"Could not save {output_path}: {error}"
        ) from error


def find_u_indices_with_multiple_E_solutions(
    records: list[SingleSpeciesLeaderBrResultRecord],
    threshold: float = _MULTIPLE_SOLUTION_THRESHOLD,
) -> list[int]:
    """Return traits whose successful E-star replicates differ materially."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    E_values_by_u_index: dict[int, list[float]] = {}
    for record in records:
        result = record.result
        if (
            result.error_message is None
            and result.equilibrium
            and _finite_scalar(result.E_star)
        ):
            E_values_by_u_index.setdefault(record.work_item.u_index, []).append(
                float(result.E_star)
            )

    multiple_solution_indices = []
    for u_index, E_values in E_values_by_u_index.items():
        if len(E_values) > 1 and max(E_values) - min(E_values) > threshold:
            multiple_solution_indices.append(u_index)
    return sorted(multiple_solution_indices)


def multiple_E_solution_warning(
    count: int,
    threshold: float = _MULTIPLE_SOLUTION_THRESHOLD,
) -> str:
    """Format the shared warning for materially different E-star replicates."""
    label = "u value" if count == 1 else "u values"
    return (
        "WARNING: "
        f"{count} {label} have multiple E*(u) solutions separated by more "
        f"than {threshold:g}."
    )


def collapse_equivalent_E_solutions(
    records: list[SingleSpeciesLeaderBrResultRecord],
    threshold: float = _MULTIPLE_SOLUTION_THRESHOLD,
) -> list[SingleSpeciesLeaderBrResultRecord]:
    """Keep one successful record per u unless distinct solutions exist."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    records_by_u_index: dict[int, list[SingleSpeciesLeaderBrResultRecord]] = {}
    for record in records:
        records_by_u_index.setdefault(record.work_item.u_index, []).append(record)

    collapsed = []
    for u_index in sorted(records_by_u_index):
        u_records = records_by_u_index[u_index]
        successful = [
            record
            for record in u_records
            if record.result.error_message is None
            and record.result.equilibrium
            and _finite_scalar(record.result.E_star)
        ]
        if not successful:
            collapsed.append(u_records[0])
            continue

        E_values = [float(record.result.E_star) for record in successful]
        if max(E_values) - min(E_values) > threshold:
            collapsed.extend(successful)
        else:
            collapsed.append(successful[0])
    return collapsed


def combine_leader_br_chunks(
    output_dir: str | Path,
    expected_u_grid,
    num_chunks: int | None = None,
    expected_preset: str | None = None,
) -> SingleSpeciesLeaderBrCombinedResult:
    """Validate and combine every fixed-trait leader-BR chunk in grid order."""
    output_dir = Path(output_dir)
    try:
        expected_u_grid = np.asarray(expected_u_grid, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("expected_u_grid must be numeric.") from error
    if expected_u_grid.ndim != 2 or expected_u_grid.shape[1] != 1:
        raise ValueError("expected_u_grid must have shape (n, 1).")
    if len(expected_u_grid) == 0 or not np.all(np.isfinite(expected_u_grid)):
        raise ValueError("expected_u_grid must be non-empty and finite.")
    if num_chunks is None:
        num_chunks = _read_num_chunks_from_metadata(output_dir)
    if num_chunks < 1:
        raise ValueError("num_chunks must be positive.")

    discovered = _discover_chunk_files(output_dir)
    expected_ids = set(range(num_chunks))
    unexpected_ids = sorted(set(discovered) - expected_ids)
    if unexpected_ids:
        raise InvalidSingleSpeciesLeaderBrChunkError(
            f"Found chunk files outside the expected range: {unexpected_ids}."
        )
    missing_ids = sorted(expected_ids - set(discovered))
    if missing_ids:
        raise MissingSingleSpeciesLeaderBrChunksError(
            f"Missing chunk IDs: {missing_ids}."
        )

    chunks = [
        _load_and_validate_chunk(
            discovered[chunk_id],
            chunk_id,
            num_chunks,
            expected_u_grid,
        )
        for chunk_id in range(num_chunks)
    ]
    presets = {chunk.preset for chunk in chunks}
    if len(presets) != 1:
        raise InvalidSingleSpeciesLeaderBrChunkError(
            f"Chunks do not agree on preset: {sorted(presets)}."
        )
    preset = presets.pop()
    if expected_preset is not None and preset != expected_preset:
        raise InvalidSingleSpeciesLeaderBrChunkError(
            f"Chunks use preset {preset!r}, expected {expected_preset!r}."
        )
    num_ics_values = {chunk.num_ics for chunk in chunks}
    if len(num_ics_values) != 1:
        raise InvalidSingleSpeciesLeaderBrChunkError(
            "Chunks do not agree on num_ics."
        )

    records = [record for chunk in chunks for record in chunk.records]
    multiple_solution_indices = find_u_indices_with_multiple_E_solutions(records)
    if multiple_solution_indices:
        print(multiple_E_solution_warning(len(multiple_solution_indices)))
    combined_records = collapse_equivalent_E_solutions(records)

    combined = SingleSpeciesLeaderBrCombinedResult(
        preset=preset,
        expected_num_chunks=num_chunks,
        source_chunk_ids=[chunk.chunk_id for chunk in chunks],
        source_chunk_files=[discovered[chunk.chunk_id].name for chunk in chunks],
        source_base_seeds=[chunk.base_seed for chunk in chunks],
        source_chunk_seeds=[chunk.chunk_seed for chunk in chunks],
        records=combined_records,
        combined_at=datetime.now().isoformat(),
    )
    output_path = output_dir / _COMBINED_FILENAME
    _save_pickle_atomically(combined, output_path)
    print(f"Combined {len(combined.records)} records from {num_chunks} chunks.")
    print(f"Saved: {output_path}")
    return combined
