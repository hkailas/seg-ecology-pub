"""Combine leader best-response chunk files into one validated result.

The workflow expects files written by ``leader_br_at_species_br.py`` and saves
``results_leader_br_at_spbr.pkl`` in the same output directory.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any

from src.result_data_structures.leader_br_at_species_br_result_data_types import (
    LeaderBrChunk,
    LeaderBrCombinedResult,
    LeaderBrOptimizationResult,
    LeaderBrResultRecord,
    LeaderBrWorkItem,
)


_CHUNK_FILENAME_PATTERN = re.compile(
    r"results_leader_br_at_spbr_chunk_(\d+)\.pkl"
)
_COMBINED_FILENAME = "results_leader_br_at_spbr.pkl"
_SUPPORTED_SCHEMA_VERSION = 1


class LeaderBrCombinationError(RuntimeError):
    """Base exception for leader-BR chunk combination errors."""


class MissingLeaderBrChunksError(LeaderBrCombinationError):
    """Raised when one or more expected leader-BR chunks are unavailable."""


class InvalidLeaderBrChunkError(LeaderBrCombinationError):
    """Raised when a chunk does not match the expected schema or run."""


def _read_num_chunks_from_metadata(output_dir: Path) -> int:
    """Read the expected number of chunks from the run metadata."""
    metadata_path = output_dir / "run_parameters.json"
    if not metadata_path.is_file():
        raise LeaderBrCombinationError(
            "num_chunks was not supplied and run_parameters.json is unavailable. "
            "Provide num_chunks explicitly."
        )

    try:
        with metadata_path.open("r") as file:
            metadata = json.load(file)
        return int(metadata["run_config"]["num_chunks"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LeaderBrCombinationError(
            "Could not determine num_chunks from run_parameters.json. "
            "Provide num_chunks explicitly."
        ) from error


def _discover_chunk_files(output_dir: Path) -> dict[int, Path]:
    """Return leader-BR chunk paths keyed by the ID in each filename."""
    if not output_dir.is_dir():
        raise LeaderBrCombinationError(
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
            raise InvalidLeaderBrChunkError(
                f"Duplicate chunk ID {chunk_id} found in {output_dir}."
            )
        chunk_files[chunk_id] = path

    return chunk_files


def _validate_record(record: Any, path: Path, record_index: int) -> None:
    """Validate the nested dataclass types in one result record."""
    if not isinstance(record, LeaderBrResultRecord):
        raise InvalidLeaderBrChunkError(
            f"{path.name} record {record_index} is not a LeaderBrResultRecord."
        )
    if not isinstance(record.work_item, LeaderBrWorkItem):
        raise InvalidLeaderBrChunkError(
            f"{path.name} record {record_index} has an invalid work item."
        )
    if not isinstance(record.result, LeaderBrOptimizationResult):
        raise InvalidLeaderBrChunkError(
            f"{path.name} record {record_index} has an invalid optimization result."
        )


def _load_and_validate_chunk(
    path: Path,
    expected_chunk_id: int,
    expected_num_chunks: int,
) -> LeaderBrChunk:
    """Load one chunk and validate its schema and chunk metadata."""
    try:
        with path.open("rb") as file:
            chunk = pickle.load(file)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise InvalidLeaderBrChunkError(
            f"Could not load {path.name}: {error}"
        ) from error

    if not isinstance(chunk, LeaderBrChunk):
        raise InvalidLeaderBrChunkError(
            f"{path.name} does not contain a LeaderBrChunk."
        )
    if chunk.schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise InvalidLeaderBrChunkError(
            f"{path.name} uses unsupported schema version "
            f"{chunk.schema_version}."
        )
    if chunk.chunk_id != expected_chunk_id:
        raise InvalidLeaderBrChunkError(
            f"{path.name} contains chunk_id={chunk.chunk_id}, "
            f"expected {expected_chunk_id}."
        )
    if chunk.num_chunks != expected_num_chunks:
        raise InvalidLeaderBrChunkError(
            f"{path.name} contains num_chunks={chunk.num_chunks}, "
            f"expected {expected_num_chunks}."
        )
    if chunk.idx_start < 0 or chunk.idx_end < chunk.idx_start:
        raise InvalidLeaderBrChunkError(
            f"{path.name} has invalid index range "
            f"[{chunk.idx_start}, {chunk.idx_end})."
        )
    if len(chunk.records) != chunk.idx_end - chunk.idx_start:
        raise InvalidLeaderBrChunkError(
            f"{path.name} has {len(chunk.records)} records for index range "
            f"[{chunk.idx_start}, {chunk.idx_end})."
        )

    for record_index, record in enumerate(chunk.records):
        _validate_record(record, path, record_index)

    return chunk


def _validate_chunk_ranges(chunks: list[LeaderBrChunk]) -> None:
    """Require complete, contiguous work-list coverage in chunk-ID order."""
    expected_start = 0
    for chunk in chunks:
        if chunk.idx_start != expected_start:
            raise InvalidLeaderBrChunkError(
                f"Chunk {chunk.chunk_id} starts at index {chunk.idx_start}; "
                f"expected {expected_start}. Chunk ranges are not contiguous."
            )
        expected_start = chunk.idx_end


def _save_pickle_atomically(
    combined: LeaderBrCombinedResult,
    output_path: Path,
) -> None:
    """Save without exposing a partially written final pickle."""
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
            pickle.dump(combined, temporary_file)
        temporary_path.replace(output_path)
    except OSError as error:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise LeaderBrCombinationError(
            f"Could not save combined result to {output_path}: {error}"
        ) from error


def combine_leader_br_at_species_br_chunks(
    output_dir: str | Path,
    num_chunks: int | None = None,
    expected_preset: str | None = None,
) -> LeaderBrCombinedResult:
    """Validate and combine every expected leader-BR chunk in work-list order."""
    output_dir = Path(output_dir)
    if num_chunks is None:
        num_chunks = _read_num_chunks_from_metadata(output_dir)
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive.")

    discovered_files = _discover_chunk_files(output_dir)
    expected_chunk_ids = set(range(num_chunks))
    discovered_chunk_ids = set(discovered_files)

    unexpected_chunk_ids = sorted(discovered_chunk_ids - expected_chunk_ids)
    if unexpected_chunk_ids:
        raise InvalidLeaderBrChunkError(
            "Found chunk files outside the expected range: "
            f"{unexpected_chunk_ids}."
        )

    missing_chunk_ids = sorted(expected_chunk_ids - discovered_chunk_ids)
    if missing_chunk_ids:
        raise MissingLeaderBrChunksError(
            "Not all leader-BR chunk results are available. Missing chunk IDs: "
            f"{missing_chunk_ids}."
        )

    chunks = [
        _load_and_validate_chunk(
            discovered_files[chunk_id],
            expected_chunk_id=chunk_id,
            expected_num_chunks=num_chunks,
        )
        for chunk_id in range(num_chunks)
    ]
    _validate_chunk_ranges(chunks)

    presets = {chunk.preset for chunk in chunks}
    if len(presets) != 1:
        raise InvalidLeaderBrChunkError(
            f"Leader-BR chunks do not agree on preset: {sorted(presets)}."
        )
    preset = presets.pop()
    if expected_preset is not None and preset != expected_preset:
        raise InvalidLeaderBrChunkError(
            f"Leader-BR chunks use preset {preset!r}, but the configured preset "
            f"is {expected_preset!r}."
        )

    combined = LeaderBrCombinedResult(
        preset=preset,
        expected_num_chunks=num_chunks,
        source_chunk_ids=[chunk.chunk_id for chunk in chunks],
        source_chunk_files=[
            discovered_files[chunk.chunk_id].name for chunk in chunks
        ],
        records=[record for chunk in chunks for record in chunk.records],
        combined_at=datetime.now().isoformat(),
    )
    output_path = output_dir / _COMBINED_FILENAME
    _save_pickle_atomically(combined, output_path)
    print(
        f"Combined {len(combined.records)} leader best-response records "
        f"from {num_chunks} chunks."
    )
    print(f"Saved combined leader best-response result to {output_path}")
    return combined


def main() -> None:
    """Run the leader-BR chunk combiner with an explicit output directory."""
    parser = argparse.ArgumentParser(
        description="Combine leader best-response-at-species-BR chunk files."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing results_leader_br_at_spbr_chunk_<id>.pkl files.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        help="Expected chunk count; defaults to run_parameters.json.",
    )
    args = parser.parse_args()

    try:
        combine_leader_br_at_species_br_chunks(
            output_dir=args.output_dir,
            num_chunks=args.num_chunks,
        )
    except LeaderBrCombinationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
