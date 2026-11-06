"""Typed results for single-species leader best responses over a trait grid."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SingleSpeciesLeaderBrWorkItem:
    """One fixed trait and initial population submitted to a worker."""

    u_index: int
    ic_index: int
    ic_seed: int
    u: np.ndarray
    initial_populations: np.ndarray


@dataclass
class SingleSpeciesLeaderBrOptimizationResult:
    """Leader optimum, or an explicit failed-result record."""

    E_star: float
    Q_star: float
    equilibrium: bool
    N_star: np.ndarray | None
    error_message: str | None = None


@dataclass
class SingleSpeciesLeaderBrResultRecord:
    """Associate an optimization result with its fixed-trait work item."""

    work_item: SingleSpeciesLeaderBrWorkItem
    result: SingleSpeciesLeaderBrOptimizationResult


@dataclass
class SingleSpeciesLeaderBrChunk:
    """Persisted result for one contiguous U_GRID chunk."""

    chunk_id: int
    num_chunks: int
    idx_start: int
    idx_end: int
    preset: str
    base_seed: int
    chunk_seed: int
    num_ics: int
    u_chunk: np.ndarray
    records: list[SingleSpeciesLeaderBrResultRecord]
    schema_version: int = 1


@dataclass
class SingleSpeciesLeaderBrCombinedResult:
    """Validated fixed-trait leader-BR records from all expected chunks."""

    preset: str
    expected_num_chunks: int
    source_chunk_ids: list[int]
    source_chunk_files: list[str]
    source_base_seeds: list[int]
    source_chunk_seeds: list[int]
    records: list[SingleSpeciesLeaderBrResultRecord]
    combined_at: str
    schema_version: int = 1
