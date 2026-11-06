"""Data types for leader best responses at species-BR equilibria."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LeaderBrWorkItem:
    """One deduplicated species-BR equilibrium submitted to a worker."""

    E: float
    branch: str
    u_full: np.ndarray
    N_full: np.ndarray
    E_idx: int
    ic_idx: int


@dataclass
class LeaderBrOptimizationResult:
    """Result of optimizing the leader decision for fixed species traits."""

    E_star: float
    Q_star: float
    eq_status: bool
    N_star_leader: np.ndarray | None


@dataclass
class LeaderBrResultRecord:
    """Associate a leader optimization result with its source work item."""

    work_item: LeaderBrWorkItem
    result: LeaderBrOptimizationResult


@dataclass
class LeaderBrChunk:
    """Persisted result of one leader-BR workflow chunk."""

    chunk_id: int
    num_chunks: int
    idx_start: int
    idx_end: int
    preset: str
    records: list[LeaderBrResultRecord]
    schema_version: int = 1


@dataclass
class LeaderBrCombinedResult:
    """Validated leader-BR records assembled from every expected chunk."""

    preset: str
    expected_num_chunks: int
    source_chunk_ids: list[int]
    source_chunk_files: list[str]
    records: list[LeaderBrResultRecord]
    combined_at: str
    schema_version: int = 1
