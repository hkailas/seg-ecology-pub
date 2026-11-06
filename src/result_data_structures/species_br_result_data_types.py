"""Data types for persisted and combined species best-response results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .simulation_result_data_types import SimulationResult


@dataclass
class SpeciesBrResultRecord:
    """A simulation result paired with the energy value that produced it."""

    E: float
    result: SimulationResult
    is_synthetic: bool = False
    synthetic_reason: str | None = None


@dataclass
class SpeciesBrChunk:
    """Validated contents of one persisted species best-response chunk."""

    source_path: Path
    chunk_id: int
    idx_start: int
    idx_end: int
    E_chunk: np.ndarray
    num_ICs_E: int
    records: list[SpeciesBrResultRecord]
    skipped_E_values: list[float]
    incomplete_E_values: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
