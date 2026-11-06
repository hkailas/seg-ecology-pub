"""Persisted result type for Stackelberg-payoff and Nash analysis."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AnalysisResult:
    """All derived analysis data from species-BR and leader-BR results."""

    preset: str
    species_br_payoff_data: dict[str, Any]
    branch_data: dict[str, dict[str, np.ndarray]]
    nash_points: list[dict[str, Any]]
    nash_payoff_curves: list[dict[str, Any]]
    nash_interpolation: bool
    nash_residual_threshold: float | None
    species_results_file: str
    leader_results_file: str
