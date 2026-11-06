"""Persisted data from the partly analytical, partly numerical Model 1 study."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AnalyticalResult:
    preset: str
    config: dict[str, Any]
    E: np.ndarray
    species_u: np.ndarray
    species_N: np.ndarray
    evolving_payoff: np.ndarray
    u_grid: np.ndarray
    leader_E: np.ndarray
    leader_payoff: np.ndarray
    fixed_trait_payoff: np.ndarray
    E_stackelberg: float
    u_stackelberg: float
    payoff_stackelberg: float
    E_nash: float
    u_nash: float
    nash_residual: float
    E_fixed_optimum: float
    payoff_fixed_optimum: float
    equilibrium_status: np.ndarray
    leader_optimizer_status: np.ndarray
    nash_optimizer_status: bool
