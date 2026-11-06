"""Stored adaptive fitness curves and the resident states that generated them."""

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptiveLandscapeResult:
    key: str
    preset: str
    model_name: str
    energy: float
    initial_populations: np.ndarray
    initial_traits: np.ndarray
    resident_populations: np.ndarray
    resident_traits: np.ndarray
    resident_time: float
    selection_rule: str
    extinction_threshold: float | None
    mutant_traits: np.ndarray
    mutant_fitness: np.ndarray
    resident_fitness: np.ndarray
    x_limits: tuple[float, float] | None
    y_limits: tuple[float, float] | None
