"""Condition and trajectory types shared by simulations and Figure 2 datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicsCondition:
    """One energy/evolution combination in the comparison."""

    key: str
    energy: float
    evo_dyn: bool
    energy_label: str
    linestyle: str

    @property
    def evolution_label(self) -> str:
        return "evolution enabled" if self.evo_dyn else "evolution disabled"


@dataclass(frozen=True)
class DynamicsTrajectory:
    """Plot-ready trajectory for one condition."""

    condition: DynamicsCondition
    time: np.ndarray
    population: np.ndarray
    trait: np.ndarray
    equilibrium_reached: bool


CONDITIONS = (
    DynamicsCondition("low_e_evolution", 0.6, True, "Low E", "-"),
    DynamicsCondition("low_e_no_evolution", 0.6, False, "Low E", "--"),
    DynamicsCondition("high_e_evolution", 1.0, True, "High E", "-"),
    DynamicsCondition("high_e_no_evolution", 1.0, False, "High E", "--"),
)
