"""Data types and status helpers for individual simulation results."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SimulationResult:
    """Output from a single species best-response simulation."""

    N_star: Any
    u_star: Any
    equilibrium: Any


def is_failed_result(result: SimulationResult) -> bool:
    """Return whether a result represents a timeout or a handled error."""
    if result.N_star is not None or result.u_star is not None:
        return False

    if not isinstance(result.equilibrium, dict):
        return False

    return bool(
        result.equilibrium.get("timeout")
        or result.equilibrium.get("error")
    )
