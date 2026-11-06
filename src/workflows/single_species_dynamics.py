"""Run the four standard single-species dynamics conditions."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Callable, Mapping

import numpy as np

from src.plotting.single_species_dynamics import (
    CONDITIONS,
    DynamicsTrajectory,
)
from src.simulate.simulation_core import run_simulation


_RUN_SIMULATION_KEYS = {
    "t_start",
    "t_max",
    "eq_thresh",
    "abs_thresh",
    "eq_tsteps",
    "chunk_size",
    "verbose",
    "evo_dyn",
    "min_thresh",
    "store_sol",
    "maxstep",
    "method",
    "model_name",
    "stop_at_equilibrium",
}


def _finite_number(value: object, name: str) -> float:
    """Return a finite float or raise a configuration error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def validate_config(
    config: Mapping[str, object],
    time_limit: float | None = None,
) -> tuple[float, float]:
    """Validate the single-species preset contract and requested end time."""
    ic_params = config.get("IC_PARAMS")
    if not isinstance(ic_params, Mapping):
        raise ValueError("The preset must define an IC_PARAMS mapping.")
    if ic_params.get("num_pops") != 1:
        raise ValueError("Single-species dynamics require IC_PARAMS.num_pops == 1.")
    if "initial_population" not in ic_params:
        raise ValueError("The preset must define IC_PARAMS.initial_population.")
    if "initial_trait" not in ic_params:
        raise ValueError("The preset must define IC_PARAMS.initial_trait.")

    initial_population = _finite_number(
        ic_params["initial_population"], "IC_PARAMS.initial_population"
    )
    initial_trait = _finite_number(
        ic_params["initial_trait"], "IC_PARAMS.initial_trait"
    )
    if initial_population < 0:
        raise ValueError("IC_PARAMS.initial_population must be nonnegative.")

    simulation_params = config.get("SIMULATION_PARAMS")
    if not isinstance(simulation_params, Mapping):
        raise ValueError("The preset must define a SIMULATION_PARAMS mapping.")
    t_start = _finite_number(
        simulation_params.get("t_start", 0.0), "SIMULATION_PARAMS.t_start"
    )
    if time_limit is not None:
        final_time = _finite_number(time_limit, "--time-limit")
        if final_time <= t_start:
            raise ValueError(
                f"--time-limit must be greater than t_start ({t_start:g})."
            )

    model_params = config.get("MODEL_PARAMS")
    if not isinstance(model_params, Mapping):
        raise ValueError("The preset must define a MODEL_PARAMS mapping.")
    return initial_population, initial_trait


def _simulation_parameters(
    configured: Mapping[str, object],
    *,
    evo_dyn: bool,
    time_limit: float | None,
) -> dict[str, object]:
    """Select solver arguments and apply condition-specific overrides."""
    parameters = {
        key: value for key, value in configured.items() if key in _RUN_SIMULATION_KEYS
    }
    parameters["store_sol"] = True
    parameters["evo_dyn"] = evo_dyn
    parameters["stop_at_equilibrium"] = time_limit is None
    if time_limit is not None:
        parameters["t_max"] = float(time_limit)
    return parameters


def run_single_species_dynamics(
    config: Mapping[str, object],
    *,
    time_limit: float | None = None,
    num_plot_points: int = 2000,
    simulation_runner: Callable[..., object] = run_simulation,
) -> dict[str, DynamicsTrajectory]:
    """Run the four conditions and return evenly sampled trajectories."""
    initial_population, initial_trait = validate_config(config, time_limit)
    if (
        isinstance(num_plot_points, bool)
        or not isinstance(num_plot_points, Integral)
        or num_plot_points < 2
    ):
        raise ValueError("num_plot_points must be an integer of at least 2.")

    model_params = dict(config["MODEL_PARAMS"])
    configured_simulation_params = config["SIMULATION_PARAMS"]
    initial_conditions = {
        "populations": [initial_population],
        "traits": [initial_trait],
    }
    trajectories: dict[str, DynamicsTrajectory] = {}

    for condition in CONDITIONS:
        condition_model_params = dict(model_params)
        condition_model_params["E"] = condition.energy
        simulation_params = _simulation_parameters(
            configured_simulation_params,
            evo_dyn=condition.evo_dyn,
            time_limit=time_limit,
        )
        solution = simulation_runner(
            {
                "populations": initial_conditions["populations"].copy(),
                "traits": initial_conditions["traits"].copy(),
            },
            **condition_model_params,
            **simulation_params,
        )

        start_time = float(solution.t[0])
        final_time = float(solution.t[-1])
        sample_times = np.linspace(start_time, final_time, num_plot_points)
        sampled = np.asarray(solution.sol(sample_times))
        if sampled.shape != (2, num_plot_points):
            raise RuntimeError(
                "A single-species solution must contain one population and one trait."
            )
        trajectories[condition.key] = DynamicsTrajectory(
            condition=condition,
            time=sample_times,
            population=sampled[0],
            trait=sampled[1],
            equilibrium_reached=bool(solution.equilibrium),
        )

    return trajectories

