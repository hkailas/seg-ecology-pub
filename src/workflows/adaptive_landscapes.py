"""Run resident dynamics and evaluate fitness landscapes from explicit presets."""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path

import numpy as np

from src.models.eco_evo_model import G, G_model3
from src.result_data_structures.adaptive_landscape_result_data_types import AdaptiveLandscapeResult
from src.simulate.simulation_core import run_simulation

FITNESS = {"model1": G, "model3": G_model3}


def simulate_resident_state(config: dict, populations: np.ndarray, traits: np.ndarray,
                            E: float, *, threshold: float = 1e-3):
    if populations.ndim != 1 or traits.ndim != 1 or populations.shape != traits.shape:
        raise ValueError("Resident populations and traits must be aligned vectors.")
    if not len(populations) or np.any(populations < 0):
        raise ValueError("Resident populations must be nonempty and nonnegative.")
    model_params = dict(config["MODEL_PARAMS"], E=float(E))
    configured = config["SIMULATION_PARAMS"]
    permitted = inspect.signature(run_simulation).parameters
    simulation_params = {key: value for key, value in configured.items()
                         if key in permitted and key not in model_params}
    model_name = simulation_params.get("model_name", "model1")
    if model_name not in FITNESS:
        raise ValueError(f"Unsupported adaptive landscape model {model_name}.")
    simulation_params["store_sol"] = model_name == "model3"
    simulation_params["stop_at_equilibrium"] = True
    solution = run_simulation({"populations": populations.tolist(),
                               "traits": traits.tolist()},
                              **model_params, **simulation_params)
    states, times = np.asarray(solution.y), np.asarray(solution.t)
    if states.ndim != 2 or states.shape[0] != 2 * len(populations):
        raise RuntimeError("Unexpected resident-state shape.")
    if model_name == "model3":
        eligible = np.flatnonzero(np.all(states[:len(populations)] > threshold, axis=0))
        if not eligible.size:
            raise RuntimeError("No Model 3 state lies above the extinction threshold.")
        index = int(eligible[-1])
        rule = "last stored state with all residents above threshold"
    else:
        index = -1
        rule = "last stored state"
    return (states[:len(populations), index].copy(),
            states[len(populations):, index].copy(), float(times[index]),
            rule, model_name)


def evaluate_adaptive_landscape(config: dict, key: str, populations,
                                traits, E: float, *, x_limits=None,
                                y_limits=None, threshold: float = 1e-3,
                                mutant_range=(0.0, 25.0), points=1000):
    populations = np.asarray(populations, dtype=float)
    traits = np.asarray(traits, dtype=float)
    resident_N, resident_u, time, rule, model_name = simulate_resident_state(
        config, populations, traits, E, threshold=threshold)
    model = dict(config["MODEL_PARAMS"], E=float(E))
    args = tuple(model[name] for name in (
        "r_max", "K", "g", "alpha", "beta", "beta_bully",
        "sigma_a_squared", "E", "d", "include_E")) + (model.get("fd", True),)
    fitness = FITNESS[model_name]
    v = np.linspace(*mutant_range, points)
    values = np.asarray([fitness(value, resident_u, resident_N, *args)
                         for value in v], dtype=float)
    resident_values = np.asarray([fitness(value, resident_u, resident_N, *args)
                                  for value in resident_u], dtype=float)
    return AdaptiveLandscapeResult(
        key=key, preset=config["PRESET_NAME"], model_name=model_name,
        energy=float(E), initial_populations=populations,
        initial_traits=traits, resident_populations=resident_N,
        resident_traits=resident_u, resident_time=time, selection_rule=rule,
        extinction_threshold=threshold if model_name == "model3" else None,
        mutant_traits=v, mutant_fitness=values, resident_fitness=resident_values,
        x_limits=x_limits, y_limits=y_limits)


def save_landscapes(results: dict[str, AdaptiveLandscapeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(results, stream)


def load_landscapes(path: Path, expected_keys: set[str],
                    expected_preset: str | None = None):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run main_adaptive_landscapes.py --stage compute.")
    with path.open("rb") as stream:
        results = pickle.load(stream)
    if not isinstance(results, dict) or not expected_keys.issubset(results):
        raise ValueError(f"Incomplete adaptive landscapes in {path}.")
    if any(not isinstance(value, AdaptiveLandscapeResult) for value in results.values()):
        raise ValueError(f"Invalid adaptive landscape result in {path}.")
    if expected_preset and any(value.preset != expected_preset for value in results.values()):
        raise ValueError(f"Adaptive landscape preset mismatch in {path}.")
    return results
