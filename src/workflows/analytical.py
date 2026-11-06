"""Model 1 analytical responses with numerical equilibrium and payoff solves."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fminbound, fsolve, minimize_scalar

from src.result_data_structures.analytical_result_data_types import AnalyticalResult


def validate_analytical_config(config: dict) -> None:
    for section in ("MODEL_PARAMS", "ECONOMIC_PARAMS", "ANALYTICAL_PARAMS", "OUTPUT_CONFIG"):
        if not config.get(section):
            raise ValueError(f"Missing {section} for analytical workflow.")
    model, economic, settings = (config[key] for key in
                                 ("MODEL_PARAMS", "ECONOMIC_PARAMS", "ANALYTICAL_PARAMS"))
    for key in ("r_max", "g", "alpha", "beta", "d", "K"):
        if not np.isfinite(model[key]):
            raise ValueError(f"MODEL_PARAMS.{key} must be finite.")
    for key in ("L", "p", "delta", "alpha_leader", "c1"):
        if not np.isfinite(economic[key]):
            raise ValueError(f"ECONOMIC_PARAMS.{key} must be finite.")
    if model["g"] <= 0 or model["alpha"] <= 0 or model["K"] <= 0 or economic["delta"] <= 0:
        raise ValueError("g, alpha, K, and delta must be positive.")
    if settings["E_points"] < 2 or settings["u_points"] < 2:
        raise ValueError("Analytical grids require at least two points.")


def growth_rate(u: float | np.ndarray, model: dict) -> float | np.ndarray:
    return model["r_max"] * np.exp(-model["g"] * u)


def equilibrium_population(E: float, u: float, model: dict) -> tuple[float, str]:
    """Retain BR_analyt's root solve and extinction test; expose solver failure."""
    K, d, a, b = (model[key] for key in ("K", "d", "alpha", "beta"))

    def growth(N: float) -> float:
        return growth_rate(u, model) * (E * (b + u) / a - 1) * (1 - N / K) - d

    if growth(0) < 0:
        return 0.0, "extinct"
    try:
        solution, _, ier, _ = fsolve(growth, K / 2, full_output=True)
    except (ArithmeticError, ValueError, RuntimeError) as error:
        return float("nan"), f"solver_error:{type(error).__name__}"
    value = float(solution[0])
    if ier != 1 or not np.isfinite(value):
        return float("nan"), "solver_failure"
    if value < 0 or value > K:
        return 0.0, "outside_capacity"
    return value, "survives"


def payoff(E: float, N: float, u: float, model: dict, economic: dict) -> float:
    base = economic["L"] * (1 - E) + economic["p"] / economic["delta"] * (
        1 - np.exp(-economic["alpha_leader"] * N)
    )
    if N == 0:
        return float(base)
    if u == 0 and economic["c1"]:
        raise ValueError("c1/u is undefined for a living population at u=0.")
    return float(base + (economic["c1"] / u if economic["c1"] else 0.0))


def compute_analytical(config: dict) -> AnalyticalResult:
    validate_analytical_config(config)
    model, economic, settings = (config[key] for key in
                                 ("MODEL_PARAMS", "ECONOMIC_PARAMS", "ANALYTICAL_PARAMS"))
    E_values = np.linspace(settings["E_min"], settings["E_max"], settings["E_points"])
    u_grid = np.linspace(settings["u_min"], settings["u_max"], settings["u_points"])
    species_u = np.empty_like(E_values)
    species_N = np.empty_like(E_values)
    evolving_payoff = np.empty_like(E_values)
    statuses = []
    for i, E in enumerate(E_values):
        u = 1 / model["g"] + model["alpha"] / E - model["beta"]
        N, status = equilibrium_population(float(E), float(u), model)
        statuses.append(status)
        species_u[i] = 0 if N == 0 else u
        species_N[i] = N
        evolving_payoff[i] = payoff(float(E), float(N), float(species_u[i]), model, economic)

    leader_E = np.empty_like(u_grid)
    leader_payoff = np.empty_like(u_grid)
    leader_status = []
    for i, u in enumerate(u_grid):
        def objective(E: float) -> float:
            N, status = equilibrium_population(E, float(u), model)
            if status.startswith("solver_"):
                raise RuntimeError(status)
            return payoff(E, N, float(u), model, economic)
        try:
            E_opt, _, optimizer_error, _ = fminbound(
                lambda E: -objective(E), settings["leader_E_min"],
                settings["E_max"], full_output=True)
            if optimizer_error:
                raise RuntimeError("Leader optimizer did not converge.")
            E_opt = float(E_opt)
            leader_E[i] = E_opt
            leader_payoff[i] = objective(E_opt)
            leader_status.append(True)
        except (ValueError, RuntimeError, ArithmeticError):
            leader_E[i] = np.nan
            leader_payoff[i] = np.nan
            leader_status.append(False)

    if not np.any(np.isfinite(evolving_payoff)):
        raise RuntimeError("No finite evolution-aware payoff.")
    stack_index = int(np.nanargmax(evolving_payoff))
    valid_leader = np.isfinite(leader_E)
    if np.count_nonzero(valid_leader) < 2:
        raise RuntimeError("Insufficient valid leader responses for Nash analysis.")
    E_interp = interp1d(u_grid[valid_leader], leader_E[valid_leader],
                        kind="linear", fill_value="extrapolate")

    def nash_distance(u: float) -> float:
        E = float(E_interp(u))
        return abs(u - (1 / model["g"] + model["alpha"] / E - model["beta"]))

    nash = minimize_scalar(nash_distance, bounds=(settings["nash_u_min"],
                            settings["nash_u_max"]), method="bounded")
    if not nash.success:
        raise RuntimeError("Nash intersection minimizer failed.")
    u_nash, E_nash = float(nash.x), float(E_interp(nash.x))

    def fixed_payoff(E: float) -> float:
        N, status = equilibrium_population(E, u_nash, model)
        if status.startswith("solver_"):
            raise RuntimeError(status)
        return payoff(E, N, u_nash, model, economic)

    fixed_curve = np.asarray([fixed_payoff(float(E)) for E in E_values])
    E_fixed_opt, _, optimizer_error, _ = fminbound(
        lambda E: -fixed_payoff(E), settings["naive_E_min"],
        settings["E_max"], full_output=True)
    if optimizer_error:
        raise RuntimeError("Fixed-trait optimizer did not converge.")
    E_fixed_opt = float(E_fixed_opt)
    return AnalyticalResult(
        preset=config["PRESET_NAME"], config=config, E=E_values,
        species_u=species_u, species_N=species_N, evolving_payoff=evolving_payoff,
        u_grid=u_grid, leader_E=leader_E, leader_payoff=leader_payoff,
        fixed_trait_payoff=fixed_curve, E_stackelberg=float(E_values[stack_index]),
        u_stackelberg=float(species_u[stack_index]),
        payoff_stackelberg=float(evolving_payoff[stack_index]),
        E_nash=E_nash, u_nash=u_nash, nash_residual=float(nash.fun),
        E_fixed_optimum=E_fixed_opt, payoff_fixed_optimum=fixed_payoff(E_fixed_opt),
        equilibrium_status=np.asarray(statuses),
        leader_optimizer_status=np.asarray(leader_status, dtype=bool),
        nash_optimizer_status=bool(nash.success),
    )


def save_analytical(result: AnalyticalResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results_analytical.pkl").open("wb") as stream:
        pickle.dump(result, stream)
    with (output_dir / "run_parameters.json").open("w") as stream:
        json.dump(result.config, stream, indent=2,
                  default=lambda value: value.tolist() if isinstance(value, np.ndarray) else
                  value.item() if isinstance(value, np.generic) else str(value))


def load_analytical(path: Path, expected_preset: str | None = None) -> AnalyticalResult:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run main_analytical.py --stage compute.")
    with path.open("rb") as stream:
        result = pickle.load(stream)
    if not isinstance(result, AnalyticalResult):
        raise ValueError(f"Unexpected analytical result type in {path}.")
    if expected_preset and result.preset != expected_preset:
        raise ValueError(f"Expected {expected_preset}, found {result.preset} in {path}.")
    return result
