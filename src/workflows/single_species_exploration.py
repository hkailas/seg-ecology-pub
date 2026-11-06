"""Small, explicit trajectory experiments for the teaching notebook."""

from dataclasses import asdict, dataclass, replace
from time import perf_counter

import numpy as np

from src.config.parameter_presets import load_preset
from src.models.eco_evo_model import G, G_model3


SCENARIOS = {
    "model1_only_density_dependence": "model1_f5_AD_evo",
    "model1_density_and_frequency_dependence": "model1_f5_BE_evo",
    "model2_density_and_frequency_dependence": "model3_f2_C",
}


def model_display_name(model_name):
    """Notebook labels are independent of the manuscript's backend identifiers."""
    return {"model1": "Model 1", "model3": "Model 2"}[model_name]


@dataclass(frozen=True)
class ExplorationSettings:
    model_name: str
    r_max: float
    K: float
    g: float
    alpha: float
    beta: float
    d: float
    sigma: float
    beta_bully: float
    sigma_a_squared: float
    fd: bool = True
    E: float = 1.0
    initial_population: float = 63.0
    initial_trait: float = 6.4637
    evo_dyn: bool = True
    duration: float = 15000.0
    frame_count: int = 350
    trait_min: float = 0.0
    trait_max: float = 30.0
    trait_points: int = 501
    extinction_threshold: float = 1e-4


@dataclass
class ExplorationResult:
    settings: ExplorationSettings
    time: np.ndarray
    population: np.ndarray
    trait: np.ndarray
    mutant_traits: np.ndarray
    mutant_fitness: np.ndarray
    resident_fitness: np.ndarray
    extinct: np.ndarray
    elapsed_seconds: float

    def metadata(self):
        return {
            "settings": asdict(self.settings),
            "elapsed_seconds": self.elapsed_seconds,
            "actual_mutant_range": [float(self.mutant_traits[0]),
                                    float(self.mutant_traits[-1])],
            "stopping_rule": "fixed duration; equilibrium is not certified",
            "extinction_rule": "x <= extinction_threshold",
        }


@dataclass
class TwoSpeciesExplorationResult:
    """Two interacting populations and their shared instantaneous landscape."""

    settings: ExplorationSettings
    time: np.ndarray
    population: np.ndarray
    trait: np.ndarray
    mutant_traits: np.ndarray
    mutant_fitness: np.ndarray
    resident_fitness: np.ndarray
    extinct: np.ndarray
    elapsed_seconds: float


def settings_for_scenario(name="model1_density_and_frequency_dependence", **overrides):
    """Copy a publication starting point without changing any registered preset."""
    if name not in SCENARIOS:
        raise ValueError(f"Choose one of {tuple(SCENARIOS)}.")
    config = load_preset(SCENARIOS[name])
    model = config["MODEL_PARAMS"]
    values = {key: model[key] for key in (
        "r_max", "K", "g", "alpha", "beta", "d", "sigma", "fd",
    )}
    # Uniform-competition presets contain NaN placeholders. Finite values let
    # readers enable trait-dependent competition later without changing fields.
    values["beta_bully"] = model["beta_bully"] if model["fd"] else 2.0
    values["sigma_a_squared"] = model["sigma_a_squared"] if model["fd"] else 4.0
    values["model_name"] = config["SIMULATION_PARAMS"]["model_name"]
    if name == "model1_only_density_dependence":
        values["fd"] = False
    if values["model_name"] == "model3":
        values.update(trait_max=60.0)
    return replace(ExplorationSettings(**values), **overrides)


def validate_settings(settings):
    if settings.model_name not in ("model1", "model3"):
        raise ValueError("model_name must be 'model1' or 'model3'.")
    numeric = [value for value in asdict(settings).values()
               if isinstance(value, (int, float, np.number))]
    if not np.all(np.isfinite(numeric)):
        raise ValueError("All numerical settings must be finite.")
    for key in ("r_max", "K", "alpha", "beta", "sigma_a_squared",
                "duration", "extinction_threshold"):
        if getattr(settings, key) <= 0:
            raise ValueError(f"{key} must be positive.")
    for key in ("g", "d", "sigma", "E", "initial_population",
                "initial_trait", "trait_min"):
        if getattr(settings, key) < 0:
            raise ValueError(f"{key} must be nonnegative.")
    if settings.trait_max <= settings.trait_min:
        raise ValueError("The upper mutant-trait bound must exceed the lower bound.")
    for key, lower, upper in (("frame_count", 3, 2001), ("trait_points", 20, 2001)):
        value = getattr(settings, key)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{key} must be an integer.")
        if not lower <= value <= upper:
            raise ValueError(f"{key} must be between {lower} and {upper}.")


def simulate_trajectory(settings, times=None):
    """Sample the existing ODE solver without calculating mutant landscapes."""
    validate_settings(settings)
    times = (np.linspace(0.0, settings.duration, settings.frame_count)
             if times is None else np.asarray(times, dtype=float))
    if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))
            or np.any(times < 0) or np.any(times > settings.duration)):
        raise ValueError("Sample times must lie within the simulation duration.")
    # Import lazily: the notebook selects a valid SEG_PRESET before importing
    # the numerical core. Every relevant solver argument is explicit below.
    from src.simulate.simulation_core import run_simulation

    model = {key: getattr(settings, key) for key in (
        "r_max", "K", "g", "alpha", "beta", "beta_bully",
        "sigma_a_squared", "E", "d", "sigma", "fd",
    )}
    solution = run_simulation(
        {"populations": [settings.initial_population],
         "traits": [settings.initial_trait]},
        **model, include_E=True, model_name=settings.model_name,
        evo_dyn=settings.evo_dyn, min_thresh=settings.extinction_threshold,
        t_start=0.0, t_max=settings.duration, store_sol=True,
        stop_at_equilibrium=False, chunk_size=settings.duration,
        eq_tsteps=min(100.0, settings.duration), eq_thresh=1e-6, abs_thresh=1e-6,
        maxstep=min(50.0, settings.duration / 200.0), method="RK45", verbose=False,
    )
    states = np.asarray(solution.sol(times))
    if states.ndim == 1 and len(times) == 1:
        states = states[:, None]
    if states.shape != (2, len(times)) or not np.all(np.isfinite(states)):
        raise RuntimeError("The solver returned an invalid trajectory.")
    # Tiny negative values can arise in dense interpolation near extinction.
    population = np.maximum(states[0], 0.0)
    traits = states[1].copy()
    return times, population, traits


def simulate_exploration(settings):
    """Integrate once and evaluate fitness at the sampled resident states.

    Below-threshold residents contribute zero abundance to the landscapes.
    Their last traits remain available to locate the extinction crosses.
    """
    start = perf_counter()
    times, population, traits = simulate_trajectory(settings)
    extinct = population <= settings.extinction_threshold
    fitness = G if settings.model_name == "model1" else G_model3
    # Expand the requested trait window if needed so the resident stays visible.
    margin = max(0.5, 0.03 * np.ptp(traits))
    left = min(settings.trait_min, max(0.0, float(np.min(traits)) - margin))
    right = max(settings.trait_max, float(np.max(traits)) + margin)
    mutant_traits = np.linspace(left, right, settings.trait_points)
    args = tuple(getattr(settings, key) for key in (
        "r_max", "K", "g", "alpha", "beta", "beta_bully",
        "sigma_a_squared", "E", "d",
    )) + (True, settings.fd)
    curves = np.empty((len(times), len(mutant_traits)))
    resident_fitness = np.empty(len(times))
    for index, (x, u) in enumerate(zip(population, traits)):
        resident_traits = np.array([u])
        resident_population = np.array([0.0 if extinct[index] else x])
        curves[index] = [fitness(v, resident_traits, resident_population, *args)
                         for v in mutant_traits]
        resident_fitness[index] = fitness(u, resident_traits, resident_population, *args)
    if not np.all(np.isfinite(curves)) or not np.all(np.isfinite(resident_fitness)):
        raise ValueError("Fitness overflowed. Reduce the mutant-trait range or trait cost.")
    return ExplorationResult(
        settings, times, population, traits, mutant_traits, curves,
        resident_fitness, extinct, perf_counter() - start,
    )


def simulate_two_species_exploration(settings, populations, traits):
    """Simulate two interacting populations and evaluate their shared landscape.

    ``populations`` and ``traits`` are the two starting values. The first can
    therefore be initialized directly from the final state of a prior
    single-species run.
    """
    populations = np.asarray(populations, dtype=float)
    traits = np.asarray(traits, dtype=float)
    if populations.shape != (2,) or traits.shape != (2,):
        raise ValueError("Exactly two initial populations and traits are required.")
    if (not np.all(np.isfinite(populations)) or np.any(populations < 0)
            or not np.all(np.isfinite(traits)) or traits[1] < 0):
        raise ValueError(
            "Initial populations must be nonnegative; Species 2's trait must "
            "also be nonnegative."
        )
    validate_settings(settings)
    if np.any(settings.beta + traits <= 0):
        raise ValueError("Each starting trait must keep beta + trait positive.")

    from src.simulate.simulation_core import run_simulation

    start = perf_counter()
    model = {key: getattr(settings, key) for key in (
        "r_max", "K", "g", "alpha", "beta", "beta_bully",
        "sigma_a_squared", "E", "d", "sigma", "fd",
    )}
    solution = run_simulation(
        {"populations": populations.tolist(), "traits": traits.tolist()},
        **model, include_E=True, model_name=settings.model_name,
        evo_dyn=settings.evo_dyn, min_thresh=settings.extinction_threshold,
        t_start=0.0, t_max=settings.duration, store_sol=True,
        stop_at_equilibrium=False, chunk_size=settings.duration,
        eq_tsteps=min(100.0, settings.duration), eq_thresh=1e-6, abs_thresh=1e-6,
        maxstep=min(50.0, settings.duration / 200.0), method="RK45", verbose=False,
    )
    times = np.linspace(0.0, settings.duration, settings.frame_count)
    states = np.asarray(solution.sol(times))
    if states.ndim == 1 and len(times) == 1:
        states = states[:, None]
    if states.shape != (4, len(times)) or not np.all(np.isfinite(states)):
        raise RuntimeError("The solver returned an invalid two-species trajectory.")

    population = np.maximum(states[:2], 0.0)
    resident_traits = states[2:].copy()
    extinct = population <= settings.extinction_threshold
    margin = max(0.5, 0.03 * np.ptp(resident_traits))
    left = max(-settings.beta + 1e-8,
               min(settings.trait_min, float(np.min(resident_traits)) - margin))
    right = max(settings.trait_max, float(np.max(resident_traits)) + margin)
    mutant_traits = np.linspace(left, right, settings.trait_points)
    fitness = G if settings.model_name == "model1" else G_model3
    args = tuple(getattr(settings, key) for key in (
        "r_max", "K", "g", "alpha", "beta", "beta_bully",
        "sigma_a_squared", "E", "d",
    )) + (True, settings.fd)
    curves = np.empty((len(times), len(mutant_traits)))
    resident_fitness = np.empty((2, len(times)))
    for index in range(len(times)):
        community_traits = resident_traits[:, index]
        community_population = np.where(extinct[:, index], 0.0, population[:, index])
        curves[index] = [fitness(v, community_traits, community_population, *args)
                         for v in mutant_traits]
        resident_fitness[:, index] = [
            fitness(u, community_traits, community_population, *args)
            for u in community_traits
        ]
    if not np.all(np.isfinite(curves)) or not np.all(np.isfinite(resident_fitness)):
        raise ValueError("Fitness overflowed. Reduce the mutant-trait range or trait cost.")
    return TwoSpeciesExplorationResult(
        settings, times, population, resident_traits, mutant_traits, curves,
        resident_fitness, extinct, perf_counter() - start,
    )


@dataclass
class MultiPopulationResult:
    settings: ExplorationSettings
    time: np.ndarray
    population: np.ndarray
    trait: np.ndarray
    mutant_traits: np.ndarray
    mutant_fitness: np.ndarray
    resident_fitness: np.ndarray
    extinct: np.ndarray
    elapsed_seconds: float


def _initial_state(settings, populations, traits):
    populations = np.asarray(populations, dtype=float)
    traits = np.asarray(traits, dtype=float)
    if (populations.ndim != 1 or traits.ndim != 1 or len(populations) != len(traits)
            or not 1 <= len(populations) <= 4):
        raise ValueError("Supply matching population and trait vectors for 1 to 4 populations.")
    if (not np.all(np.isfinite(populations)) or np.any(populations < 0)
            or not np.all(np.isfinite(traits))):
        raise ValueError("Starting populations must be nonnegative and all starting values finite.")
    if np.any(settings.beta + traits <= 0):
        raise ValueError("Each starting trait must keep beta + trait positive.")
    validate_settings(settings)
    return populations, traits


def simulate_multi_population_trajectory(settings, populations, traits, times=None):
    """Integrate any 1–4 resident populations without evaluating landscapes."""
    populations, traits = _initial_state(settings, populations, traits)
    times = (np.linspace(0.0, settings.duration, settings.frame_count)
             if times is None else np.asarray(times, dtype=float))
    if (times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times))
            or np.any(times < 0) or np.any(times > settings.duration)):
        raise ValueError("Sample times must lie within the simulation duration.")

    from src.simulate.simulation_core import run_simulation

    model = {key: getattr(settings, key) for key in (
        "r_max", "K", "g", "alpha", "beta", "beta_bully",
        "sigma_a_squared", "E", "d", "sigma", "fd",
    )}
    solution = run_simulation(
        {"populations": populations.tolist(), "traits": traits.tolist()},
        **model, include_E=True, model_name=settings.model_name,
        evo_dyn=settings.evo_dyn, min_thresh=settings.extinction_threshold,
        t_start=0.0, t_max=settings.duration, store_sol=True,
        stop_at_equilibrium=False, chunk_size=settings.duration,
        eq_tsteps=min(100.0, settings.duration), eq_thresh=1e-6, abs_thresh=1e-6,
        maxstep=min(50.0, settings.duration / 200.0), method="RK45", verbose=False,
    )
    states = np.asarray(solution.sol(times))
    if states.ndim == 1 and len(times) == 1:
        states = states[:, None]
    count = len(populations)
    if states.shape != (2 * count, len(times)) or not np.all(np.isfinite(states)):
        raise RuntimeError("The solver returned an invalid multi-population trajectory.")
    return times, np.maximum(states[:count], 0.0), states[count:].copy()


def simulate_multi_population_exploration(settings, populations, traits):
    """Run a community simulation and evaluate its shared mutant fitness curve."""
    start = perf_counter()
    times, population, resident_traits = simulate_multi_population_trajectory(
        settings, populations, traits,
    )
    extinct = population <= settings.extinction_threshold
    margin = max(0.5, 0.03 * float(np.ptp(resident_traits)))
    left = max(-settings.beta + 1e-8,
               min(settings.trait_min, float(np.min(resident_traits)) - margin))
    right = max(settings.trait_max, float(np.max(resident_traits)) + margin)
    mutant_traits = np.linspace(left, right, settings.trait_points)
    fitness = G if settings.model_name == "model1" else G_model3
    args = tuple(getattr(settings, key) for key in (
        "r_max", "K", "g", "alpha", "beta", "beta_bully",
        "sigma_a_squared", "E", "d",
    )) + (True, settings.fd)
    curves = np.empty((len(times), len(mutant_traits)))
    resident_fitness = np.empty_like(resident_traits)
    for index in range(len(times)):
        environment_traits = resident_traits[:, index]
        environment_population = np.where(extinct[:, index], 0.0, population[:, index])
        curves[index] = [fitness(v, environment_traits, environment_population, *args)
                         for v in mutant_traits]
        resident_fitness[:, index] = [
            fitness(u, environment_traits, environment_population, *args)
            for u in environment_traits
        ]
    if not np.all(np.isfinite(curves)) or not np.all(np.isfinite(resident_fitness)):
        raise ValueError("Fitness overflowed. Reduce the mutant-trait range or trait cost.")
    return MultiPopulationResult(
        settings, times, population, resident_traits, mutant_traits, curves,
        resident_fitness, extinct, perf_counter() - start,
    )
