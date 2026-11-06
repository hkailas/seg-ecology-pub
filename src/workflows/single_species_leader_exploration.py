"""Single-resident leader payoffs and candidate game solutions for the notebook."""

from dataclasses import asdict, dataclass, replace
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from src.models.eco_evo_model import G, G_model3, dGdv, dGdv_model3
from src.workflows.single_species_exploration import simulate_trajectory, validate_settings


@dataclass(frozen=True)
class LeaderParameters:
    L: float = 100.0
    p: float = 10.0
    delta: float = 0.18
    alpha_leader: float = 0.01
    c1: float = 500.0


@dataclass(frozen=True)
class LeaderScanSettings:
    E_min: float = 0.0
    E_max: float = 1.0
    E_points: int = 200
    trait_min: float = 0.0
    trait_max: float = 40.0
    trait_points: int = 200
    convergence_tolerance: float = 1e-4
    nash_tolerance: float = 1e-4


@dataclass
class GameCandidate:
    E: float
    x: float
    u: float
    payoff: float
    stationary: bool
    residual: float | None = None
    sampled_x: float | None = None


@dataclass
class NashPayoffCurve:
    point: GameCandidate
    E: np.ndarray
    population: np.ndarray
    payoff: np.ndarray


@dataclass
class LeaderAnalysis:
    biology: object
    economics: LeaderParameters
    scan: LeaderScanSettings
    E: np.ndarray
    population: np.ndarray
    sampled_population: np.ndarray
    trait: np.ndarray
    payoff: np.ndarray
    stationary: np.ndarray
    residual: np.ndarray
    leader_traits: np.ndarray
    leader_E: np.ndarray
    nash_payoff_curves: list
    stackelberg: GameCandidate
    nash_points: list
    failures: dict
    elapsed_seconds: float

    def metadata(self):
        return {
            "biology": asdict(self.biology), "economics": asdict(self.economics),
            "scan": asdict(self.scan), "stackelberg_candidate": asdict(self.stackelberg),
            "nash_candidates": [asdict(point) for point in self.nash_points],
            "fixed_trait_comparison": "One curve per detected Nash trait; omitted when no Nash point is found.",
            "payoff_convention": "Shared objective; ecology equilibrated at each simulated resident trait.",
            "failures": self.failures, "elapsed_seconds": self.elapsed_seconds,
            "interpretation": "Candidates use one initial condition and a finite response duration.",
        }


def validate_leader_settings(biology, economics, scan):
    validate_settings(biology)
    if not all(np.isfinite(value) for value in asdict(economics).values()):
        raise ValueError("All payoff parameters must be finite.")
    if economics.delta <= 0 or min(economics.L, economics.p, economics.alpha_leader) < 0:
        raise ValueError("delta must be positive; L, p, and alpha_leader must be nonnegative.")
    if not all(np.isfinite(value) for value in asdict(scan).values()):
        raise ValueError("All scan settings must be finite.")
    if not 0 <= scan.E_min < scan.E_max:
        raise ValueError("Choose nonnegative, increasing resource-support bounds.")
    if not 0 <= scan.trait_min < scan.trait_max:
        raise ValueError("Choose nonnegative trait values and increasing trait bounds.")
    for key in ("E_points", "trait_points"):
        value = getattr(scan, key)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 3:
            raise ValueError(f"{key} must be an integer of at least 3; received {value!r}.")
    if min(scan.convergence_tolerance, scan.nash_tolerance) <= 0:
        raise ValueError("Convergence and Nash tolerances must be positive.")


def state_payoff(E, x, u, economics):
    """Use the repository's payoff implementation, including its survival cutoff."""
    from src.models.economic_model import Q_leader

    energy, population, traits = np.broadcast_arrays(
        np.asarray(E, dtype=float), np.asarray(x, dtype=float), np.asarray(u, dtype=float),
    )
    return np.asarray(Q_leader(
        energy, economics.L, economics.p, economics.delta, economics.alpha_leader,
        population, N_star=population[..., None], u_star=traits[..., None],
        c1=economics.c1, configured_num_pops=1,
    ))


def fixed_trait_population(E, trait, biology):
    """Stable single-resident ecological equilibrium with traits held fixed.

    Self-competition is exactly one for both kernel choices. Thus each variant
    has x*=capacity*(1-d/effective_growth) on the viable positive branch.
    This avoids nested time integrations without changing the model equations.
    """
    if not np.isfinite(trait) or trait < 0:
        raise ValueError("Fixed traits must be finite and nonnegative.")
    energy = np.asarray(E, dtype=float)
    if biology.initial_population <= biology.extinction_threshold:
        return np.zeros_like(energy)
    rate = biology.r_max * np.exp(-biology.g * trait) if biology.model_name == "model1" else biology.r_max
    capacity = biology.K if biology.model_name == "model1" else biology.K * np.exp(-biology.g * trait)
    effective_growth = rate * (energy * (biology.beta + trait) / biology.alpha - 1.0)
    viable = effective_growth > biology.d
    ratio = np.divide(biology.d, effective_growth, out=np.ones_like(energy), where=viable)
    population = np.where(viable, capacity * (1.0 - ratio), 0.0)
    return np.where(population > biology.extinction_threshold, population, 0.0)


def leader_best_response(trait, biology, economics, bounds=(0.0, 1.0)):
    """Maximize over the extinct and viable branches, including both endpoints.

    On the viable branch the abundance benefit is concave in E, while c1/u
    is constant. Search that branch separately because the payoff can jump at
    the survival cutoff. A bounded search over the whole interval can miss it.
    """
    lower, upper = bounds
    if not np.isfinite(trait) or trait < 0:
        raise ValueError("Leader responses require a finite, nonnegative trait.")

    def payoff(energy):
        return state_payoff(energy, fixed_trait_population(energy, trait, biology), trait, economics)

    candidates = [float(lower), float(upper)]
    rate = biology.r_max * np.exp(-biology.g * trait) if biology.model_name == "model1" else biology.r_max
    capacity = biology.K if biology.model_name == "model1" else biology.K * np.exp(-biology.g * trait)
    cutoff = max(1e-4, biology.extinction_threshold)  # Q_leader's own survival cutoff.
    if rate > 0 and capacity > cutoff and biology.initial_population > biology.extinction_threshold:
        critical = (biology.alpha / (biology.beta + trait)
                    * (1.0 + biology.d / (rate * (1.0 - cutoff / capacity))))
        start = max(lower, critical + max(1e-10, (upper - lower) * 1e-9))
        if start < upper:
            candidates.append(float(start))
            optimum = minimize_scalar(lambda E: -float(payoff(E)), bounds=(start, upper),
                                      method="bounded", options={"xatol": 1e-8})
            if optimum.success:
                candidates.append(float(optimum.x))
    values = np.asarray([float(payoff(E)) for E in candidates])
    if not np.all(np.isfinite(values)):
        raise ValueError("The leader payoff is not finite for these settings.")
    index = int(np.argmax(values))
    return candidates[index], float(values[index])


def _stationary(settings, E, x, u, population_window, trait_window, tolerance):
    if x <= settings.extinction_threshold:
        return True
    fitness = G if settings.model_name == "model1" else G_model3
    gradient = dGdv if settings.model_name == "model1" else dGdv_model3
    residents, populations = np.array([u]), np.array([x])
    common = (settings.r_max, settings.K, settings.g, settings.alpha, settings.beta,
              settings.beta_bully, settings.sigma_a_squared, E)
    per_capita = fitness(u, residents, populations, *common, settings.d, True, settings.fd)
    if settings.model_name == "model1":
        selection = gradient(u, residents, populations, *common, True, 0, settings.fd)
    else:
        selection = gradient(u, residents, populations, *common, True, settings.fd)
        if u <= 1e-8 and selection < 0:
            selection = 0.0
    window_change = max(np.ptp(population_window) / max(1.0, x), np.ptp(trait_window))
    return bool(abs(per_capita) <= tolerance and abs(selection) <= tolerance
                and window_change <= tolerance)


def calculate_leader_analysis(biology, economics=None, scan=None, progress=None):
    """Sweep evolving responses, optimize fixed-trait policies, and refine roots.

    A candidate is flagged stationary only if its final state and final 10%
    of the simulated trajectory satisfy the stated convergence tolerance.
    Nash brackets are never joined across known extinct or failed responses.
    """
    economics = economics or LeaderParameters()
    scan = scan or LeaderScanSettings()
    validate_leader_settings(biology, economics, scan)
    start = perf_counter()
    bounds = (scan.E_min, scan.E_max)
    energies = np.linspace(*bounds, scan.E_points)
    samples = np.linspace(0.9 * biology.duration, biology.duration, 5)
    responses, failures = {}, {}

    def response(E):
        key = float(E)
        if key not in responses:
            settings = replace(biology, E=key, evo_dyn=True)
            _, population, traits = simulate_trajectory(settings, times=samples)
            sampled_x, u = float(population[-1]), float(traits[-1])
            if u < 0:
                raise ValueError("The response reached a negative trait; this leader scan uses nonnegative traits.")
            stationary = _stationary(settings, key, sampled_x, u, population, traits, scan.convergence_tolerance)
            # Compare both payoff curves at the same ecological equilibrium.
            # A finite-time trait response remains flagged nonstationary above.
            x = (float(fixed_trait_population(key, u, settings))
                 if sampled_x > biology.extinction_threshold else 0.0)
            payoff = float(state_payoff(key, x, u, economics))
            residual = None
            if x > biology.extinction_threshold:
                best_E, _ = leader_best_response(u, biology, economics, bounds)
                residual = best_E - key
            responses[key] = GameCandidate(key, x, u, payoff, stationary, residual, sampled_x)
        return responses[key]

    points = []
    for index, E in enumerate(energies):
        try:
            points.append(response(E))
        except (ValueError, RuntimeError, ArithmeticError) as error:
            failures[f"{E:.12g}"] = str(error)
            points.append(None)
        if progress is not None:
            progress(index + 1, len(energies))
    valid = [point for point in points if point is not None and np.isfinite(point.payoff)]
    if not valid:
        raise RuntimeError("No resource-support value produced a valid response. Check duration and parameters.")
    # A coarse cell may start extinct and end alive, with a genuine living
    # intersection inside it. Sample the interior first; never infer a root by
    # assigning a residual to the extinct endpoint itself.
    sampled_supports = set(float(E) for E in energies)
    for left, right in zip(points[:-1], points[1:]):
        if left is None or right is None or (left.residual is None) == (right.residual is None):
            continue
        for E in np.linspace(left.E, right.E, 5)[1:-1]:
            sampled_supports.add(float(E))
            try:
                response(E)
            except (ValueError, RuntimeError, ArithmeticError) as error:
                failures[f"{E:.12g}"] = str(error)
    points = [responses.get(E) for E in sorted(sampled_supports)]
    valid = [point for point in points if point is not None and np.isfinite(point.payoff)]
    nash = []

    def keep_nash(point):
        if (point.residual is not None and abs(point.residual) <= scan.nash_tolerance
                and all(abs(point.E - previous.E) > 1e-4 for previous in nash)):
            nash.append(point)

    for point in valid:
        keep_nash(point)
    for left, right in zip(points[:-1], points[1:]):
        if left is None or right is None or left.residual is None or right.residual is None:
            continue
        if left.residual * right.residual >= 0:
            continue

        def residual(E):
            point = response(E)
            if point.residual is None:
                raise ValueError("The response went extinct inside the bracket.")
            return point.residual

        try:
            root = brentq(residual, left.E, right.E, xtol=1e-6, maxiter=35)
            keep_nash(response(root))
        except (ValueError, RuntimeError, ArithmeticError) as error:
            failures[f"Nash bracket {left.E:.6g}–{right.E:.6g}"] = str(error)

    nash = sorted(nash, key=lambda point: point.E)
    # Include the refined Nash support values, rather than expecting a line
    # interpolated between coarse grid values to pass through their payoffs.
    energies = np.unique(np.r_[sorted(sampled_supports), [point.E for point in nash]])
    points = [responses.get(float(E)) for E in energies]
    valid = [point for point in points if point is not None]
    stackelberg = max(valid, key=lambda point: point.payoff)

    def values(key):
        return np.asarray([getattr(point, key) if point is not None else np.nan for point in points])

    traits = np.linspace(scan.trait_min, max(scan.trait_max, max(point.u for point in valid)),
                         scan.trait_points)
    leader_E = np.array([leader_best_response(u, biology, economics, bounds)[0] for u in traits])
    nash_curves = []
    for point in nash:
        population = fixed_trait_population(energies, point.u, biology)
        nash_curves.append(NashPayoffCurve(
            point, energies.copy(), population,
            state_payoff(energies, population, point.u, economics),
        ))
    return LeaderAnalysis(
        biology, economics, scan, energies, values("x"), values("sampled_x"), values("u"), values("payoff"),
        np.array([point.stationary if point else False for point in points]),
        np.array([point.residual if point and point.residual is not None else np.nan for point in points]),
        traits, leader_E, nash_curves,
        stackelberg, nash, failures, perf_counter() - start,
    )
