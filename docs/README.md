# Understanding the models and running your own experiments

This guide explains what the code calculates, how the model choices fit
together, and how to explore different parameter values. For installation,
start with the [main README](../README.md). For the exact publication commands,
datasets, and SLURM pipeline, use the [reproduction guide](../reproduction/README.md).
Run the examples below from the repository root with the `SEG` environment active.

## What the code calculates

The project couples population dynamics with the evolution of a continuous
trait. The environment affects fitness, traits affect environmental tolerance
and demographic costs, and competition can depend on the traits of other
populations. A policymaker, called the **leader** in the code, chooses the
effective resource support `E` and receives a payoff determined by that choice and the
resulting biological state.

The main calculations are:

| Calculation | Question it answers | Implementation |
| --- | --- | --- |
| Population and trait trajectories | How do abundance and traits change over time, with and without evolution? | [`single_species_dynamics.py`](../src/workflows/single_species_dynamics.py) |
| Adaptive fitness landscapes | What fitness would a rare mutant with trait `v` have in the resident environment? | [`adaptive_landscapes.py`](../src/workflows/adaptive_landscapes.py) |
| Species best responses (species BR) | At a fixed `E`, which population and trait equilibria are reached from different initial conditions? | [`species_br.py`](../src/workflows/species_br.py) |
| Leader best responses (leader BR) | With traits held fixed and population sizes allowed to equilibrate, which `E` maximizes the leader's payoff? | [`optimization.py`](../src/simulate/optimization.py) |
| Stackelberg and Nash analysis | Which policy is best when the leader anticipates evolution, and where are the mutual best responses? | [`analyze_stackelberg_and_nash.py`](../src/workflows/analyze_stackelberg_and_nash.py) |

Species BR is calculated by simulating ecological and evolutionary dynamics. Multiple initial conditions help discover alternative outcomes. The numerical
Stackelberg calculation selects the highest finite leader payoff among the
sampled species-BR equilibrium records. Nash points occur where the leader's
best response to a species equilibrium agrees with the `E` that generated it.
Grid resolution and the equilibria found therefore affect both calculations.

## The ecological and evolutionary models

For population `i`, the state consists of abundance `x_i` and resident trait
`u_i`. The solver stores all abundances first, followed by all traits:
`[x_1, ..., x_n, u_1, ..., u_n]`. The core equations are implemented in
[`eco_evo_model.py`](../src/models/eco_evo_model.py):

```text
dx_i/dt = x_i * G(u_i; resident traits, resident abundances, E)
du_i/dt = sigma * dG/dv evaluated at v = u_i
```

The second equation is a fitness-gradient model of evolution. Setting
`evo_dyn=False` holds traits fixed while abundance continues to change.
The implementation also stops trait change below the extinction threshold;
Model 2 prevents the trait from decreasing below zero.

Both model variants use the trait-dependent energetic compensation threshold
`E_c(v) = alpha / (beta + v)`. With `include_E=True`, fitness contains the
factor `E / E_c(v) - 1`. Increasing the trait lowers this threshold, but also
incurs a demographic cost. The variants differ in where that cost enters:

| Choice | Model 1 (`model1`): cost in growth rate | Model 2 (`model3`): cost in carrying capacity |
| --- | --- | --- |
| Growth rate | `r(v) = r_max * exp(-g * v)` | Constant `r_max` |
| Carrying capacity | Constant `K` | `K(v) = K * exp(-g * v)` |
| Fitness | `r(v) * (1 - A(v)/K) * (E/E_c(v) - 1) - d` | `r_max * (1 - A(v)/K(v)) * (E/E_c(v) - 1) - d` |
| Selection in configuration | `SIMULATION_PARAMS['model_name'] = 'model1'` | `SIMULATION_PARAMS['model_name'] = 'model3'` |

Here `A(v)` is the sum of resident abundances weighted by their competition
with trait `v`. The publication uses these variants to illustrate evolutionary
rescue and evolutionary suicide. A particular outcome depends on parameters,
initial conditions, and `E`; the model name alone does not guarantee it.

### Competition, evolution, and population number

These are separate choices that can be combined with either model in the ODE
solver:

| Setting | Effect |
| --- | --- |
| `MODEL_PARAMS['fd'] = False` | All competition coefficients are one; competition is independent of traits. |
| `MODEL_PARAMS['fd'] = True` | Competition depends on trait differences through the kernel below. |
| `evo_dyn=True` / `False` in `run_simulation` | Enable trait evolution / hold traits fixed. |
| `MODEL_PARAMS['include_E'] = False` | Replace the environmental multiplier by one, removing this dependence of fitness on `E`. Publication examples use `True`. |
| Number of entries in the initial population and trait lists | Number of populations in a direct simulation; the lists must have equal lengths. |
| `IC_PARAMS['num_pops']` | Number of populations generated by the best-response workflows. |

With `fd=True`, the competition kernel is

```text
a(v, u) = 1 + exp(-(v - u + beta_bully)^2 / (2 * sigma_a_squared))
            - exp(-beta_bully^2 / (2 * sigma_a_squared))
A(v) = sum_i x_i * a(v, u_i)
```

`beta_bully` shifts the kernel and introduces asymmetry; `sigma_a_squared`
controls its width. Setting `beta_bully=0` gives a symmetric trait-dependent
kernel. It does not disable trait dependence: use `fd=False` for that.
When enabling `fd`, provide finite `beta_bully` and positive
`sigma_a_squared`; some presets with `fd=False` use `NaN` placeholders.

In multi-population calculations, populations can disappear or converge to
the same trait. The combiners group equivalent traits and label branches by
the number of surviving distinct trait groups. Starting with eight populations
does not imply that eight distinct groups will survive.

## The leader's objective and the game comparison

The base leader payoff in
[`economic_model.py`](../src/models/economic_model.py) is

```text
Q_base = L * (1 - E) + (p / delta) * (1 - exp(-alpha_leader * x_total))
```

The first term favors lower `E`; the second values total abundance with a
saturating response. Additional terms can value traits and diversity:

| Parameter | Implemented addition to the payoff |
| --- | --- |
| `c1` | `c1 / u` for a surviving population in a configured single-population experiment. The numerical implementation floors the denominator at `epsilon`. |
| `c4` | `c4` times the abundance-weighted variance of surviving traits. |
| `c6` | Log-abundance richness term. Each lost population or trait group contributes `log(epsilon)` up to the configured population count. |

These weights change what the leader values. They are independent of whether
evolution occurs and whether the leader anticipates it. In particular,
`model1_f5_BE_evo` and `model1_f5_BE_eco` have the same biological experiment;
the latter sets `c1=0`. Both use evolving species responses. The `_eco` suffix
implies that the leader does not account for traits in their objective. 

A **Stackelberg** policy maximizes payoff along the species responses to `E`,
thereby anticipating how traits and abundances change. A **Nash** solution is
a mutual best response: the leader optimizes `E` with traits fixed, ecology
re-equilibrates, and those traits are also a species response to that `E`.
Payoff curves evaluated with traits fixed are consequently different from
curves that allow evolution at every `E`.

## Which presets combine these choices?

Presets are dictionaries in
[`parameter_presets.py`](../src/config/parameter_presets.py). The figure
recipes use the following combinations:

| Preset | Biological model and competition | Use |
| --- | --- | --- |
| `model1_f2_A` | Model 1, uniform competition, one population | Figure 2 trajectories |
| `model1_f2_B` | Model 1, asymmetric trait-dependent competition, one population | Figure 2 trajectories |
| `model3_f2_C` | Model 2, asymmetric trait-dependent competition, one population | Figure 2 trajectories |
| `model1_f3_adaptive_landscapes` | Model 1, asymmetric trait-dependent competition | Figure 3 resident-state and mutant-fitness calculations |
| `model3_f3_adaptive_landscapes` | Model 2, asymmetric trait-dependent competition | Figure 3 resident-state and mutant-fitness calculations |
| `model1_f4_adaptive_landscapes` | Model 1, asymmetric trait-dependent competition, eight initial populations; `c4=2.5`, `c6=0.2` | Figure 4 and Figure 5C,F |
| `model1_f5_BE_evo` / `model1_f5_BE_eco` | Model 1, asymmetric trait-dependent competition, one population; `c1=500` / `0` | Figure 5B,E |
| `model1_analytical` / `model1_analytical_c0` | Simplified single-species Model 1 calculation; `c1=500` / `0` | Figure 5A,D |

The analytical route uses an explicit species trait response with numerical
population root solves and leader optimization. It is a separate calculation
from the ODE route and does not implement the competition-kernel choices.
Additional presets are present in the source; the table identifies the ones
selected by the current publication recipes. List available presets with:

```bash
python -m src.config.parameter_presets
```

Figure conditions also live in
[`panel_specs.py`](../reproduction/panel_specs.py). In particular, landscape
recipes supply their own `E`, resident initial conditions, and mutant ranges.
Figure 2's trajectory workflow explicitly runs `E=0.6` and `E=1.0`, each with
evolution enabled and disabled. Changing the preset's `MODEL_PARAMS['E']`
does not change those fixed comparisons.

## Parameters to change

| Configuration | Parameters and purpose |
| --- | --- |
| `MODEL_PARAMS` | `r_max`: growth scale; `K`: carrying capacity or its maximum; `g`: strength of the trait cost; `alpha`, `beta`: trait-dependent energetic compensation threshold; `d`: mortality; `sigma`: evolutionary rate; `E`: effective resource support in a direct simulation; competition switches and kernel parameters as above. |
| `SIMULATION_PARAMS` | Model selection, final time `t_max`, integration `method` and `maxstep`, integration `chunk_size`, equilibrium window `eq_tsteps`, convergence tolerances `eq_thresh` and `abs_thresh`, extinction cutoff `min_thresh`, and species-workflow `timeout_seconds`. |
| `IC_PARAMS` | `num_pops`, population and trait sampling bounds, `base_seed`, species-BR samples per environmental value (`num_ICs_E`), and single-species leader-BR samples per trait (`num_ICs`). The trajectory entry point instead uses `initial_population` and `initial_trait`. |
| `ECONOMIC_PARAMS` | `L`, `p`, `delta`, and `alpha_leader` in the base payoff, plus `c1`, `c4`, and `c6`. These do not enter the population and trait ODEs. |
| `E_GRID` | Effective resource support values sampled by the species-BR workflow; these override the scalar model `E` for each run. |
| `u_grid_n` | Trait values sampled by the single-species leader-BR workflow. Loading a preset derives `U_GRID` from this array and `num_pops`. |
| `ANALYTICAL_PARAMS` | Effective resource support and trait grid bounds and sizes, optimization bounds, and plotting settings for the analytical route. |
| `OUTPUT_CONFIG['date']` | Name of the experiment's output directory; despite its name, it can be a descriptive label. |


## Register a custom preset for best-response calculations

Standalone multiprocessing commands load presets by name in fresh processes.
For those commands, register an experiment in
[`parameter_presets.py`](../src/config/parameter_presets.py). Here's an example, which is a copy of the present `model1_f5_BE_evo`, but with some parameters changed. You can insert this code
after the existing preset definitions and before `def load_preset`:

```python
PRESETS["trial_single_species"] = copy.deepcopy(PRESETS["model1_f5_BE_evo"])
trial = PRESETS["trial_single_species"]
trial["description"] = "Exploratory single-species run with a stronger trait cost"
trial["MODEL_PARAMS"]["g"] = 0.4
trial["OUTPUT_CONFIG"]["date"] = "trial_single_species"
trial["E_GRID"] = np.linspace(0.4, 1.0, 13)
trial["u_grid_n"] = np.linspace(0.0, 20.0, 60)
trial["IC_PARAMS"].update(num_ICs_E=3, num_ICs=2)
```

This example reduces the grids and number of sampled initial conditions for
exploration. It retains the source preset's convergence settings. Registering
a new name and output label also makes the saved run identifiable.

Select the preset in the shell. Use one of the following forms:

```powershell
# PowerShell
$env:SEG_PRESET = "trial_single_species"
```

```bash
# Bash
export SEG_PRESET=trial_single_species
```

Then run the stages in order (these commands work in either shell):

```bash
python main_speciesBR.py --chunk_id 0 --num_chunks 1 --num_cpus 2
python main_leaderBR.py --chunk_id 0 --num_chunks 1 --num_cpus 2
python main_analyze_stackelberg_and_nash.py --leader-br-source leader_br --payoff-curve-num-cpus 2
```

With one chunk, each best-response entry point combines its own results.
The species stage evolves traits at each value in `E_GRID`. The leader stage
holds each trait in `u_grid_n` fixed while optimizing `E` and re-solving
ecology. The analysis stage combines these responses and calculates payoff
curves. Its outputs are data files. The default output tree is:

```text
results/trial_single_species/
  compute_data/local/results_species_br.pkl
  compute_data/local/run_parameters.json
  leader_br/results_leader_br.pkl
  leader_br/run_parameters.json
  analysis/results_stack_nash.pkl
  analysis/input_provenance.json
```

Use the functions in [`src/plotting/`](../src/plotting/) to draw custom saved
results. The publication figure recipes select their own presets and input
paths; they do not switch to a new experiment merely because `SEG_PRESET` is set.

### Extending the experiment

- **Several populations:** start from `model1_f4_adaptive_landscapes`, change
  `IC_PARAMS['num_pops']` and the sampling bounds, and use a new output label.
  The leader command then optimizes at distinct species-BR equilibrium
  branches and writes to `leader_br_at_spbr/`. Analyze with
  `--leader-br-source leader_br_at_spbr`. The current branch processing is
  designed around up to four surviving distinct trait groups. Larger surviving
  sets require changing the processing code -- NOTE: If `beta_bully` is < 2 for multi-species simulations, more than 4 species will be able to coexist but the current code will not handle that.
- **A different leader objective:** change `ECONOMIC_PARAMS` independently of
  the biological parameters. The Figure 5 BE presets show how a species dataset
  can be shared when only the objective changes. The reproduction guide
  documents the explicit shared-input commands and metadata checks.
- **The analytical case:** copy `model1_analytical`, change its model,
  economic, or `ANALYTICAL_PARAMS` values, and give it a new output label. Run
  `python main_analytical.py --preset <your_name> --stage compute` to save data
  and draw the analytical comparisons.
- **Model 2:** the ODE solver and direct landscape calculations support it.
  The single-species grid-based leader-BR workflow explicitly requires Model 1;
  changing `model_name` alone does not extend that workflow to Model 2.

## Interpreting and refining an exploratory run

Small grids are useful for learning the workflow, but can miss alternative
equilibria, narrow branches, or Nash intersections. If no Nash point is
reported, inspect the response curves and residuals before concluding that
none exists. Refine `E_GRID` and the leader trait grid, and increase initial
condition sampling for the region of interest. The analysis command also
offers `--interpolate-nash`; interpolated candidates still need numerical
assessment on a finer grid.

Check equilibrium and optimization status before interpreting endpoints as
best responses. `species_br` always enables evolution, and the fixed-trait
leader calculation always disables it; changing the preset's `evo_dyn` field
does not reverse those workflow definitions. For a direct simulation, choose
that flag explicitly as in the example above. Keep `chunk_size >= eq_tsteps`
when changing integration settings.

Record the full parameters, seed, and chunk partition for comparisons. The
best-response seed hierarchy depends on chunk IDs, so changing the number of
chunks changes sampled initial conditions even with the same base seed.
Choose a new output label for each experiment: rerunning into an existing
directory can replace saved files. `SEG_RESULTS_ROOT` selects another root
for ordinary entry-point outputs. For large grids, chunk combination, and
cluster execution, continue with the [reproduction guide](../reproduction/README.md).
