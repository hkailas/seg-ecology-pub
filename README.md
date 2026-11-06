# Stackelberg evolutionary games for conservation

Code accompanying *When Policy Shapes Selection: Anticipating Evolutionary
Feedbacks in Conservation*. The simulation figures in the manuscript are
Figures 2–5.

For an explanation of the models, their combinations, and examples of running
different parameterizations, see the [model and experimentation guide](docs/README.md).

## Install

Use Conda or Micromamba with `environment_SEG.yaml`.

```bash
conda env create --file environment_SEG.yaml
conda activate SEG
```

Run commands from the repository root. Simulation datasets use Python pickle.

## Explore the single-species models

Open [the exploration notebook](notebooks/explore_single_species.ipynb) to change
parameters for either model variant, compare dynamics with and without evolution,
and inspect fitness landscapes with an interactive time slider. Resident markers
change from circles to crosses of the same colour at extinction.
The leader-analysis section calculates payoffs, best-response curves, and
candidate Stackelberg and Nash points for the selected single-species model.

```bash
jupyter lab notebooks/explore_single_species.ipynb
```

Select the `SEG` Python environment as the notebook kernel and run all cells.

## Reproduce the figures

Published datasets are included in `reproduction/datasets/`, so you can
reproduce all four figures without running simulations:

| Figure | Generated components | Command |
| --- | --- | --- |
| 2 | Three single-species dynamics comparisons | `python -m reproduction.figure2 --stage plot_precomputed` |
| 3 | Four adaptive fitness landscapes | `python -m reproduction.figure3 --stage plot_precomputed` |
| 4 | Shared species dataset and eight landscapes | `python -m reproduction.figure4 --stage plot_precomputed` |
| 5 | Six panels from analytical, BE, and shared Figure 4 data | `python -m reproduction.figure5 --stage plot_precomputed` |

For simulation commands, parameter presets, dataset paths, and HPC/SLURM
execution, see the [reproduction guide](reproduction/README.md).

## Code layout

`src/config/` holds publication presets. `src/models/` and `src/simulate/`
hold the scientific model and solver. `src/workflows/` calculates and stores
results; `src/plotting/` draws supplied results. `reproduction/` records
panel conditions and assembles manuscript figures.
