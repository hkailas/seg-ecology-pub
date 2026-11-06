"""Inventory and checksum the manuscript's precomputed result bundle."""

import argparse

from reproduction.common import DATA_ROOT, MANIFEST, PRECOMPUTED_MANIFEST, dataset_path, require, write_manifest
from reproduction.figure4 import (ANALYSIS as FIG4_ANALYSIS, LANDSCAPES,
                                  LEADER as FIG4_LEADER, SPECIES as FIG4_SPECIES)
from reproduction.figure5 import (ANALYTICAL, ANALYTICAL_C0, ECO_ANALYSIS,
                                  ECO_LEADER, EVO_ANALYSIS, EVO_LEADER, EVO_SPECIES)

DATASETS = {
    "figure4_species": FIG4_SPECIES,
    "figure4_leader": FIG4_LEADER,
    "figure4_analysis": FIG4_ANALYSIS,
    "figure4_landscapes": LANDSCAPES,
    "BE_species": EVO_SPECIES,
    "BE_evo_leader": EVO_LEADER,
    "BE_eco_leader": ECO_LEADER,
    "BE_evo_analysis": EVO_ANALYSIS,
    "BE_eco_analysis": ECO_ANALYSIS,
    "analytical": ANALYTICAL,
    "analytical_c0": ANALYTICAL_C0,
}
PANELS = {
    "figure4_species": ["4A", "5C", "5F"],
    "figure4_leader": ["5C", "5F"],
    "figure4_analysis": ["4A", "5C", "5F"],
    "figure4_landscapes": ["4B", "4C", "4D", "4E", "4F", "4G", "4H", "4I"],
    "BE_species": ["5B", "5E"],
    "BE_evo_leader": ["5B", "5E"],
    "BE_eco_leader": ["5E"],
    "BE_evo_analysis": ["5B", "5E"],
    "BE_eco_analysis": ["5E"],
    "analytical": ["5A", "5D"],
    "analytical_c0": ["5D"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Inventory only currently available files.")
    parser.add_argument("--precomputed", action="store_true",
                        help="Inventory reproduction/datasets, including Figures 2 and 3, instead of computed results.")
    args = parser.parse_args(argv)
    stage = "plot_precomputed" if args.precomputed else "plot"
    sources = {key: dataset_path(path, stage) for key, path in DATASETS.items()}
    panels = dict(PANELS)
    manifest = PRECOMPUTED_MANIFEST if args.precomputed else MANIFEST
    if args.precomputed:
        for label in "ABC":
            sources[f"figure2_{label}"] = dataset_path(
                DATA_ROOT / f"Fig2-{label}" / "dynamics" / "results_dynamics.pkl", stage)
            panels[f"figure2_{label}"] = [f"2{label}"]
        sources["figure3_landscapes"] = dataset_path(
            DATA_ROOT / "Fig3" / "landscapes" / "results_landscapes.pkl", stage)
        panels["figure3_landscapes"] = ["3"]
    try:
        if not args.allow_partial:
            for key, path in sources.items():
                require(path, f"produce publication dataset {key}", stage=stage)
        datasets = {key: path for key, path in sources.items() if path.is_file()}
        write_manifest(manifest, datasets, panels=panels)
    except FileNotFoundError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
