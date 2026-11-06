"""Compute or plot adaptive fitness landscapes for Figure 3 or Figure 4."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SEG_PRESET", "model1_f4_adaptive_landscapes")

from reproduction.panel_specs import FIGURE3, FIGURE4, FIGURE4_PRESET
from src.plotting.adaptive_landscapes import plot_adaptive_landscape
from src.plotting.manuscript_figures import save_figure
from src.utils.preset_loader import load_and_unpack_preset
from src.utils.result_paths import result_root
from src.workflows.adaptive_landscapes import (
    evaluate_adaptive_landscape, load_landscapes, save_landscapes,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=(3, 4), required=True)
    parser.add_argument("--preset", help="Optional assertion for the figure's preset")
    parser.add_argument("--stage", choices=("compute", "plot"), default="compute")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path,
                        help="Directory for standalone adaptive landscapes")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=("png", "pdf", "svg"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    specs = FIGURE3 if args.figure == 3 else FIGURE4
    if args.preset and args.preset not in {spec.preset for spec in specs}:
        parser.error(f"Preset {args.preset} is not used for Figure {args.figure}.")
    if args.figure == 4 and args.preset and args.preset != FIGURE4_PRESET:
        parser.error("Figure 4 has one adaptive-landscape preset.")
    root = Path(__file__).resolve().parent
    output_dir = args.output_dir or (result_root(root) / "Fig4" / "landscapes"
                                     if args.figure == 4 else result_root(root) / "Fig3" / "landscapes")
    path = output_dir / "results_landscapes.pkl"
    try:
        if args.stage == "compute":
            results = {}
            metadata = {}
            for spec in specs:
                config = load_and_unpack_preset(spec.preset)
                metadata[spec.key] = {
                    "preset": spec.preset, "MODEL_PARAMS": config["MODEL_PARAMS"],
                    "SIMULATION_PARAMS": config["SIMULATION_PARAMS"],
                    "IC_PARAMS": config["IC_PARAMS"],
                    "E": spec.E, "initial_populations": spec.populations,
                    "initial_traits": spec.traits, "threshold": spec.threshold,
                    "mutant_range": spec.mutant_range, "points": spec.points,
                }
                results[spec.key] = evaluate_adaptive_landscape(
                    config, spec.key, spec.populations, spec.traits, spec.E,
                    x_limits=spec.x_limits, y_limits=spec.y_limits,
                    threshold=spec.threshold, mutant_range=spec.mutant_range,
                    points=spec.points)
            save_landscapes(results, path)
            with (output_dir / "run_parameters.json").open("w") as stream:
                json.dump(metadata, stream, indent=2,
                          default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))
        else:
            results = load_landscapes(path, {spec.key for spec in specs})
            for spec in specs:
                if results[spec.key].preset != spec.preset or results[spec.key].energy != spec.E:
                    raise ValueError(f"Saved landscape {spec.key} does not match panel specifications.")
        figure_dir = args.figure_dir or root / "figures" / f"figure{args.figure}"
        for spec in specs:
            figure, _ = plot_adaptive_landscape(results[spec.key])
            save_figure(figure, figure_dir, spec.key, args.formats,
                        show=args.show, save=not args.no_save)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
