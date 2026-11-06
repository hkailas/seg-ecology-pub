"""Compute or plot persisted partly analytical Model 1 results."""

import argparse
from pathlib import Path

from reproduction.common import PRECOMPUTED_ROOT, FIGURE_ROOT, require, verify_stage_manifest

from src.plotting.analytical import (
    plot_best_responses, plot_payoffs
)
from src.plotting.manuscript_figures import save_figure
from src.utils.preset_loader import load_and_unpack_preset
from src.utils.result_paths import result_root
from src.workflows.analytical import compute_analytical, load_analytical, save_analytical


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--stage", choices=("compute", "plot", "plot_precomputed"), default="compute")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path,
                        help="Directory for standalone analytical plots")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=("png", "pdf", "svg"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "plot_precomputed" and args.output_dir:
        parser.error("--output-dir cannot override reproduction/datasets with --stage plot_precomputed; use --stage plot for custom data")
    if args.no_save and not args.show and args.stage != "compute":
        parser.error("--no-save requires --show for plot-only use")
    config = load_and_unpack_preset(args.preset)
    root = Path(__file__).resolve().parent
    result_dir = args.output_dir or result_root(root) / config["OUTPUT_CONFIG"]["date"]
    result_path = result_dir / "results_analytical.pkl"
    if args.stage == "plot_precomputed":
        result_path = PRECOMPUTED_ROOT / config["OUTPUT_CONFIG"]["date"] / "results_analytical.pkl"
    try:
        if args.stage == "compute":
            result = compute_analytical(config)
            save_analytical(result, result_dir)
        else:
            require(result_path, f"main_analytical.py --preset {args.preset} --stage compute", stage=args.stage)
            if args.stage == "plot_precomputed":
                verify_stage_manifest(args.stage, {config["OUTPUT_CONFIG"]["date"]: result_path})
            result = load_analytical(result_path, args.preset)
        default_figures = FIGURE_ROOT if args.stage == "plot_precomputed" else root / "figures"
        figure_dir = args.figure_dir or default_figures / "figure5" / args.preset
        for stem, function in (
            ("best_response_curves", plot_best_responses),
            ("leader_payoff_comparison", plot_payoffs),
        ):
            figure, _ = function(result)
            save_figure(figure, figure_dir, stem, args.formats,
                        show=args.show, save=not args.no_save)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
