"""Entry point for species-BR payoff and Nash analysis.

Example:
    $env:SEG_PRESET='model1_test'
    python main_analyze_stackelberg_and_nash.py

For fixed-trait single-species leader-BR results:
    python main_analyze_stackelberg_and_nash.py --leader-br-source leader_br
"""

import argparse
import json
import os
from pathlib import Path

from src.utils.preset_loader import load_and_unpack_preset
from src.utils.result_paths import result_root
from src.utils.publication_data import sha256_file, validate_shared_species
from src.workflows.analyze_stackelberg_and_nash import (
    StackNashAnalysisError,
    run_stack_nash_analysis,
)


def _preset_name() -> str:
    """Read the required preset name from the environment."""
    preset_name = os.environ.get("SEG_PRESET", "").strip()
    if not preset_name:
        raise ValueError(
            "Environment variable 'SEG_PRESET' not set. "
            "Please set it to a valid preset name."
        )
    return preset_name


def main() -> None:
    """Load configured result paths and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze species-BR payoffs and Nash equilibria."
    )
    parser.add_argument(
        "--interpolate-nash",
        action="store_true",
        help="Interpolate Nash points instead of snapping to E-grid points.",
    )
    parser.add_argument(
        "--nash-residual-threshold",
        type=float,
        default=1e-3,
        help="Maximum absolute residual for a snapped Nash point.",
    )
    parser.add_argument(
        "--payoff-curve-num-cpus",
        type=int,
        default=1,
        help="Worker processes for fixed-trait Nash payoff curves.",
    )
    parser.add_argument(
        "--leader-br-source",
        choices=("auto", "leader_br", "leader_br_at_spbr"),
        default="auto",
        help=(
            "Leader-BR input source. 'auto' prefers results/<date>/leader_br/ "
            "when its combined result exists."
        ),
    )
    parser.add_argument("--species-results", type=Path,
                        help="Complete species-BR result file, including shared objective inputs")
    parser.add_argument("--leader-results", type=Path,
                        help="Combined leader-BR result file")
    parser.add_argument("--output-dir", type=Path,
                        help="Analysis output directory")
    args = parser.parse_args()

    preset_name = _preset_name()
    config = load_and_unpack_preset(preset_name)
    output_config = config["OUTPUT_CONFIG"]
    date = output_config["date"]
    compute_dir = output_config.get("compute_dir", "compute_data")
    project_root = Path(__file__).resolve().parent
    data_root = result_root(project_root)

    species_results_path = (
        data_root
        / date
        / compute_dir
        / "results_species_br.pkl"
    )
    if args.species_results is not None:
        species_results_path = args.species_results.resolve()
    grid_leader_results_path = (
        data_root
        / date
        / "leader_br"
        / "results_leader_br.pkl"
    )
    species_equilibrium_leader_results_path = (
        data_root
        / date
        / "leader_br_at_spbr"
        / "results_leader_br_at_spbr.pkl"
    )
    if args.leader_br_source == "leader_br":
        leader_results_path = grid_leader_results_path
    elif args.leader_br_source == "leader_br_at_spbr":
        leader_results_path = species_equilibrium_leader_results_path
    elif grid_leader_results_path.is_file():
        leader_results_path = grid_leader_results_path
    else:
        leader_results_path = species_equilibrium_leader_results_path
    analysis_output_dir = data_root / date / "analysis"
    if args.leader_results is not None:
        leader_results_path = args.leader_results.resolve()
    if args.output_dir is not None:
        analysis_output_dir = args.output_dir.resolve()
    print(f"Using leader-BR results: {leader_results_path}")

    try:
        provenance = validate_shared_species(species_results_path, config)
        run_stack_nash_analysis(
            config=config,
            species_results_path=species_results_path,
            leader_results_path=leader_results_path,
            output_dir=analysis_output_dir,
            interpolate_nash=args.interpolate_nash,
            nash_residual_threshold=args.nash_residual_threshold,
            payoff_curve_num_cpus=args.payoff_curve_num_cpus,
        )
        provenance["leader_preset"] = preset_name
        provenance["leader_sha256"] = sha256_file(leader_results_path)
        with (analysis_output_dir / "input_provenance.json").open("w") as stream:
            json.dump(provenance, stream, indent=2)
    except (StackNashAnalysisError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
