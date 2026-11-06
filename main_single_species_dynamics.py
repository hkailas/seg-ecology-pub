"""Compute trajectories for four single-species energy/evolution conditions.

Example:
    python main_single_species_dynamics.py --preset model1_f2_A --save-data
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from src.utils.result_paths import result_root


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare single-species population and trait dynamics at E=0.6 "
            "and E=1, with evolution enabled and disabled."
        )
    )
    parser.add_argument(
        "--preset",
        required=True,
        help=(
            "Parameter preset containing IC_PARAMS.initial_population and "
            "IC_PARAMS.initial_trait."
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        help=(
            "Absolute final simulation time. When omitted, each run stops at "
            "equilibrium or at the preset t_max."
        ),
    )
    parser.add_argument("--save-data", action="store_true",
                        help="Persist trajectories and run parameters for manuscript panel assembly.")
    parser.add_argument("--data-dir", type=Path,
                        help="Directory for persisted trajectories (default: results/<preset date>/dynamics)")
    return parser


def main() -> None:
    """Load the selected preset and run the dynamics comparison."""
    parser = build_parser()
    args = parser.parse_args()

    # Preset-dependent simulation modules read SEG_PRESET while importing. Set it
    # before those imports so the explicit CLI choice always takes precedence.
    os.environ["SEG_PRESET"] = args.preset

    from src.utils.preset_loader import load_and_unpack_preset

    try:
        config = load_and_unpack_preset(args.preset)
        output_config = config["OUTPUT_CONFIG"]
        date = output_config.get("date")
        if not isinstance(date, str) or not date.strip():
            raise ValueError("The preset must define a non-empty OUTPUT_CONFIG.date.")

        from src.workflows.single_species_dynamics import run_single_species_dynamics

        trajectories = run_single_species_dynamics(config, time_limit=args.time_limit)
        if args.save_data:
            project_root = Path(__file__).resolve().parent
            data_dir = args.data_dir or result_root(project_root) / date / "dynamics"
            data_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                **config,
                "entrypoint": "main_single_species_dynamics.py",
                "time_limit": args.time_limit,
                "conditions": [
                    {
                        **asdict(trajectory.condition),
                        "num_plot_points": len(trajectory.time),
                        "simulation_overrides": {
                            "store_sol": True,
                            "evo_dyn": trajectory.condition.evo_dyn,
                            "stop_at_equilibrium": args.time_limit is None,
                            **({"t_max": args.time_limit} if args.time_limit is not None else {}),
                        },
                    }
                    for trajectory in trajectories.values()
                ],
            }
            metadata_json = json.dumps(
                metadata, indent=2,
                default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value),
            )
            with (data_dir / "results_dynamics.pkl").open("wb") as stream:
                pickle.dump(trajectories, stream)
            (data_dir / "run_parameters.json").write_text(metadata_json, encoding="utf-8")
            print(f"Saved: {data_dir / 'results_dynamics.pkl'}")
            print(f"Saved: {data_dir / 'run_parameters.json'}")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    for condition in trajectories.values():
        status = "equilibrium reached" if condition.equilibrium_reached else "ended"
        print(
            f"{condition.condition.key}: {status} at "
            f"t={condition.time[-1]:g}"
        )


if __name__ == "__main__":
    main()
