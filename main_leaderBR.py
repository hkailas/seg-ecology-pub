"""Run one chunk of leader best responses for any supported species count.

Example (PowerShell):
    $env:SEG_PRESET = "model1_1species"
    python main_leaderBR.py --chunk_id 0 --num_chunks 1 --num_cpus 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.preset_loader import load_and_unpack_preset
from src.utils.result_paths import result_root
from src.utils.publication_data import validate_shared_species
from src.workflows.combine_species_br import combine_species_br_chunk_results
from src.workflows.combine_leader_br import combine_leader_br_chunks
from src.workflows.combine_leader_br_at_species_br import (
    combine_leader_br_at_species_br_chunks,
)
from src.workflows.leader_br import (
    run_leader_br,
    validate_leader_br_arguments,
    validate_leader_br_config,
)
from src.workflows.leader_br_at_species_br import run_leader_br_at_species_br


def _preset_name() -> str:
    """Read the required preset name from the environment."""
    preset_name = os.environ.get("SEG_PRESET", "").strip()
    if not preset_name:
        raise ValueError(
            "Environment variable 'SEG_PRESET' not set. "
            "Please set it to a valid preset name."
        )
    return preset_name


def _json_default(value: Any) -> Any:
    """Convert NumPy metadata values to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_run_parameters(
    output_dir: Path,
    config: dict,
    args: argparse.Namespace,
    num_species: int,
) -> None:
    """Persist the configuration summary for this leader-BR run."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "preset_name": config["PRESET_NAME"],
        "MODEL_PARAMS": config["MODEL_PARAMS"].copy(),
        "ECONOMIC_PARAMS": config["ECONOMIC_PARAMS"].copy(),
        "SIMULATION_PARAMS": config["SIMULATION_PARAMS"].copy(),
        "IC_PARAMS": config["IC_PARAMS"].copy(),
        "OUTPUT_CONFIG": config["OUTPUT_CONFIG"].copy(),
        "run_config": {
            "workflow": (
                "leader_br" if num_species == 1 else "leader_br_at_species_br"
            ),
            "num_species": num_species,
            "num_chunks": args.num_chunks,
            "num_cpus": args.num_cpus,
        },
    }
    if num_species == 1:
        u_grid = np.asarray(config["U_GRID"], dtype=float)
        snapshot["U_GRID_info"] = {
            "length": len(u_grid),
            "u_range": (
                [float(u_grid[:, 0].min()), float(u_grid[:, 0].max())]
                if len(u_grid)
                else None
            ),
        }
    else:
        e_grid = np.asarray(config["E_GRID"], dtype=float)
        snapshot["E_GRID_info"] = {
            "length": len(e_grid),
            "min": float(e_grid.min()) if len(e_grid) else None,
            "max": float(e_grid.max()) if len(e_grid) else None,
        }

    with (output_dir / "run_parameters.json").open("w") as file:
        json.dump(snapshot, file, indent=2, default=_json_default)


def _num_species(config: dict) -> int:
    """Return the configured number of species."""
    value = config["IC_PARAMS"].get("num_pops")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("IC_PARAMS.num_pops must be an integer.")
    if int(value) < 1:
        raise ValueError("IC_PARAMS.num_pops must be positive.")
    return int(value)


def _ensure_species_br_combined(config: dict, args: argparse.Namespace) -> None:
    """Combine multi-species species-BR chunks when the master file is absent."""
    output_config = config["OUTPUT_CONFIG"]
    species_br_dir = (
        result_root(Path(__file__).resolve().parent)
        / output_config["date"]
        / output_config.get("compute_dir", "compute_data")
    )
    if args.species_results is not None:
        species_br_dir = args.species_results.resolve().parent
    combined_path = species_br_dir / "results_species_br.pkl"
    if combined_path.is_file():
        print(f"Combined species-BR results found: {combined_path}")
        return

    if args.num_chunks > 1 and args.chunk_id != 0:
        print(
            "Combined species-BR results were not found; waiting for leader-BR "
            "chunk 0 to combine them."
        )
        for elapsed_seconds in range(10, 61, 10):
            time.sleep(10)
            if combined_path.is_file():
                print(f"Combined species-BR results found: {combined_path}")
                return
            if elapsed_seconds < 60:
                print(
                    "Combined species-BR results are not available yet; "
                    f"waiting another 10 seconds ({elapsed_seconds}/60 seconds)."
                )
        raise RuntimeError(
            "Combined species-BR results were not created by leader-BR chunk 0 "
            "within 60 seconds."
        )

    print(
        "Combined species-BR results were not found; combining species-BR "
        "chunks before computing the leader's best response."
    )
    combine_species_br_chunk_results(output_dir=species_br_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Leader best responses for single- or multi-species simulations"
    )
    parser.add_argument("--chunk_id", type=int, required=True, help="0-based chunk index")
    parser.add_argument("--num_chunks", type=int, required=True, help="Total chunks")
    parser.add_argument("--num_cpus", type=int, required=True, help="Parallel workers")
    parser.add_argument("--species-results", type=Path,
                        help="Combined species-BR file to reuse for multi-species leader BR")
    return parser


def main() -> None:
    """Load configuration and dispatch one leader-BR chunk."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        preset_name = _preset_name()
        print(f"Loading preset: {preset_name!r}...")
        config = load_and_unpack_preset(preset_name)
        print(f"Preset loaded: {config['PRESET_NAME']}")
        validate_leader_br_arguments(args)
        num_species = _num_species(config)
        date = config["OUTPUT_CONFIG"].get("date")
        if not isinstance(date, str) or not date.strip():
            raise ValueError("The preset must define a non-empty OUTPUT_CONFIG.date.")

        if num_species == 1:
            print(
                "This is a single-species simulation; hence, computing the "
                "leader's best response across the trait dimension."
            )
            validate_leader_br_config(config)
            workflow_name = "leader_br"
        else:
            print(
                f"This is a multi-species simulation ({num_species} species); "
                "hence, computing the leader's best response at the species best "
                "response to find possible Nash points."
            )
            if args.species_results is not None:
                if args.species_results.name != "results_species_br.pkl":
                    raise ValueError("--species-results must name results_species_br.pkl")
                validate_shared_species(args.species_results, config)
            _ensure_species_br_combined(config, args)
            workflow_name = "leader_br_at_spbr"

        output_dir = result_root(Path(__file__).resolve().parent) / date / workflow_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.chunk_id == 0:
            _write_run_parameters(output_dir, config, args, num_species)
            print("Configuration saved")

        if num_species == 1:
            run_leader_br(args, config, output_dir)
        else:
            run_leader_br_at_species_br(args, config, output_dir,
                                        species_results_path=args.species_results)

        if args.num_chunks == 1:
            print("Combining the single leader-BR chunk...")
            if num_species == 1:
                combine_leader_br_chunks(
                    output_dir=output_dir,
                    expected_u_grid=config["U_GRID"],
                    num_chunks=1,
                    expected_preset=config["PRESET_NAME"],
                )
            else:
                combine_leader_br_at_species_br_chunks(
                    output_dir=output_dir,
                    num_chunks=1,
                    expected_preset=config["PRESET_NAME"],
                )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
