"""Combine single- or multi-species chunks produced by ``main_leaderBR.py``."""

from __future__ import annotations

import argparse
import os
from numbers import Integral
from pathlib import Path

from src.utils.preset_loader import load_and_unpack_preset
from src.utils.result_paths import result_root
from src.workflows.combine_leader_br import (
    SingleSpeciesLeaderBrCombinationError,
    combine_leader_br_chunks,
)
from src.workflows.combine_leader_br_at_species_br import (
    LeaderBrCombinationError,
    combine_leader_br_at_species_br_chunks,
)


def _preset_name() -> str:
    """Read the required preset name from the environment."""
    preset_name = os.environ.get("SEG_PRESET", "").strip()
    if not preset_name:
        raise ValueError("Environment variable 'SEG_PRESET' is not set.")
    return preset_name


def _num_species(config: dict) -> int:
    """Return the configured number of species."""
    value = config["IC_PARAMS"].get("num_pops")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("IC_PARAMS.num_pops must be an integer.")
    if int(value) < 1:
        raise ValueError("IC_PARAMS.num_pops must be positive.")
    return int(value)


def main() -> None:
    """Load the selected grid and combine all expected chunks."""
    parser = argparse.ArgumentParser(
        description="Combine single- or multi-species leader-BR chunks"
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        help="Expected count; defaults to run_parameters.json.",
    )
    args = parser.parse_args()

    try:
        config = load_and_unpack_preset(_preset_name())
        num_species = _num_species(config)
        date = config["OUTPUT_CONFIG"].get("date")
        if not isinstance(date, str) or not date.strip():
            raise ValueError("The preset must define a non-empty OUTPUT_CONFIG.date.")

        if num_species == 1:
            print(
                "This is a single-species simulation; hence, combining the "
                "leader's best responses across the trait dimension."
            )
            output_dir = (
                result_root(Path(__file__).resolve().parent) / date / "leader_br"
            )
            combine_leader_br_chunks(
                output_dir,
                expected_u_grid=config["U_GRID"],
                num_chunks=args.num_chunks,
                expected_preset=config["PRESET_NAME"],
            )
        else:
            print(
                f"This is a multi-species simulation ({num_species} species); "
                "hence, combining the leader's best responses at the species "
                "best responses to find possible Nash points."
            )
            output_dir = (
                result_root(Path(__file__).resolve().parent)
                / date
                / "leader_br_at_spbr"
            )
            combine_leader_br_at_species_br_chunks(
                output_dir=output_dir,
                num_chunks=args.num_chunks,
                expected_preset=config["PRESET_NAME"],
            )
    except (
        KeyError,
        TypeError,
        ValueError,
        SingleSpeciesLeaderBrCombinationError,
        LeaderBrCombinationError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
