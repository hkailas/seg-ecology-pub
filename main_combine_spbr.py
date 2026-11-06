"""Combine chunks produced by ``main_speciesBR.py``.

Example:
    $env:SEG_PRESET='model_name_here'
    python main_combine_spbr.py
"""

import argparse
import os
from pathlib import Path

from src.utils.preset_loader import load_and_unpack_preset
from src.utils.result_paths import result_root
from src.workflows.combine_species_br import (
    SpeciesBrCombinationError,
    combine_species_br_chunk_results,
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
    """Load the run configuration and combine species-BR chunks."""
    parser = argparse.ArgumentParser(
        description="Combine species best-response chunk result files."
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        help="Expected chunk count; defaults to run_parameters.json.",
    )
    parser.add_argument(
        "--missing-chunks",
        choices=("stop", "combine-available"),
        default="stop",
        help="Policy when one or more expected chunk files are missing.",
    )
    parser.add_argument(
        "--incomplete-ics",
        choices=("stop", "common-count", "zero-fill"),
        default="stop",
        help="Policy when an included non-skipped E has too few ICs.",
    )
    args = parser.parse_args()

    preset_name = _preset_name()
    config = load_and_unpack_preset(preset_name)
    output_config = config["OUTPUT_CONFIG"]
    date = output_config["date"]
    compute_dir = output_config.get("compute_dir", "compute_data")
    output_dir = result_root(Path(__file__).resolve().parent) / date / compute_dir

    try:
        combine_species_br_chunk_results(
            output_dir=output_dir,
            num_chunks=args.num_chunks,
            missing_chunks=args.missing_chunks,
            incomplete_ics=args.incomplete_ics,
        )
    except SpeciesBrCombinationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
