"""Entry point for running the species best-response workflow.

When running on Windows:
    $env:SEG_PRESET='model_name_here'
    python main_speciesBR.py --chunk_id 0 --num_chunks 1 --num_cpus 10
"""

import argparse
import json
import os
import time
from datetime import datetime

from src.utils.preset_loader import load_and_unpack_preset
from src.utils.result_paths import result_root
from pathlib import Path
from src.workflows.combine_species_br import combine_species_br_chunk_results
from src.workflows.species_br import run_species_br


if os.environ.get('SEG_PRESET') is not None:
    preset_name = os.environ.get('SEG_PRESET', '').strip()
else:
    raise ValueError("Environment variable 'SEG_PRESET' not set. Please set it to a valid preset name.")


def main():
    """Load configuration and run one species best-response chunk."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk_id",
        type=int,
        required=True,
        help="Chunk index from 0 to NUM_CHUNKS-1",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        required=True,
        help="Total number of chunks",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        required=True,
        help="Number of CPUs to use",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=10,
        help="Healing iterations",
    )
    args = parser.parse_args()

    print(f"Loading preset: '{preset_name}'...")
    try:
        config = load_and_unpack_preset(preset_name)
    except ValueError as e:
        print(f"\nError: {e}")
        return

    print(f"Preset loaded: {config['PRESET_NAME']}")

    output_config = config['OUTPUT_CONFIG']
    date = output_config['date']
    compute_dir = output_config.get('compute_dir', 'compute_data')
    output_dir = os.path.join(
        result_root(Path(__file__).resolve().parent),
        date,
        compute_dir,
    )

    if not os.path.exists(output_dir) and args.chunk_id == 0:
        os.makedirs(output_dir)
    else:
        time.sleep(2)

    if args.chunk_id == 0:
        model_params = config['MODEL_PARAMS']
        economic_params = config['ECONOMIC_PARAMS']
        simulation_params = config['SIMULATION_PARAMS']
        ic_params = config['IC_PARAMS']
        e_grid = config['E_GRID']
        u_grid = config['U_GRID']

        param_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'preset_name': config['PRESET_NAME'],
            'MODEL_PARAMS': model_params.copy(),
            'ECONOMIC_PARAMS': economic_params.copy(),
            'SIMULATION_PARAMS': simulation_params.copy(),
            'IC_PARAMS': ic_params.copy(),
            'OUTPUT_CONFIG': output_config.copy(),
            'E_GRID': e_grid.tolist(),
            'U_GRID': u_grid.tolist(),
            'E_GRID_info': {
                'length': len(e_grid),
                'min': float(e_grid.min()) if len(e_grid) > 0 else None,
                'max': float(e_grid.max()) if len(e_grid) > 0 else None,
            },
            'U_GRID_info': {
                'length': len(u_grid),
                'u_range': [
                    float(u_grid[:, 0].min()),
                    float(u_grid[:, 0].max()),
                ] if len(u_grid) > 0 else None,
            },
            'run_config': {
                'workflow': 'species_br',
                'num_chunks': args.num_chunks,
                'num_cpus': args.num_cpus,
                'max_attempts': args.max_attempts,
            },
        }

        json_path = os.path.join(output_dir, 'run_parameters.json')
        with open(json_path, 'w') as f:
            json.dump(param_snapshot, f, indent=2)

        print('Configuration saved')

    run_species_br(args, config, output_dir)

    if args.num_chunks == 1:
        print("Combining the single species-BR chunk...")
        combine_species_br_chunk_results(
            output_dir=output_dir,
            num_chunks=1,
        )


if __name__ == "__main__":
    main()
