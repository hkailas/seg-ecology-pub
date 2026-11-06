"""
Dynamic parameter loader - loads configuration based on active preset.

This module automatically loads parameter configurations from presets defined
in parameter_presets.py. To select a preset, set the SEG_PRESET environment
variable before running:

    export SEG_PRESET=model1_default
    python main3.py --chunk_id 0 --num_chunks 1 --num_cpus 4 ...

Or inline:
    SEG_PRESET=model1_test python main3.py ...
To see available presets:
    python -m src.config.parameter_presets
"""

import numpy as np
import os
import sys

### Preset selection

# Default preset name (can be overridden by environment variable)
# Set to 'model1_default'
ACTIVE_PRESET = os.environ.get('SEG_PRESET', 'model1_default')

# Load preset configuration
try:
    from src.config.parameter_presets import PRESETS, load_preset
except ImportError:
    print("ERROR: Could not import parameter_presets module.", file=sys.stderr)
    raise

# Validate and load the preset
if ACTIVE_PRESET not in PRESETS:
    from src.config.parameter_presets import list_presets
    print(f"\nERROR: Unknown preset '{ACTIVE_PRESET}'", file=sys.stderr)
    list_presets()
    sys.exit(1)

# Load the configuration
_config = load_preset(ACTIVE_PRESET)

### Export the parameters

SIMULATION_PARAMS = _config.get('SIMULATION_PARAMS', {})
MODEL_PARAMS = _config.get('MODEL_PARAMS', {})
ECONOMIC_PARAMS = _config.get('ECONOMIC_PARAMS', {})
IC_PARAMS = _config.get('IC_PARAMS', {})
LOGGING_PARAMS = _config.get('LOGGING_PARAMS', {})
OUTPUT_CONFIG = _config.get('OUTPUT_CONFIG', {})

E_GRID = _config.get('E_GRID', np.array([]))
u1_grid = _config.get('u1_grid', np.array([]))
u2_grid = _config.get('u2_grid', np.array([]))

### Compute derived grids

# Compute U_GRID from u1_grid and u2_grid
U_GRID_FULL = np.array(np.meshgrid(u1_grid, u2_grid)).T.reshape(-1, 2)

# Filter to upper triangle (u1 <= u2) because it is symmetric
U_GRID = U_GRID_FULL[U_GRID_FULL[:, 0] <= U_GRID_FULL[:, 1]]

### Metadata for tracking

# Store preset metadata for logging and reproducibility
PRESET_NAME = ACTIVE_PRESET
PRESET_DESCRIPTION = _config.get('description', '')

# Print active configuration when module is imported
if __name__ != "__main__":
    # Only print when imported, not when run directly
    print(f"[CONFIG] Using preset: '{PRESET_NAME}' - {PRESET_DESCRIPTION}")
