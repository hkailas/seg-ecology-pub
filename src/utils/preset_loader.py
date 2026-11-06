"""
Utility functions for loading and unpacking preset configurations.
"""

import numpy as np
from src.config.parameter_presets import load_preset

def load_and_unpack_preset(preset_name):
    """
    Load a preset configuration and unpack all parameters.
    
    Args:
        preset_name (str): Name of the preset to load
        
    Returns:
        dict: Dictionary containing all unpacked parameters:
            - SIMULATION_PARAMS
            - MODEL_PARAMS
            - ECONOMIC_PARAMS
            - IC_PARAMS
            - OUTPUT_CONFIG
            - E_GRID
            - U_GRID
            - PRESET_NAME
            - PRESET_DESCRIPTION
            
    Raises:
        ValueError: If preset cannot be loaded
    """
    # Load preset configuration
    config = load_preset(preset_name)
    
    # Extract parameters from preset
    unpacked = {
        'SIMULATION_PARAMS': config.get('SIMULATION_PARAMS', {}),
        'MODEL_PARAMS': config.get('MODEL_PARAMS', {}),
        'ECONOMIC_PARAMS': config.get('ECONOMIC_PARAMS', {}),
        'IC_PARAMS': config.get('IC_PARAMS', {}),
        'OUTPUT_CONFIG': config.get('OUTPUT_CONFIG', {}),
        'ANALYTICAL_PARAMS': config.get('ANALYTICAL_PARAMS', {}),
        'E_GRID': config.get('E_GRID', np.array([])),
        'U_GRID': config.get('U_GRID', np.array([])),
        'PRESET_NAME': preset_name,
        'PRESET_DESCRIPTION': config.get('description', 'No description')
    }
    
    return unpacked
