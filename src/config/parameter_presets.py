"""
Parameter preset configurations for different models and runs.

This module defines all available parameter configurations. To use a preset,
set the SEG_PRESET environment variable before running the simulation
"""

import numpy as np
import itertools
import copy

# Base configuration templates (shared parameters)
_BASE_SIMULATION_PARAMS = dict(
    t_start=0,
    t_max=1800000000,
    num_points=None,
    eq_tsteps=5000,
    eq_thresh=1e-6,
    abs_thresh=1e-6,
    chunk_size=20000,
    verbose=False,
    evo_dyn=True,
    store_sol=False,
    min_thresh=1e-4,
    timeout_seconds=130,
    method='RK45',
    maxstep=50,
)

_BASE_IC_PARAMS = dict(
    num_pops=2,
    min_pop=10,
    max_pop=500,
    min_trait=0,
    base_seed=546,
)

_BASE_LOGGING_PARAMS = dict(
    log_level='WARNING'
)

_BASE_ECONOMIC_PARAMS = dict(
    L=100.0,
    p=9.0,
    delta=0.1,
    alpha_leader=0.02,
    c1=0, c4=0, c6=0
)


# PRESET DEFINITIONS
# ==========================================

PRESETS = {

    # Fig 2 

    'model1_f2_A': {
            'description': 'Model 1 single species without bully - for dynamics',
            
            'SIMULATION_PARAMS': {
                **_BASE_SIMULATION_PARAMS,
                'model_name': 'model1'
            },
            
            'MODEL_PARAMS': {
                'r_max': 0.05,
                'K': 1000,
                'g': 0.1,
                'alpha': 1,
                'beta': 1,
                'beta_bully': np.nan,
                'sigma_a_squared': np.nan,
                'E': 0.001,
                'd': 0.1,
                'sigma': 0.5,
                'fd': False,  # Trait-dependent competition; False uses alpha=1.
                'include_E': True
            },
            
            'ECONOMIC_PARAMS': {
                'L': 100.0,
                'p': 10,
                'delta': 0.18,
                'alpha_leader': 0.01,
                'c1': 500, 'c4': 0, 'c6': 0
            },
            
            'IC_PARAMS': {
                **_BASE_IC_PARAMS,
                'num_pops': 1,
                'max_trait': 30,
                'num_ICs': 20,
                'num_ICs_E': 50,
                'initial_population': 63,
                'initial_trait': 6.4637
            },
            
            'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
            
            'OUTPUT_CONFIG': {
                'date': 'Fig2-A',
                'compute_dir': 'compute_data/local',
                'base_dir': 'results'
            },
            
            'E_GRID': np.concatenate((
                np.linspace(0.001, 0.56, 10),
                np.arange(0.5605, 1, 0.0005)
            )),
        
            },
        
    'model1_f2_B': {
        'description': 'Model 1 single species with bully - for dynamics',
        
        'SIMULATION_PARAMS': {
            **_BASE_SIMULATION_PARAMS,
            'model_name': 'model1'
        },
        
        'MODEL_PARAMS': {
            'r_max': 0.05,
            'K': 1000,
            'g': 0.1,
            'alpha': 1,
            'beta': 1,
            'beta_bully': 2,
            'sigma_a_squared': 4,
            'E': 0.001,
            'd': 0.1,
            'sigma': 0.5,
            'fd': True,  # Trait-dependent competition; False uses alpha=1.
            'include_E': True
        },
        
        'ECONOMIC_PARAMS': {
            'L': 100.0,
            'p': 10,
            'delta': 0.18,
            'alpha_leader': 0.01,
            'c1': 500, 'c4': 0, 'c6': 0
        },
        
        'IC_PARAMS': {
            **_BASE_IC_PARAMS,
            'num_pops': 1,
            'max_trait': 30,
            'num_ICs': 20,
            'num_ICs_E': 50,
            'initial_population': 63,
            'initial_trait': 6.4637
        },
        
        'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
        
        'OUTPUT_CONFIG': {
            'date': 'Fig2-B',
            'compute_dir': 'compute_data/local',
            'base_dir': 'results'
        },
        
        'E_GRID': np.concatenate((
            np.linspace(0.001, 0.56, 10),
            np.arange(0.5605, 1, 0.0005)
        )),
    
        },
    
    'model3_f2_C': {
        'description': 'Model 3 - 1 species - for dynamics',
        
        'SIMULATION_PARAMS': {
            **_BASE_SIMULATION_PARAMS,
            't_max': 1000000,
            'model_name': 'model3'
        },
        
        'MODEL_PARAMS': {
            'r_max': 0.2,
            'K': 1000,
            'g': 0.2,
            'alpha': 1,
            'beta': 1,
            'beta_bully': 2,
            'sigma_a_squared': 4,
            'E': 0.001,
            'd': 0.1,
            'sigma': 0.05,
            'fd': True,  # Trait-dependent competition; False uses alpha=1.
            'include_E': True
        },
        
        'ECONOMIC_PARAMS': {
            'L': 100.0,
            'p': 8,
            'delta': 0.1,
            'alpha_leader': 0.1
        },
        
        'IC_PARAMS': {
            **_BASE_IC_PARAMS,
            'num_pops': 1,
            'max_trait': 10,
            'num_ICs': 5,
            'num_ICs_E': 5,
            'initial_population': 63,
            'initial_trait': 6.4637
        },
        
        'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
        
        'OUTPUT_CONFIG': {
            'date': 'Fig2-C',
            'compute_dir': 'compute_data/local',
            'base_dir': 'results'
        },
        
        'E_GRID': np.linspace(0.001, 1.0, 30),
        
    },
    # Fig 3
    'model3_f3_adaptive_landscapes': {
        'description': 'Model 3 - 1 species - adaptive landscapes (fig 3)',
        
        'SIMULATION_PARAMS': {
            **_BASE_SIMULATION_PARAMS,
            't_max': 1000000,
            'model_name': 'model3'
        },
        
        'MODEL_PARAMS': {
            'r_max': 0.2,
            'K': 1000,
            'g': 0.2,
            'alpha': 1,
            'beta': 1,
            'beta_bully': 2,
            'sigma_a_squared': 4,
            'E': 0.001,
            'd': 0.1,
            'sigma': 0.05,
            'fd': True,  # Trait-dependent competition; False uses alpha=1.
            'include_E': True
        },
        
        'ECONOMIC_PARAMS': {
            'L': 100.0,
            'p': 8,
            'delta': 0.1,
            'alpha_leader': 0.1
        },
        
        'IC_PARAMS': {
            **_BASE_IC_PARAMS,
            'num_pops': 1,
            'max_trait': 10,
            'num_ICs': 5,
            'num_ICs_E': 5,
            'base_seed': 546
        },
        
        'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
        
        'OUTPUT_CONFIG': {
            'date': 'Fig3-A3-A4',
            'compute_dir': 'compute_data/local',
            'base_dir': 'results'
        },
        
        'E_GRID': np.linspace(0.001, 1.0, 30),
        
    },

    'model1_f3_adaptive_landscapes': {
        'description': 'Model 1 - multi-species adaptive landscapes for Fig 3',
        
        'SIMULATION_PARAMS': {
            **_BASE_SIMULATION_PARAMS,
            'model_name': 'model1'
        },
        
        'MODEL_PARAMS': {
            'r_max': 0.05,
            'K': 1000,
            'g': 0.1,
            'alpha': 1,
            'beta': 1,
            'beta_bully': 2,
            'sigma_a_squared': 4,
            'E': 0.001,
            'd': 0.1,
            'sigma': 0.1,
            'fd': True,  # Trait-dependent competition; False uses alpha=1.
            'include_E': True
        },
        
        'ECONOMIC_PARAMS': {
            **_BASE_ECONOMIC_PARAMS,
            'p': 100,
            'delta': 0.1,
            'alpha_leader': 0.01,
            'c1': 0, 'c4': 2.5, 'c6': 0.2
        },
        
        'IC_PARAMS': {
            **_BASE_IC_PARAMS,
            'num_pops': 8,
            'max_trait': 30,
            'num_ICs': 20,
            'num_ICs_E': 80,
            'base_seed': 546
        },
        
        'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
        
        'OUTPUT_CONFIG': {
            'date': 'Fig3-A1-A2',
            'compute_dir': 'compute_data/local',
            'base_dir': 'results'
        },
        
        'E_GRID': np.concatenate((
            np.linspace(0.001, 0.56, 10),
            np.arange(0.5601, 1, 0.0001)
        )),

    },
        
    # Fig 4
    'model1_f4_adaptive_landscapes': {
        'description': 'Model 1 - multi-species adaptive landscapes for Fig 4',
        
        'SIMULATION_PARAMS': {
            **_BASE_SIMULATION_PARAMS,
            'model_name': 'model1'
        },
        
        'MODEL_PARAMS': {
            'r_max': 0.05,
            'K': 1000,
            'g': 0.1,
            'alpha': 1,
            'beta': 1,
            'beta_bully': 2,
            'sigma_a_squared': 4,
            'E': 0.001,
            'd': 0.1,
            'sigma': 0.1,
            'fd': True,  # Trait-dependent competition; False uses alpha=1.
            'include_E': True
        },
        
        'ECONOMIC_PARAMS': {
            **_BASE_ECONOMIC_PARAMS,
            'p': 10,
            'delta': 0.1,
            'alpha_leader': 0.01,
            'c1': 0, 'c4': 2.5, 'c6': 0.2
        },
        
        'IC_PARAMS': {
            **_BASE_IC_PARAMS,
            'num_pops': 8,
            'max_trait': 30,
            'num_ICs': 20,
            'num_ICs_E': 80,
            'base_seed': 546
        },
        
        'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
        
        'OUTPUT_CONFIG': {
            'date': 'Fig4',
            'compute_dir': 'compute_data/local',
            'base_dir': 'results'
        },
        
        'E_GRID': np.concatenate((
            np.linspace(0.001, 0.56, 10),
            np.arange(0.5601, 1, 0.0001)
        )),

    },
    
    # Fig 5
    'model1_f5_AD_evo': {
        'description': 'Model 1 single species without bully - payoff+= c1/trait',
        
        'SIMULATION_PARAMS': {
            **_BASE_SIMULATION_PARAMS,
            'model_name': 'model1'
        },
        
        'MODEL_PARAMS': {
            'r_max': 1,
            'K': 1000,
            'g': 0.3,
            'alpha': 2,
            'beta': 2,
            'beta_bully': 0,
            'sigma_a_squared': 4,
            'E': 0.001,
            'd': 0.1,
            'sigma': 0.1,
            'fd': True,  # Trait-dependent competition; False uses alpha=1.
            'include_E': True
        },
        
        'ECONOMIC_PARAMS': {
            'L': 100.0,
            'p': 10,
            'delta': 0.18,
            'alpha_leader': 0.01,
            'c1': 500, 'c4': 0, 'c6': 0
        },
        
        'IC_PARAMS': {
            **_BASE_IC_PARAMS,
            'num_pops': 1,
            'max_trait': 30,
            'num_ICs': 5,
            'num_ICs_E': 5,
        },
        
        'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
        
        'OUTPUT_CONFIG': {
            'date': 'Fig5_AD_evo',
            'compute_dir': 'compute_data/local',
            'base_dir': 'results'
        },
        
        'E_GRID': np.concatenate((
            np.linspace(0.001, 0.3, 10),
            np.arange(0.3005, 1, 0.0001)
        )),

        'u_grid_n': np.linspace(0, 30, 1000),
        

    },

    'model1_f5_BE_evo': {
        'description': 'Model 1 single species with bully - payoff+= c1/trait',
        
        'SIMULATION_PARAMS': {
            **_BASE_SIMULATION_PARAMS,
            'model_name': 'model1'
        },
        
        'MODEL_PARAMS': {
            'r_max': 1,
            'K': 1000,
            'g': 0.3,
            'alpha': 2,
            'beta': 2,
            'beta_bully': 2,
            'sigma_a_squared': 4,
            'E': 0.001,
            'd': 0.1,
            'sigma': 0.1,
            'fd': True,  # Trait-dependent competition; False uses alpha=1.
            'include_E': True
        },
        
        'ECONOMIC_PARAMS': {
            'L': 100.0,
            'p': 10,
            'delta': 0.18,
            'alpha_leader': 0.01,
            'c1': 500, 'c4': 0, 'c6': 0
        },
        
        'IC_PARAMS': {
            **_BASE_IC_PARAMS,
            'num_pops': 1,
            'max_trait': 30,
            'num_ICs': 5,
            'num_ICs_E': 10,
        },
        
        'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
        
        'OUTPUT_CONFIG': {
            'date': 'Fig5_BE_evo',
            'compute_dir': 'compute_data/local',
            'base_dir': 'results'
        },
        
        'E_GRID': np.concatenate((
            np.linspace(0.001, 0.3, 10),
            np.arange(0.3005, 1, 0.0001)
        )),

        'u_grid_n': np.linspace(0, 30, 10000),
        
        },

    'model1_f5_AD_eco': {
        'description': 'Model 1 single species without bully - eco-aware',
        
        'SIMULATION_PARAMS': {
            **_BASE_SIMULATION_PARAMS,
            'model_name': 'model1'
        },
        
        'MODEL_PARAMS': {
            'r_max': 1,
            'K': 1000,
            'g': 0.3,
            'alpha': 2,
            'beta': 2,
            'beta_bully': np.nan,
            'sigma_a_squared': np.nan,
            'E': 0.001,
            'd': 0.1,
            'sigma': 0.1,
            'fd': False,  # Trait-dependent competition; False uses alpha=1.
            'include_E': True
        },
        
        'ECONOMIC_PARAMS': {
            **_BASE_ECONOMIC_PARAMS,
            'c1': 0, 'c4': 0, 'c6': 0
        },
        
        'IC_PARAMS': {
            **_BASE_IC_PARAMS,
            'num_pops': 1,
            'max_trait': 30,
            'num_ICs': 5,
            'num_ICs_E': 5,
        },
        
        'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
        
        'OUTPUT_CONFIG': {
            'date': 'Fig5_AD_eco',
            'compute_dir': 'compute_data/local',
            'base_dir': 'results'
        },
        
        'E_GRID': np.concatenate((
            np.linspace(0.001, 0.3, 10),
            np.arange(0.3005, 1, 0.0001)
        )),

        'u_grid_n': np.linspace(0, 30, 10000),
        

        },
    
    'model1_f5_CF_evo': {
            'description': 'Model 1 - multi-species - spbr+ldbr_at_spbr - evo-aware',
            
            'SIMULATION_PARAMS': {
                **_BASE_SIMULATION_PARAMS,
                'model_name': 'model1'
            },
            
            'MODEL_PARAMS': {
                'r_max': 0.05,
                'K': 1000,
                'g': 0.1,
                'alpha': 1,
                'beta': 1,
                'beta_bully': 2,
                'sigma_a_squared': 4,
                'E': 0.001,
                'd': 0.1,
                'sigma': 0.1,
                'fd': True,  # Trait-dependent competition; False uses alpha=1.
                'include_E': True
            },
            
            'ECONOMIC_PARAMS': {
                **_BASE_ECONOMIC_PARAMS,
                'p': 10,
                'delta': 0.1,
                'alpha_leader': 0.01,
                'c1': 0, 'c4': 2.5, 'c6': 0.2
            },
            
            'IC_PARAMS': {
                **_BASE_IC_PARAMS,
                'num_pops': 8,
                'max_trait': 30,
                'num_ICs': 20,
                'num_ICs_E': 80,
                'base_seed': 546
            },
            
            'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
            
            'OUTPUT_CONFIG': {
                'date': 'Fig5_CF_evo',
                'compute_dir': 'compute_data/local',
                'base_dir': 'results'
            },
            
            'E_GRID': np.concatenate((
                np.linspace(0.001, 0.3, 10),
                np.arange(0.3005, 1, 0.0001)
            )),
    
        },
        
    'model1_f5_CF_eco': {
            'description': 'Model 1 - multi-species - spbr+ldbr_at_spbr - eco-aware',
            
            'SIMULATION_PARAMS': {
                **_BASE_SIMULATION_PARAMS,
                'model_name': 'model1'
            },
            
            'MODEL_PARAMS': {
                'r_max': 0.05,
                'K': 1000,
                'g': 0.1,
                'alpha': 1,
                'beta': 1,
                'beta_bully': 2,
                'sigma_a_squared': 4,
                'E': 0.001,
                'd': 0.1,
                'sigma': 0.1,
                'fd': True,  # Trait-dependent competition; False uses alpha=1.
                'include_E': True
            },
            
            'ECONOMIC_PARAMS': {
                **_BASE_ECONOMIC_PARAMS,
                'p': 10,
                'delta': 0.1,
                'alpha_leader': 0.01,
                'c1': 0, 'c4': 2.5, 'c6': 0.2
            },
            
            'IC_PARAMS': {
                **_BASE_IC_PARAMS,
                'num_pops': 8,
                'max_trait': 30,
                'num_ICs': 20,
                'num_ICs_E': 80,
                'base_seed': 546
            },
            
            'LOGGING_PARAMS': _BASE_LOGGING_PARAMS,
            
            'OUTPUT_CONFIG': {
                'date': 'Fig5_CF_eco',
                'compute_dir': 'compute_data/local',
                'base_dir': 'results'
            },
            
            'E_GRID': np.concatenate((
                np.linspace(0.001, 0.3, 10),
                np.arange(0.3005, 1, 0.0001)
            )),
        
            },

}

# Publication datasets use a fixed seed.  A run's chunk partition must still be
# retained because the per-chunk RNG changes the sampled initial conditions.
for _publication_name in tuple(PRESETS):
    if _publication_name.startswith(("model1_f", "model3_f")):
        PRESETS[_publication_name]["IC_PARAMS"]["base_seed"] = 546

# The ecological objective is the same experiment with only c1 removed.
PRESETS["model1_f5_BE_eco"] = copy.deepcopy(PRESETS["model1_f5_BE_evo"])
PRESETS["model1_f5_BE_eco"]["description"] = "Figure 5B,E: same ecology, c1=0"
PRESETS["model1_f5_BE_eco"]["ECONOMIC_PARAMS"]["c1"] = 0
PRESETS["model1_f5_BE_eco"]["OUTPUT_CONFIG"]["date"] = "Fig5_BE_eco"
for _be_name in ("model1_f5_BE_evo", "model1_f5_BE_eco"):
    PRESETS[_be_name]["u_grid_n"] = np.linspace(0, 30, 1000)

_analytical = {
    "description": "Figure 5A,D: analytical single-species model",
    "MODEL_PARAMS": {"r_max": 1.0, "g": 0.3, "alpha": 2.0,
                     "beta": 2.0, "d": 0.1, "K": 1000.0},
    "ECONOMIC_PARAMS": {"L": 100.0, "p": 10.0, "delta": 0.18,
                        "alpha_leader": 0.01, "c1": 500.0},
    "SIMULATION_PARAMS": {"model_name": "model1"},
    "IC_PARAMS": {"num_pops": 1, "base_seed": 546},
    "OUTPUT_CONFIG": {"date": "analytical", "base_dir": "results"},
    "ANALYTICAL_PARAMS": {
        "E_min": 0.01, "E_max": 1.0, "E_points": 10000,
        "u_min": 0.0, "u_max": 20.0, "u_points": 10000,
        "leader_E_min": 0.01, "naive_E_min": 0.00001,
        "nash_u_min": 0.01, "nash_u_max": 20.0,
        "near_optimal_fraction": 0.01,
    },
}
PRESETS["model1_analytical"] = _analytical
PRESETS["model1_analytical_c0"] = copy.deepcopy(_analytical)
PRESETS["model1_analytical_c0"]["description"] = "Figure 5D: analytical model with c1=0"
PRESETS["model1_analytical_c0"]["ECONOMIC_PARAMS"]["c1"] = 0.0
PRESETS["model1_analytical_c0"]["OUTPUT_CONFIG"]["date"] = "analytical_c0"


def load_preset(preset_name, overrides=None):
    """
    Load a parameter preset by name.
    
    Args:
        preset_name: Name of preset from PRESETS dictionary
        overrides: Dictionary of parameters to override (optional)
        
    Returns:
        Complete parameter configuration dictionary
        
    Raises:
        ValueError: If preset_name is not found
    """
    if preset_name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'.\n"
            f"Available presets: {available}\n"
            f"Set SEG_PRESET environment variable to one of these values."
        )
    
    # Deep copy to avoid modifying the original
    import copy
    preset = copy.deepcopy(PRESETS[preset_name])
    
    # Apply overrides if provided
    if overrides:
        for key, value in overrides.items():
            if key in preset:
                if isinstance(preset[key], dict) and isinstance(value, dict):
                    preset[key].update(value)
                else:
                    preset[key] = value
            else:
                preset[key] = value

    n_sp = preset['IC_PARAMS']['num_pops']
    u_grid_n = preset.get('u_grid_n', np.array([]))
    
    # generate the upper triangular part of the grid only
    U_GRID = np.array(list(itertools.combinations_with_replacement(u_grid_n, n_sp)))
    # Add computed U_GRID to the preset
    preset['U_GRID'] = U_GRID

    return preset


def list_presets():
    """Print all available presets with descriptions."""
    print("\n" + "="*70)
    print("Available Parameter Presets")
    print("="*70)
    
    for name, config in PRESETS.items():
        desc = config.get('description', 'No description')
        model = config.get('SIMULATION_PARAMS', {}).get('model_name', 'unknown')
        date = config.get('OUTPUT_CONFIG', {}).get('date', 'N/A')
        
        print(f"\n  {name}")
        print(f"    Model: {model}")
        print(f"    Date:  {date}")
        print(f"    Desc:  {desc}")
    
    print("\n" + "="*70)
    print("Usage:")
    print("  export SEG_PRESET=model1_default")
    print("  python main3.py --chunk_id 0 --num_chunks 1 --num_cpus 4 ...")
    print("="*70 + "\n")


if __name__ == "__main__":
    # If run directly, list all presets
    list_presets()
