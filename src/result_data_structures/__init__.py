"""Shared data structures for simulation results and workflow result records."""

from .analysis_result_data_types import AnalysisResult
from .simulation_result_data_types import SimulationResult, is_failed_result
from .leader_br_at_species_br_result_data_types import (
    LeaderBrChunk,
    LeaderBrCombinedResult,
    LeaderBrOptimizationResult,
    LeaderBrResultRecord,
    LeaderBrWorkItem,
)
from .species_br_result_data_types import SpeciesBrChunk, SpeciesBrResultRecord
from .single_species_leader_br_result_data_types import (
    SingleSpeciesLeaderBrChunk,
    SingleSpeciesLeaderBrCombinedResult,
    SingleSpeciesLeaderBrOptimizationResult,
    SingleSpeciesLeaderBrResultRecord,
    SingleSpeciesLeaderBrWorkItem,
)

__all__ = [
    "AnalysisResult",
    "LeaderBrChunk",
    "LeaderBrCombinedResult",
    "LeaderBrOptimizationResult",
    "LeaderBrResultRecord",
    "LeaderBrWorkItem",
    "SimulationResult",
    "SingleSpeciesLeaderBrChunk",
    "SingleSpeciesLeaderBrCombinedResult",
    "SingleSpeciesLeaderBrOptimizationResult",
    "SingleSpeciesLeaderBrResultRecord",
    "SingleSpeciesLeaderBrWorkItem",
    "SpeciesBrChunk",
    "SpeciesBrResultRecord",
    "is_failed_result",
]
