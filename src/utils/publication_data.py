"""Validate shared publication data against its scientific configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_shared_species(path: Path, config: dict) -> dict:
    """Allow a different payoff coefficient, never a different species experiment."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing complete species-BR data: {path}")
    metadata_path = path.parent / "run_parameters.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing species-BR run metadata: {metadata_path}")
    with metadata_path.open() as stream:
        recorded = json.load(stream)
    if recorded.get("MODEL_PARAMS") != config["MODEL_PARAMS"]:
        raise ValueError("Shared species-BR model parameters differ from the active preset.")
    if recorded.get("SIMULATION_PARAMS") != config["SIMULATION_PARAMS"]:
        raise ValueError("Shared species-BR simulation parameters differ from the active preset.")
    if recorded.get("IC_PARAMS") != config["IC_PARAMS"]:
        raise ValueError("Shared species-BR initial-condition parameters differ from the active preset.")
    E_grid = np.asarray(config["E_GRID"], dtype=float)
    if "E_GRID" not in recorded or "U_GRID" not in recorded:
        raise ValueError(
            "Species-BR run metadata lacks exact grids. Recompute with the current "
            "main_speciesBR.py before publication analysis."
        )
    if recorded["E_GRID"] != E_grid.tolist():
        raise ValueError("Shared species-BR E grid values differ from the active preset.")
    grid_info = recorded.get("E_GRID_info", {})
    if (grid_info.get("length") != len(E_grid)
            or grid_info.get("min") != float(np.min(E_grid))
            or grid_info.get("max") != float(np.max(E_grid))):
        raise ValueError("Shared species-BR E grid differs from the active preset.")
    U_grid = np.asarray(config["U_GRID"], dtype=float)
    if recorded["U_GRID"] != U_grid.tolist():
        raise ValueError("Shared species-BR trait grid values differ from the active preset.")
    trait_info = recorded.get("U_GRID_info", {})
    if trait_info.get("length") != len(U_grid):
        raise ValueError("Shared species-BR trait grid differs from the active preset.")
    if len(U_grid) and trait_info.get("u_range") != [float(U_grid[:, 0].min()),
                                                       float(U_grid[:, 0].max())]:
        raise ValueError("Shared species-BR trait range differs from the active preset.")
    return {"source_preset": recorded.get("preset_name"),
            "species_sha256": sha256_file(path),
            "run_metadata_sha256": sha256_file(metadata_path)}
