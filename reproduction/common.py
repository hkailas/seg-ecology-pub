"""Shared CLI and dataset helpers for manuscript figures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.utils.publication_data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = Path(
    os.environ.get("SEG_REPRODUCTION_ROOT", ROOT / "results" / "reproduce_paper")
).resolve()
DATA_ROOT = REPRO_ROOT / "datasets"
FIGURE_ROOT = REPRO_ROOT / "figures"
MANIFEST = REPRO_ROOT / "publication_manifest.json"
PRECOMPUTED_ROOT = ROOT / "reproduction" / "datasets"
PRECOMPUTED_MANIFEST = PRECOMPUTED_ROOT / "publication_manifest.json"


def figure_parser(number: int, *, stages=("compute", "analyze", "plot")):
    parser = argparse.ArgumentParser(description=f"Reproduce manuscript Figure {number}.")
    parser.add_argument("--stage", choices=(*stages, "plot_precomputed"),
                        default="plot" if number >= 4 else "compute",
                        help="plot reads computed results; plot_precomputed reads reproduction/datasets")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=("png", "pdf", "svg"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def dataset_path(path: Path, stage: str) -> Path:
    """Select published inputs without changing any computation destinations."""
    return PRECOMPUTED_ROOT / path.relative_to(DATA_ROOT) if stage == "plot_precomputed" else path


def require(path: Path, command: str, *, stage: str | None = None) -> Path:
    if not path.is_file():
        if stage == "plot_precomputed":
            raise FileNotFoundError(
                f"Missing precomputed dataset {path}. Restore the published data "
                "under reproduction/datasets, or use --stage compute to generate "
                "local results followed by --stage plot."
            )
        raise FileNotFoundError(f"Missing {path}. Generate it with: {command}")
    return path


def verify_stage_manifest(stage: str, datasets: dict[str, Path]):
    """Precomputed plots must validate their own inventory, never a local run's."""
    path = PRECOMPUTED_MANIFEST if stage == "plot_precomputed" else MANIFEST
    if stage == "plot_precomputed":
        require(path, "python main_publication_manifest.py --precomputed", stage=stage)
    verify_manifest(path, datasets)


def command(script: str, preset: str, *args: str):
    """Separate process prevents import-time SEG_PRESET state from leaking."""
    environment = os.environ.copy()
    environment["SEG_PRESET"] = preset
    environment["SEG_RESULTS_ROOT"] = str(DATA_ROOT)
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                   cwd=ROOT, env=environment, check=True)


def chunk_pipeline(preset: str, *, species: bool, chunks: int, cpus: int):
    if chunks < 1 or cpus < 1:
        raise ValueError("--num-chunks and --num-cpus must be positive.")
    if species:
        for index in range(chunks):
            command("main_speciesBR.py", preset, "--chunk_id", index,
                    "--num_chunks", chunks, "--num_cpus", cpus)
        if chunks > 1:
            command("main_combine_spbr.py", preset, "--num_chunks", chunks)
    for index in range(chunks):
        command("main_leaderBR.py", preset, "--chunk_id", index,
                "--num_chunks", chunks, "--num_cpus", cpus)
    if chunks > 1:
        command("main_combine_leader_br.py", preset, "--num_chunks", chunks)


def write_manifest(path: Path, datasets: dict[str, Path], *, panels: dict[str, list[str]]):
    """Write an inventory after computation or analysis, never during plot-only use."""
    import importlib.metadata
    import subprocess as process

    versions = {}
    for package in ("numpy", "scipy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    commit = process.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=False).stdout.strip() or None
    inventory = {}
    for key, dataset in datasets.items():
        require(dataset, f"reproduce the source dataset for {key}")
        metadata = dataset.parent / "run_parameters.json"
        if not metadata.is_file():
            metadata = dataset.parent / "input_provenance.json"
        run_config = None
        if metadata.is_file():
            with metadata.open() as stream:
                run_config = json.load(stream)
        inventory[key] = {"path": Path(os.path.relpath(dataset, path.parent)).as_posix(),
                          "sha256": sha256_file(dataset),
                          "schema_version": 1,
                          "run_parameters": run_config,
                          "metadata_path": Path(os.path.relpath(metadata, path.parent)).as_posix() if metadata.is_file() else None,
                          "panels": panels.get(key, [])}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump({"commit": commit, "dependencies": versions,
                   "datasets": inventory}, stream, indent=2)


def verify_manifest(path: Path, datasets: dict[str, Path]):
    """Check supplied bundle files against the inventory when it is present."""
    if not path.is_file():
        return
    with path.open() as stream:
        inventory = json.load(stream).get("datasets", {})
    for key, dataset in datasets.items():
        entry = inventory.get(key)
        if entry is None:
            option = " --precomputed" if path == PRECOMPUTED_MANIFEST else ""
            raise ValueError(
                f"Publication manifest does not list {key}; regenerate it with "
                f"python main_publication_manifest.py{option}."
            )
        if entry.get("sha256") != sha256_file(dataset):
            raise ValueError(f"Publication dataset checksum mismatch: {dataset}.")
