"""Figure saving shared by manuscript entry points."""

from pathlib import Path
from collections.abc import Sequence
import matplotlib.pyplot as plt

FORMATS = ("png", "pdf", "svg")


def save_figure(figure, output_dir: Path, stem: str,
                formats: Sequence[str] = FORMATS, *, show: bool = False,
                save: bool = True) -> list[Path]:
    invalid = set(formats) - set(FORMATS)
    if invalid or not formats:
        raise ValueError(f"Invalid formats: {sorted(invalid)}")
    paths = []
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        for extension in dict.fromkeys(formats):
            path = output_dir / f"{stem}.{extension}"
            figure.savefig(path, format=extension, dpi=300, bbox_inches="tight")
            paths.append(path)
            print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(figure)
    return paths
