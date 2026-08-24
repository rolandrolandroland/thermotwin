"""Shared output locations for generated ThermoTwin figures."""

from pathlib import Path


FIGURES_DIRECTORY = Path(__file__).resolve().parent.parent / "figures"


def default_figure_path(filename: str) -> Path:
    """Return an absolute path inside the package figures directory."""

    candidate = Path(filename)
    if not filename or candidate.name != filename:
        raise ValueError("figure filename must be one plain filename")
    return FIGURES_DIRECTORY / filename
