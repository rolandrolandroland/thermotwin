"""Compatibility facade for :mod:`thermotwin.observations.quality`."""

from .observations.quality import *  # noqa: F401,F403
from .reports.dataset_quality import main, reference_dataset_quality_report


if __name__ == "__main__":
    main()
