"""Compatibility facade for :mod:`thermotwin.design.operating_map`."""

from .design.operating_map import *  # noqa: F401,F403
from .design.operating_map import _match_heat_rate


if __name__ == "__main__":
    main()
