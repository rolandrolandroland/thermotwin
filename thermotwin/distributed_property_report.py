"""Compatibility command for the distributed-property report."""

from .reports.distributed_properties import *  # noqa: F401,F403
from .reports.distributed_properties import main


if __name__ == "__main__":
    main()
