"""Compatibility entry point for the distributed observation-sufficiency report."""

from .reports.distributed_observation_identifiability import main


if __name__ == "__main__":
    raise SystemExit(main())
