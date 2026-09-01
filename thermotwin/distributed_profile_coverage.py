"""Compatibility entry point for the distributed profile-coverage report."""

from .reports.distributed_profile_coverage import main


if __name__ == "__main__":
    raise SystemExit(main())
