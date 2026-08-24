"""Compatibility facade for :mod:`thermotwin.design.codesign`.

The implementation is split into typed models, physical evaluation, sampling,
optimization, robustness, and campaign orchestration modules. Existing imports
and ``python -m thermotwin.material_geometry_codesign`` remain supported.
"""

from .design.codesign import *  # noqa: F401,F403


def main() -> None:
    """Run the default CPU-first campaign through the compatibility API."""

    print(format_codesign_campaign_report(run_codesign_campaign()))


if __name__ == "__main__":
    main()
