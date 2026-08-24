"""Compatibility facade for :mod:`thermotwin.physics.thermoelectric`.

New code should import the implementation from ``thermotwin.physics``.  This
module remains indefinitely so existing notebooks and documented imports keep
working.
"""

from .physics.thermoelectric import *  # noqa: F401,F403
