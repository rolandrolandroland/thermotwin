"""Dependency-light shared data structures for ThermoTwin."""

from .controls import (
    CurrentInput,
    PiecewiseConstantCurrent,
    current_at,
)

__all__ = ["CurrentInput", "PiecewiseConstantCurrent", "current_at"]
