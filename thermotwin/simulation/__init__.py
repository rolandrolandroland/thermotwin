"""Diagnostics, experiments, and conventional transient execution.

The package deliberately avoids eager imports. Shared current-input structures
live in :mod:`thermotwin.core.controls`; ``simulation.controls`` remains a
compatibility facade.
"""

from .distributed import (
    DistributedExperimentResult,
    DistributedLegExperiment,
    DistributedTemperatureTrajectory,
    DistributedTrajectoryDiagnostics,
    distributed_reference_experiment,
    distributed_identifiability_experiments,
    distributed_inverse_constant_experiments,
    evaluate_distributed_trajectory,
    integrate_distributed_leg,
    reference_distributed_material,
    run_distributed_leg_experiment,
)

__all__ = [
    "DistributedExperimentResult",
    "DistributedLegExperiment",
    "DistributedTemperatureTrajectory",
    "DistributedTrajectoryDiagnostics",
    "distributed_reference_experiment",
    "distributed_identifiability_experiments",
    "distributed_inverse_constant_experiments",
    "evaluate_distributed_trajectory",
    "integrate_distributed_leg",
    "reference_distributed_material",
    "run_distributed_leg_experiment",
]
