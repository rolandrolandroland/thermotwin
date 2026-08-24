"""Dependency-free space-filling design generation and feature encoding."""

import random
from typing import Tuple

from ..materials import N_TYPE_SAMPLES, P_TYPE_SAMPLES
from .models import ModuleGeometry, PrototypeDesign


def latin_hypercube(
    count: int,
    dimensions: int,
    seed: int,
) -> Tuple[Tuple[float, ...], ...]:
    """Return a reproducible unit-cube Latin hypercube without dependencies."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions <= 0
    ):
        raise ValueError("dimensions must be a positive integer")
    generator = random.Random(seed)
    columns = []
    for _ in range(dimensions):
        strata = list(range(count))
        generator.shuffle(strata)
        columns.append(
            tuple((stratum + generator.random()) / count for stratum in strata)
        )
    return tuple(
        tuple(columns[dimension][row] for dimension in range(dimensions))
        for row in range(count)
    )


def generate_space_filling_designs(
    count: int,
    *,
    seed: int,
    prefix: str,
) -> Tuple[PrototypeDesign, ...]:
    """Map an eight-dimensional Latin hypercube into the design envelope."""

    rows = latin_hypercube(count, 8, seed)
    designs = []
    for index, row in enumerate(rows):
        p_index = min(int(row[0] * len(P_TYPE_SAMPLES)), len(P_TYPE_SAMPLES) - 1)
        n_index = min(int(row[1] * len(N_TYPE_SAMPLES)), len(N_TYPE_SAMPLES) - 1)
        couple_count = int(round(80.0 + 80.0 * row[2]))
        leg_length = (0.8 + 1.6 * row[3]) * 1.0e-3
        leg_area = (0.8 + 1.6 * row[4]) * 1.0e-6
        contact_resistance = 0.10 + 0.40 * row[5]
        cold_conductance = 1.5 + 3.5 * row[6]
        hot_conductance = 3.0 + 5.0 * row[7]
        designs.append(
            PrototypeDesign(
                f"{prefix}-{index + 1:03d}",
                p_index,
                n_index,
                ModuleGeometry(couple_count, leg_length, leg_area),
                contact_resistance,
                cold_conductance,
                hot_conductance,
            )
        )
    return tuple(designs)


def design_features(design: PrototypeDesign) -> Tuple[float, ...]:
    """Return normalized continuous features plus one-hot material choices."""

    continuous = (
        (design.geometry.couple_count - 80.0) / 80.0,
        (design.geometry.leg_length / 1.0e-3 - 0.8) / 1.6,
        (design.geometry.leg_area / 1.0e-6 - 0.8) / 1.6,
        (design.symmetric_contact_resistance - 0.10) / 0.40,
        (design.cold_exchanger_conductance - 1.5) / 3.5,
        (design.hot_exchanger_conductance - 3.0) / 5.0,
    )
    p_one_hot = tuple(
        1.0 if index == design.p_sample_index else 0.0
        for index in range(len(P_TYPE_SAMPLES))
    )
    n_one_hot = tuple(
        1.0 if index == design.n_sample_index else 0.0
        for index in range(len(N_TYPE_SAMPLES))
    )
    return continuous + p_one_hot + n_one_hot
