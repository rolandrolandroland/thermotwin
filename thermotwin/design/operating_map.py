"""Steady cooling/heating COP maps across current, lift, and contacts."""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Optional, Sequence, Tuple

from ..numerics.integration import first_rising_crossing_bracket
from ..simulation.four_node_experiments import constant_current_contact_reference_experiment
from ..physics.four_node import four_node_contact_steady_state
from ..simulation.two_node_experiments import constant_current_reference_experiment
from ..physics.thermoelectric import (
    cold_side_heat,
    electrical_power,
    hot_side_heat,
)
from ..physics.two_node import two_node_steady_state


@dataclass(frozen=True)
class COPOperatingMapConfig:
    """Frozen generic operating envelope and fair-comparison targets."""

    mean_reservoir_temperature: float = 300.0
    external_temperature_lifts: Tuple[float, ...] = (
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
    )
    currents: Tuple[float, ...] = tuple(
        0.05 * index for index in range(1, 31)
    )
    symmetric_contact_resistances: Tuple[float, ...] = (0.10, 0.25, 0.50)
    minimum_useful_heat_rate: float = 1.0
    equal_cooling_target: float = 3.0
    equal_heating_target: float = 5.0
    maximum_current: float = 1.5
    current_tolerance: float = 1e-7
    heat_rate_bracket_subdivisions: int = 64

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.mean_reservoir_temperature)
            or self.mean_reservoir_temperature <= 0.0
        ):
            raise ValueError("mean reservoir temperature must be positive and finite")
        for name, values, allow_zero in (
            ("external temperature lifts", self.external_temperature_lifts, True),
            ("currents", self.currents, False),
            ("contact resistances", self.symmetric_contact_resistances, False),
        ):
            if not values or any(
                not math.isfinite(value)
                or value < 0.0
                or (not allow_zero and value == 0.0)
                for value in values
            ):
                raise ValueError(f"{name} contain invalid values")
        if tuple(sorted(set(self.external_temperature_lifts))) != tuple(
            self.external_temperature_lifts
        ):
            raise ValueError("external temperature lifts must be unique and ordered")
        if tuple(sorted(set(self.currents))) != tuple(self.currents):
            raise ValueError("currents must be unique and ordered")
        if len(set(self.symmetric_contact_resistances)) != len(
            self.symmetric_contact_resistances
        ):
            raise ValueError("contact resistances must be unique")
        for name, value in (
            ("minimum useful heat rate", self.minimum_useful_heat_rate),
            ("equal cooling target", self.equal_cooling_target),
            ("equal heating target", self.equal_heating_target),
            ("maximum current", self.maximum_current),
            ("current tolerance", self.current_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.currents[-1] > self.maximum_current:
            raise ValueError("current grid exceeds the maximum current")
        if (
            isinstance(self.heat_rate_bracket_subdivisions, bool)
            or not isinstance(self.heat_rate_bracket_subdivisions, int)
            or self.heat_rate_bracket_subdivisions < 2
        ):
            raise ValueError("heat-rate bracket subdivisions must be at least two")


class SteadyOperatingPoint(NamedTuple):
    """One steady operating point with three temperature-lift definitions."""

    topology: str
    symmetric_contact_resistance: Optional[float]
    external_temperature_lift: float
    current: float
    cold_reservoir_temperature: float
    hot_reservoir_temperature: float
    cold_face_temperature: float
    hot_face_temperature: float
    cold_exchanger_temperature: float
    hot_exchanger_temperature: float
    face_temperature_lift: float
    exchanger_temperature_lift: float
    cold_contact_temperature_drop: float
    hot_contact_temperature_drop: float
    delivered_cooling_rate: float
    delivered_heating_rate: float
    electrical_power: float
    cooling_cop: Optional[float]
    heating_cop: Optional[float]


class OptimalOperatingSummary(NamedTuple):
    topology: str
    symmetric_contact_resistance: Optional[float]
    external_temperature_lift: float
    mode: str
    maximum_cop: Optional[float]
    current_at_maximum_cop: Optional[float]
    heat_rate_at_maximum_cop: Optional[float]
    maximum_heat_rate: float
    current_at_maximum_heat_rate: float
    cop_at_maximum_heat_rate: Optional[float]


class EqualLoadContactComparison(NamedTuple):
    """Reduced and contact-aware points delivering the same requested heat."""

    mode: str
    external_temperature_lift: float
    symmetric_contact_resistance: float
    target_heat_rate: float
    feasible: bool
    reduced_current: Optional[float]
    contact_current: Optional[float]
    reduced_cop: Optional[float]
    contact_cop: Optional[float]
    contact_cop_penalty_percent: Optional[float]
    extra_face_lift: Optional[float]


class COPOperatingMapResult(NamedTuple):
    config: COPOperatingMapConfig
    points: Tuple[SteadyOperatingPoint, ...]
    optima: Tuple[OptimalOperatingSummary, ...]
    equal_load_comparisons: Tuple[EqualLoadContactComparison, ...]


def _reservoir_temperatures(
    mean_temperature: float,
    external_temperature_lift: float,
) -> Tuple[float, float]:
    half_lift = 0.5 * external_temperature_lift
    return mean_temperature - half_lift, mean_temperature + half_lift


def reduced_steady_operating_point(
    current: float,
    external_temperature_lift: float,
    *,
    mean_reservoir_temperature: float = 300.0,
) -> SteadyOperatingPoint:
    """Evaluate the reduced topology with no explicit interface nodes."""

    reference = constant_current_reference_experiment()
    cold_reservoir, hot_reservoir = _reservoir_temperatures(
        mean_reservoir_temperature, external_temperature_lift
    )
    state = two_node_steady_state(
        reference.thermoelectric_parameters,
        reference.thermal_parameters,
        current=current,
        cold_reservoir_temperature=cold_reservoir,
        hot_reservoir_temperature=hot_reservoir,
    )
    delivered_cooling = (
        reference.thermal_parameters.cold_reservoir_conductance
        * (cold_reservoir - state.cold)
    )
    delivered_heating = (
        reference.thermal_parameters.hot_reservoir_conductance
        * (state.hot - hot_reservoir)
    )
    power = electrical_power(
        reference.thermoelectric_parameters,
        current,
        state.hot,
        state.cold,
    )
    return SteadyOperatingPoint(
        topology="reduced_no_explicit_contact",
        symmetric_contact_resistance=None,
        external_temperature_lift=external_temperature_lift,
        current=current,
        cold_reservoir_temperature=cold_reservoir,
        hot_reservoir_temperature=hot_reservoir,
        cold_face_temperature=state.cold,
        hot_face_temperature=state.hot,
        cold_exchanger_temperature=state.cold,
        hot_exchanger_temperature=state.hot,
        face_temperature_lift=state.hot - state.cold,
        exchanger_temperature_lift=state.hot - state.cold,
        cold_contact_temperature_drop=0.0,
        hot_contact_temperature_drop=0.0,
        delivered_cooling_rate=delivered_cooling,
        delivered_heating_rate=delivered_heating,
        electrical_power=power,
        cooling_cop=(
            delivered_cooling / power
            if delivered_cooling > 0.0 and power > 0.0
            else None
        ),
        heating_cop=(
            delivered_heating / power
            if delivered_heating > 0.0 and power > 0.0
            else None
        ),
    )


def contact_steady_operating_point(
    current: float,
    external_temperature_lift: float,
    symmetric_contact_resistance: float,
    *,
    mean_reservoir_temperature: float = 300.0,
) -> SteadyOperatingPoint:
    """Evaluate the four-node topology at one symmetric contact resistance."""

    if (
        not math.isfinite(symmetric_contact_resistance)
        or symmetric_contact_resistance <= 0.0
    ):
        raise ValueError("contact resistance must be positive and finite")
    reference = constant_current_contact_reference_experiment()
    thermal = replace(
        reference.thermal_parameters,
        cold_contact_resistance=symmetric_contact_resistance,
        hot_contact_resistance=symmetric_contact_resistance,
    )
    cold_reservoir, hot_reservoir = _reservoir_temperatures(
        mean_reservoir_temperature, external_temperature_lift
    )
    state = four_node_contact_steady_state(
        reference.thermoelectric_parameters,
        thermal,
        current=current,
        cold_reservoir_temperature=cold_reservoir,
        hot_reservoir_temperature=hot_reservoir,
    )
    delivered_cooling = thermal.cold_reservoir_conductance * (
        cold_reservoir - state.cold_exchanger
    )
    delivered_heating = thermal.hot_reservoir_conductance * (
        state.hot_exchanger - hot_reservoir
    )
    module_cooling = cold_side_heat(
        reference.thermoelectric_parameters,
        current,
        state.hot_face,
        state.cold_face,
    )
    module_heating = hot_side_heat(
        reference.thermoelectric_parameters,
        current,
        state.hot_face,
        state.cold_face,
    )
    if not math.isclose(
        delivered_cooling, module_cooling, rel_tol=1e-10, abs_tol=1e-10
    ) or not math.isclose(
        delivered_heating, module_heating, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise RuntimeError("steady contact and module heat rates do not agree")
    power = electrical_power(
        reference.thermoelectric_parameters,
        current,
        state.hot_face,
        state.cold_face,
    )
    return SteadyOperatingPoint(
        topology="four_node_contact",
        symmetric_contact_resistance=symmetric_contact_resistance,
        external_temperature_lift=external_temperature_lift,
        current=current,
        cold_reservoir_temperature=cold_reservoir,
        hot_reservoir_temperature=hot_reservoir,
        cold_face_temperature=state.cold_face,
        hot_face_temperature=state.hot_face,
        cold_exchanger_temperature=state.cold_exchanger,
        hot_exchanger_temperature=state.hot_exchanger,
        face_temperature_lift=state.hot_face - state.cold_face,
        exchanger_temperature_lift=(
            state.hot_exchanger - state.cold_exchanger
        ),
        cold_contact_temperature_drop=(
            state.cold_exchanger - state.cold_face
        ),
        hot_contact_temperature_drop=(
            state.hot_face - state.hot_exchanger
        ),
        delivered_cooling_rate=delivered_cooling,
        delivered_heating_rate=delivered_heating,
        electrical_power=power,
        cooling_cop=(
            delivered_cooling / power
            if delivered_cooling > 0.0 and power > 0.0
            else None
        ),
        heating_cop=(
            delivered_heating / power
            if delivered_heating > 0.0 and power > 0.0
            else None
        ),
    )


def _optimal_summary(
    points: Sequence[SteadyOperatingPoint],
    *,
    mode: str,
    minimum_useful_heat_rate: float,
) -> OptimalOperatingSummary:
    if mode not in {"cooling", "heating"}:
        raise ValueError("mode must be cooling or heating")
    heat_field = (
        "delivered_cooling_rate" if mode == "cooling" else "delivered_heating_rate"
    )
    cop_field = "cooling_cop" if mode == "cooling" else "heating_cop"
    maximum_heat_point = max(points, key=lambda point: getattr(point, heat_field))
    useful = tuple(
        point
        for point in points
        if getattr(point, heat_field) >= minimum_useful_heat_rate
        and getattr(point, cop_field) is not None
    )
    maximum_cop_point = (
        max(useful, key=lambda point: getattr(point, cop_field))
        if useful
        else None
    )
    first = points[0]
    return OptimalOperatingSummary(
        topology=first.topology,
        symmetric_contact_resistance=first.symmetric_contact_resistance,
        external_temperature_lift=first.external_temperature_lift,
        mode=mode,
        maximum_cop=(
            getattr(maximum_cop_point, cop_field)
            if maximum_cop_point is not None
            else None
        ),
        current_at_maximum_cop=(
            maximum_cop_point.current if maximum_cop_point is not None else None
        ),
        heat_rate_at_maximum_cop=(
            getattr(maximum_cop_point, heat_field)
            if maximum_cop_point is not None
            else None
        ),
        maximum_heat_rate=getattr(maximum_heat_point, heat_field),
        current_at_maximum_heat_rate=maximum_heat_point.current,
        cop_at_maximum_heat_rate=getattr(maximum_heat_point, cop_field),
    )


def _point_for_topology(
    topology: str,
    current: float,
    lift: float,
    resistance: Optional[float],
    config: COPOperatingMapConfig,
) -> SteadyOperatingPoint:
    if topology == "reduced_no_explicit_contact":
        return reduced_steady_operating_point(
            current,
            lift,
            mean_reservoir_temperature=config.mean_reservoir_temperature,
        )
    if resistance is None:
        raise ValueError("contact topology requires a resistance")
    return contact_steady_operating_point(
        current,
        lift,
        resistance,
        mean_reservoir_temperature=config.mean_reservoir_temperature,
    )


def _match_heat_rate(
    topology: str,
    lift: float,
    resistance: Optional[float],
    target: float,
    mode: str,
    config: COPOperatingMapConfig,
) -> Optional[SteadyOperatingPoint]:
    heat_field = (
        "delivered_cooling_rate" if mode == "cooling" else "delivered_heating_rate"
    )
    evaluated = {}

    def point_at(current: float) -> SteadyOperatingPoint:
        if current not in evaluated:
            evaluated[current] = _point_for_topology(
                topology,
                current,
                lift,
                resistance,
                config,
            )
        return evaluated[current]

    upper_point = point_at(config.maximum_current)
    if getattr(upper_point, heat_field) >= target:
        bracket = (0.0, config.maximum_current)
    else:
        bracket = first_rising_crossing_bracket(
            lambda current: getattr(point_at(current), heat_field),
            target=target,
            maximum_input=config.maximum_current,
            subdivisions=config.heat_rate_bracket_subdivisions,
        )
    if bracket is None:
        return None
    lower, upper = bracket
    if lower == upper:
        return point_at(upper)
    while upper - lower > config.current_tolerance:
        candidate = 0.5 * (lower + upper)
        point = point_at(candidate)
        if getattr(point, heat_field) < target:
            lower = candidate
        else:
            upper = candidate
    return point_at(0.5 * (lower + upper))


def _equal_load_comparison(
    lift: float,
    resistance: float,
    mode: str,
    target: float,
    config: COPOperatingMapConfig,
) -> EqualLoadContactComparison:
    reduced = _match_heat_rate(
        "reduced_no_explicit_contact",
        lift,
        None,
        target,
        mode,
        config,
    )
    contact = _match_heat_rate(
        "four_node_contact",
        lift,
        resistance,
        target,
        mode,
        config,
    )
    if reduced is None or contact is None:
        return EqualLoadContactComparison(
            mode,
            lift,
            resistance,
            target,
            False,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    reduced_cop = reduced.cooling_cop if mode == "cooling" else reduced.heating_cop
    contact_cop = contact.cooling_cop if mode == "cooling" else contact.heating_cop
    if reduced_cop is None or contact_cop is None:
        raise RuntimeError("matched positive heat must have defined COP")
    return EqualLoadContactComparison(
        mode=mode,
        external_temperature_lift=lift,
        symmetric_contact_resistance=resistance,
        target_heat_rate=target,
        feasible=True,
        reduced_current=reduced.current,
        contact_current=contact.current,
        reduced_cop=reduced_cop,
        contact_cop=contact_cop,
        contact_cop_penalty_percent=100.0 * (contact_cop / reduced_cop - 1.0),
        extra_face_lift=contact.face_temperature_lift - reduced.face_temperature_lift,
    )


def run_cop_operating_map(
    config: COPOperatingMapConfig = COPOperatingMapConfig(),
) -> COPOperatingMapResult:
    """Evaluate the complete generic steady operating envelope."""

    points = []
    optima = []
    comparisons = []
    topologies = (("reduced_no_explicit_contact", None),) + tuple(
        ("four_node_contact", resistance)
        for resistance in config.symmetric_contact_resistances
    )
    for lift in config.external_temperature_lifts:
        for topology, resistance in topologies:
            group = tuple(
                _point_for_topology(topology, current, lift, resistance, config)
                for current in config.currents
            )
            points.extend(group)
            optima.extend(
                (
                    _optimal_summary(
                        group,
                        mode="cooling",
                        minimum_useful_heat_rate=config.minimum_useful_heat_rate,
                    ),
                    _optimal_summary(
                        group,
                        mode="heating",
                        minimum_useful_heat_rate=config.minimum_useful_heat_rate,
                    ),
                )
            )
        for resistance in config.symmetric_contact_resistances:
            comparisons.extend(
                (
                    _equal_load_comparison(
                        lift,
                        resistance,
                        "cooling",
                        config.equal_cooling_target,
                        config,
                    ),
                    _equal_load_comparison(
                        lift,
                        resistance,
                        "heating",
                        config.equal_heating_target,
                        config,
                    ),
                )
            )
    return COPOperatingMapResult(
        config=config,
        points=tuple(points),
        optima=tuple(optima),
        equal_load_comparisons=tuple(comparisons),
    )


def points_for(
    result: COPOperatingMapResult,
    *,
    topology: str,
    external_temperature_lift: float,
    symmetric_contact_resistance: Optional[float] = None,
) -> Tuple[SteadyOperatingPoint, ...]:
    return tuple(
        point
        for point in result.points
        if point.topology == topology
        and point.external_temperature_lift == external_temperature_lift
        and point.symmetric_contact_resistance == symmetric_contact_resistance
    )


def format_cop_operating_map_report(result: COPOperatingMapResult) -> str:
    """Format key baseline-contact operating conclusions."""

    baseline_resistance = min(
        result.config.symmetric_contact_resistances,
        key=lambda value: abs(value - 0.25),
    )
    lines = [
        "ThermoTwin steady COP operating map",
        "delta-T definition: hot reservoir minus cold reservoir",
        (
            f"grid: {len(result.config.external_temperature_lifts)} lifts x "
            f"{len(result.config.currents)} currents x "
            f"{1 + len(result.config.symmetric_contact_resistances)} topologies"
        ),
        (
            f"reported maximum COP requires at least "
            f"{result.config.minimum_useful_heat_rate:.1f} W useful heat"
        ),
        f"baseline symmetric contact resistance: {baseline_resistance:.2f} K/W",
    ]
    for lift in result.config.external_temperature_lifts:
        for mode in ("cooling", "heating"):
            summary = next(
                item
                for item in result.optima
                if item.topology == "four_node_contact"
                and item.symmetric_contact_resistance == baseline_resistance
                and item.external_temperature_lift == lift
                and item.mode == mode
            )
            if summary.maximum_cop is None:
                lines.append(f"  lift {lift:.0f} K {mode}: no >=1 W point")
            else:
                lines.append(
                    f"  lift {lift:.0f} K {mode}: max COP={summary.maximum_cop:.3f} "
                    f"at I={summary.current_at_maximum_cop:.2f} A, "
                    f"heat={summary.heat_rate_at_maximum_cop:.2f} W"
                )
    lines.append("equal-load baseline contact penalties:")
    for item in result.equal_load_comparisons:
        if item.symmetric_contact_resistance != baseline_resistance:
            continue
        if not item.feasible:
            lines.append(
                f"  lift {item.external_temperature_lift:.0f} K {item.mode}: "
                f"target {item.target_heat_rate:.1f} W infeasible"
            )
        else:
            lines.append(
                f"  lift {item.external_temperature_lift:.0f} K {item.mode}: "
                f"COP penalty={item.contact_cop_penalty_percent:.2f}%, "
                f"extra face lift={item.extra_face_lift:.2f} K"
            )
    return "\n".join(lines)


def main() -> None:
    print(format_cop_operating_map_report(run_cop_operating_map()))


if __name__ == "__main__":
    main()
