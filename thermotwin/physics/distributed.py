"""One-dimensional distributed thermoelectric-leg physics.

The spatial coordinate points from the cold face (``x = 0``) to the hot face
(``x = L``). Positive current density points in the same direction. The local
constitutive equations are

``E = rho_e(T) * J + alpha(T) * dT/dx``

``q = alpha(T) * T * J - kappa(T) * dT/dx``.

Consequently, local energy conservation can be written either in conservative
form, ``rho_m c_p dT/dt = -dq/dx + J E``, or in expanded form,

``rho_m c_p dT/dt = d/dx(kappa dT/dx) + rho_e J**2
                    - tau J dT/dx``

with the Kelvin relation ``tau = T d(alpha)/dT``.  The finite-volume model
uses the conservative form so its whole-system energy balance closes to
roundoff even when properties vary with temperature.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class TemperatureProperty(Protocol):
    """A scalar material property that can be evaluated and integrated in T."""

    def value(self, temperature: float) -> float:
        """Return the property value at an absolute temperature."""

    def integral(self, start_temperature: float, end_temperature: float) -> float:
        """Return the oriented integral of the property with respect to T."""


@dataclass(frozen=True)
class ConstantProperty:
    """A temperature-independent scalar property."""

    constant: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.constant):
            raise ValueError("constant property value must be finite")

    def value(self, temperature: float) -> float:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("property temperature must be positive kelvin")
        return self.constant

    def integral(self, start_temperature: float, end_temperature: float) -> float:
        self.value(start_temperature)
        self.value(end_temperature)
        return self.constant * (end_temperature - start_temperature)


@dataclass(frozen=True)
class PiecewiseLinearProperty:
    """A piecewise-linear property with constant endpoint extrapolation.

    Constant extrapolation is deliberate: it keeps a transient integration
    numerically defined if it moves only slightly outside a fitted temperature
    interval. Reports must still disclose the temperature interval actually
    explored; extrapolated values are not new material evidence.
    """

    temperatures: Tuple[float, ...]
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        temperatures = tuple(float(value) for value in self.temperatures)
        values = tuple(float(value) for value in self.values)
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "values", values)
        if len(temperatures) < 2 or len(temperatures) != len(values):
            raise ValueError(
                "piecewise property temperatures and values must have equal "
                "length of at least two"
            )
        if any(not math.isfinite(value) for value in temperatures + values):
            raise ValueError("piecewise property entries must be finite")
        if any(value <= 0.0 for value in temperatures):
            raise ValueError("property temperatures must be positive kelvin")
        if any(
            right <= left
            for left, right in zip(temperatures, temperatures[1:])
        ):
            raise ValueError("property temperatures must strictly increase")

    def value(self, temperature: float) -> float:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("property temperature must be positive kelvin")
        if temperature <= self.temperatures[0]:
            return self.values[0]
        if temperature >= self.temperatures[-1]:
            return self.values[-1]
        for left_index, (left, right) in enumerate(
            zip(self.temperatures, self.temperatures[1:])
        ):
            if temperature <= right:
                fraction = (temperature - left) / (right - left)
                return self.values[left_index] + fraction * (
                    self.values[left_index + 1] - self.values[left_index]
                )
        raise RuntimeError("property interpolation interval was not found")

    def integral(self, start_temperature: float, end_temperature: float) -> float:
        if end_temperature == start_temperature:
            self.value(start_temperature)
            return 0.0
        if end_temperature < start_temperature:
            return -self.integral(end_temperature, start_temperature)
        self.value(start_temperature)
        self.value(end_temperature)
        breakpoints = [start_temperature]
        breakpoints.extend(
            value
            for value in self.temperatures
            if start_temperature < value < end_temperature
        )
        breakpoints.append(end_temperature)
        return sum(
            0.5 * (self.value(left) + self.value(right)) * (right - left)
            for left, right in zip(breakpoints, breakpoints[1:])
        )

    def with_values(self, values: Sequence[float]) -> "PiecewiseLinearProperty":
        """Return the same temperature basis with replacement coefficients."""

        return PiecewiseLinearProperty(self.temperatures, tuple(values))


@dataclass(frozen=True)
class DistributedThermoelectricMaterial:
    """Temperature-dependent properties for one oriented thermoelectric leg.

    ``seebeck_coefficient`` may be signed. ``electrical_resistivity`` and
    ``thermal_conductivity`` must remain positive over every simulated state.
    Mass density and specific heat are held constant in the first distributed
    model so the inverse problem can focus on the three transport functions.
    """

    seebeck_coefficient: TemperatureProperty
    electrical_resistivity: TemperatureProperty
    thermal_conductivity: TemperatureProperty
    mass_density: float
    specific_heat_capacity: float

    def __post_init__(self) -> None:
        for name, prop in (
            ("Seebeck coefficient", self.seebeck_coefficient),
            ("electrical resistivity", self.electrical_resistivity),
            ("thermal conductivity", self.thermal_conductivity),
        ):
            if not isinstance(prop, TemperatureProperty):
                raise TypeError(f"{name} must implement value() and integral()")
        for name, value in (
            ("mass density", self.mass_density),
            ("specific heat capacity", self.specific_heat_capacity),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DistributedLegGeometry:
    """Length in metres and cross-sectional area in square metres."""

    length: float
    area: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("leg length must be finite and positive")
        if not math.isfinite(self.area) or self.area <= 0.0:
            raise ValueError("leg area must be finite and positive")


@dataclass(frozen=True)
class DistributedFaceThermalParameters:
    """Thermal masses and reservoir links attached to the two leg faces."""

    cold_thermal_capacitance: float
    hot_thermal_capacitance: float
    cold_reservoir_conductance: float
    hot_reservoir_conductance: float

    def __post_init__(self) -> None:
        for name, value in (
            ("cold thermal capacitance", self.cold_thermal_capacitance),
            ("hot thermal capacitance", self.hot_thermal_capacitance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("cold reservoir conductance", self.cold_reservoir_conductance),
            ("hot reservoir conductance", self.hot_reservoir_conductance),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


class DistributedTemperatureRates(NamedTuple):
    """Face and cell temperature rates, all in K/s."""

    cold_face: float
    cells: Tuple[float, ...]
    hot_face: float


class DistributedStateDiagnostics(NamedTuple):
    """Instantaneous electrical, thermal, and energy-balance quantities."""

    voltage: float
    electrical_power: float
    cold_side_heat: float
    hot_side_heat: float
    cold_reservoir_heat: float
    hot_reservoir_heat: float
    stored_energy_rate: float
    expected_energy_rate: float
    energy_balance_residual: float


def _validated_temperatures(
    cold_face_temperature: float,
    cell_temperatures: Sequence[float],
    hot_face_temperature: float,
    *,
    cell_count: int | None = None,
) -> Tuple[float, ...]:
    cells = tuple(float(value) for value in cell_temperatures)
    if len(cells) < 2:
        raise ValueError("the distributed leg requires at least two cells")
    if cell_count is not None and len(cells) != cell_count:
        raise ValueError("cell-temperature count does not match cell count")
    values = (float(cold_face_temperature),) + cells + (
        float(hot_face_temperature),
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("distributed temperatures must be finite")
    if any(value <= 0.0 for value in values):
        raise ValueError("distributed temperatures must be positive kelvin")
    return values


def linear_cell_temperatures(
    cold_face_temperature: float,
    hot_face_temperature: float,
    cell_count: int,
) -> Tuple[float, ...]:
    """Return a linear initial profile at finite-volume cell centres."""

    if not isinstance(cell_count, int) or isinstance(cell_count, bool) or cell_count < 2:
        raise ValueError("cell count must be an integer of at least two")
    _validated_temperatures(
        cold_face_temperature,
        (cold_face_temperature, hot_face_temperature),
        hot_face_temperature,
    )
    return tuple(
        cold_face_temperature
        + (hot_face_temperature - cold_face_temperature)
        * ((index + 0.5) / cell_count)
        for index in range(cell_count)
    )


def _face_temperatures(
    cold_face_temperature: float,
    cell_temperatures: Tuple[float, ...],
    hot_face_temperature: float,
) -> Tuple[float, ...]:
    return (
        cold_face_temperature,
        *(
            0.5 * (left + right)
            for left, right in zip(cell_temperatures, cell_temperatures[1:])
        ),
        hot_face_temperature,
    )


def _positive_transport_properties(
    material: DistributedThermoelectricMaterial,
    temperatures: Sequence[float],
) -> None:
    for temperature in temperatures:
        resistivity = material.electrical_resistivity.value(temperature)
        conductivity = material.thermal_conductivity.value(temperature)
        if not math.isfinite(resistivity) or resistivity <= 0.0:
            raise ValueError(
                "electrical resistivity must remain finite and positive"
            )
        if not math.isfinite(conductivity) or conductivity <= 0.0:
            raise ValueError(
                "thermal conductivity must remain finite and positive"
            )


def distributed_leg_fluxes_and_voltage_drops(
    material: DistributedThermoelectricMaterial,
    geometry: DistributedLegGeometry,
    *,
    cold_face_temperature: float,
    cell_temperatures: Sequence[float],
    hot_face_temperature: float,
    current: float,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return heat fluxes (W/m²) and cell voltage drops (V).

    Heat flux is positive from cold to hot. The boundary fluxes multiplied by
    area are therefore exactly the package's ``Q_c`` and ``Q_h`` signs.
    Cell voltage drops include both resistive and Seebeck contributions.
    """

    temperatures = _validated_temperatures(
        cold_face_temperature,
        cell_temperatures,
        hot_face_temperature,
    )
    if not math.isfinite(current):
        raise ValueError("current must be finite")
    cells = temperatures[1:-1]
    _positive_transport_properties(material, temperatures)
    cell_count = len(cells)
    spacing = geometry.length / cell_count
    current_density = current / geometry.area
    faces = _face_temperatures(temperatures[0], cells, temperatures[-1])

    cold_gradient = (-8.0 * temperatures[0] + 9.0 * cells[0] - cells[1]) / (
        3.0 * spacing
    )
    hot_gradient = (8.0 * temperatures[-1] - 9.0 * cells[-1] + cells[-2]) / (
        3.0 * spacing
    )
    gradients = [cold_gradient]
    gradients.extend(
        (right - left) / spacing
        for left, right in zip(cells, cells[1:])
    )
    gradients.append(hot_gradient)

    heat_fluxes = tuple(
        material.seebeck_coefficient.value(face_temperature)
        * face_temperature
        * current_density
        - material.thermal_conductivity.value(face_temperature) * gradient
        for face_temperature, gradient in zip(faces, gradients)
    )
    voltage_drops = tuple(
        material.electrical_resistivity.value(cell_temperature)
        * current_density
        * spacing
        + material.seebeck_coefficient.integral(left_face, right_face)
        for cell_temperature, left_face, right_face in zip(
            cells, faces, faces[1:]
        )
    )
    if any(
        not math.isfinite(value)
        for value in heat_fluxes + voltage_drops
    ):
        raise ValueError("distributed flux or voltage became nonfinite")
    return heat_fluxes, voltage_drops


def distributed_leg_rhs(
    material: DistributedThermoelectricMaterial,
    geometry: DistributedLegGeometry,
    face_parameters: DistributedFaceThermalParameters,
    *,
    cold_face_temperature: float,
    cell_temperatures: Sequence[float],
    hot_face_temperature: float,
    current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> DistributedTemperatureRates:
    """Return the coupled face-node and finite-volume temperature rates."""

    temperatures = _validated_temperatures(
        cold_face_temperature,
        cell_temperatures,
        hot_face_temperature,
    )
    for name, value in (
        ("cold reservoir temperature", cold_reservoir_temperature),
        ("hot reservoir temperature", hot_reservoir_temperature),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive kelvin")
    for name, value in (
        ("cold external heat", cold_external_heat),
        ("hot external heat", hot_external_heat),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    cells = temperatures[1:-1]
    heat_fluxes, voltage_drops = distributed_leg_fluxes_and_voltage_drops(
        material,
        geometry,
        cold_face_temperature=temperatures[0],
        cell_temperatures=cells,
        hot_face_temperature=temperatures[-1],
        current=current,
    )
    spacing = geometry.length / len(cells)
    volumetric_heat_capacity = material.mass_density * material.specific_heat_capacity
    cell_rates = tuple(
        (
            (left_flux - right_flux) / spacing
            + (current / geometry.area) * voltage_drop / spacing
        )
        / volumetric_heat_capacity
        for left_flux, right_flux, voltage_drop in zip(
            heat_fluxes, heat_fluxes[1:], voltage_drops
        )
    )
    cold_side_heat = geometry.area * heat_fluxes[0]
    hot_side_heat = geometry.area * heat_fluxes[-1]
    cold_reservoir_heat = face_parameters.cold_reservoir_conductance * (
        cold_reservoir_temperature - temperatures[0]
    )
    hot_reservoir_heat = face_parameters.hot_reservoir_conductance * (
        hot_reservoir_temperature - temperatures[-1]
    )
    return DistributedTemperatureRates(
        cold_face=(
            cold_reservoir_heat + cold_external_heat - cold_side_heat
        )
        / face_parameters.cold_thermal_capacitance,
        cells=cell_rates,
        hot_face=(
            hot_reservoir_heat + hot_external_heat + hot_side_heat
        )
        / face_parameters.hot_thermal_capacitance,
    )


def evaluate_distributed_state(
    material: DistributedThermoelectricMaterial,
    geometry: DistributedLegGeometry,
    face_parameters: DistributedFaceThermalParameters,
    *,
    cold_face_temperature: float,
    cell_temperatures: Sequence[float],
    hot_face_temperature: float,
    current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> DistributedStateDiagnostics:
    """Evaluate voltage, heat rates, and exact semidiscrete energy closure."""

    rates = distributed_leg_rhs(
        material,
        geometry,
        face_parameters,
        cold_face_temperature=cold_face_temperature,
        cell_temperatures=cell_temperatures,
        hot_face_temperature=hot_face_temperature,
        current=current,
        cold_reservoir_temperature=cold_reservoir_temperature,
        hot_reservoir_temperature=hot_reservoir_temperature,
        cold_external_heat=cold_external_heat,
        hot_external_heat=hot_external_heat,
    )
    heat_fluxes, voltage_drops = distributed_leg_fluxes_and_voltage_drops(
        material,
        geometry,
        cold_face_temperature=cold_face_temperature,
        cell_temperatures=cell_temperatures,
        hot_face_temperature=hot_face_temperature,
        current=current,
    )
    voltage = sum(voltage_drops)
    power = current * voltage
    cold_heat = geometry.area * heat_fluxes[0]
    hot_heat = geometry.area * heat_fluxes[-1]
    cold_reservoir_heat = face_parameters.cold_reservoir_conductance * (
        cold_reservoir_temperature - cold_face_temperature
    )
    hot_reservoir_heat = face_parameters.hot_reservoir_conductance * (
        hot_reservoir_temperature - hot_face_temperature
    )
    cell_capacitance = (
        material.mass_density
        * material.specific_heat_capacity
        * geometry.area
        * geometry.length
        / len(rates.cells)
    )
    stored_energy_rate = (
        face_parameters.cold_thermal_capacitance * rates.cold_face
        + cell_capacitance * sum(rates.cells)
        + face_parameters.hot_thermal_capacitance * rates.hot_face
    )
    expected_energy_rate = (
        cold_reservoir_heat
        + hot_reservoir_heat
        + cold_external_heat
        + hot_external_heat
        + power
    )
    return DistributedStateDiagnostics(
        voltage=voltage,
        electrical_power=power,
        cold_side_heat=cold_heat,
        hot_side_heat=hot_heat,
        cold_reservoir_heat=cold_reservoir_heat,
        hot_reservoir_heat=hot_reservoir_heat,
        stored_energy_rate=stored_energy_rate,
        expected_energy_rate=expected_energy_rate,
        energy_balance_residual=stored_energy_rate - expected_energy_rate,
    )


def distributed_stored_energy(
    material: DistributedThermoelectricMaterial,
    geometry: DistributedLegGeometry,
    face_parameters: DistributedFaceThermalParameters,
    *,
    cold_face_temperature: float,
    cell_temperatures: Sequence[float],
    hot_face_temperature: float,
) -> float:
    """Return sensible-energy content relative to zero kelvin, in joules."""

    temperatures = _validated_temperatures(
        cold_face_temperature,
        cell_temperatures,
        hot_face_temperature,
    )
    cell_capacitance = (
        material.mass_density
        * material.specific_heat_capacity
        * geometry.area
        * geometry.length
        / len(cell_temperatures)
    )
    return (
        face_parameters.cold_thermal_capacitance * temperatures[0]
        + cell_capacitance * sum(temperatures[1:-1])
        + face_parameters.hot_thermal_capacitance * temperatures[-1]
    )


def recommended_explicit_time_step(
    material: DistributedThermoelectricMaterial,
    geometry: DistributedLegGeometry,
    *,
    cell_count: int,
    temperature_range: Tuple[float, float],
    safety_factor: float = 0.20,
) -> float:
    """Return a conservative diffusion-based RK4 time-step recommendation."""

    if not isinstance(cell_count, int) or isinstance(cell_count, bool) or cell_count < 2:
        raise ValueError("cell count must be an integer of at least two")
    if not math.isfinite(safety_factor) or not 0.0 < safety_factor <= 0.5:
        raise ValueError("safety factor must lie in (0, 0.5]")
    low, high = temperature_range
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high < low:
        raise ValueError("temperature range must be ordered positive kelvin")
    samples = tuple(low + (high - low) * index / 8.0 for index in range(9))
    diffusivity = max(
        material.thermal_conductivity.value(temperature)
        / (material.mass_density * material.specific_heat_capacity)
        for temperature in samples
    )
    if diffusivity <= 0.0 or not math.isfinite(diffusivity):
        raise ValueError("thermal diffusivity must be finite and positive")
    spacing = geometry.length / cell_count
    return safety_factor * spacing * spacing / diffusivity
