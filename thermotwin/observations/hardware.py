"""Validated CSV bridge for future benchtop ThermoTwin measurements."""

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
from typing import IO, NamedTuple, Optional, Tuple, Union

from .test_stand import (
    IdealTemperatureSensor,
    ObservationDataset,
    TemperatureObservation,
    TemperatureSensorLocation,
)


REQUIRED_HARDWARE_COLUMNS = (
    "time_s",
    "current_A",
    "cold_exchanger_K",
    "hot_exchanger_K",
)


class HardwareVoltageObservation(NamedTuple):
    time: float
    voltage: float


class HardwareInputObservation(NamedTuple):
    time: float
    current: float


class HardwareDataset(NamedTuple):
    temperatures: ObservationDataset
    inputs: Tuple[HardwareInputObservation, ...]
    voltage: Tuple[HardwareVoltageObservation, ...]


@dataclass(frozen=True)
class HardwareDatasetSummary:
    """Compact ingestion checks without claiming physical validation."""

    duration: float
    row_count: int
    temperature_record_count: int
    cold_temperature_count: int
    hot_temperature_count: int
    voltage_count: int
    median_sampling_interval: float
    minimum_current: float
    maximum_current: float
    minimum_temperature: float
    maximum_temperature: float


def _finite_float(value: str, *, field_name: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number}: {field_name} must be numeric"
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number}: {field_name} must be finite")
    return parsed


def _optional_float(
    value: Optional[str],
    *,
    field_name: str,
    row_number: int,
) -> Optional[float]:
    if value is None or not value.strip():
        return None
    return _finite_float(value, field_name=field_name, row_number=row_number)


def read_hardware_csv(stream: IO[str]) -> HardwareDataset:
    """Parse required SI-unit columns and preserve blank temperature readings."""

    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        raise ValueError("hardware CSV must contain a header row")
    missing_columns = set(REQUIRED_HARDWARE_COLUMNS) - set(reader.fieldnames)
    if missing_columns:
        raise ValueError(
            "hardware CSV is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    sensors = (
        IdealTemperatureSensor(
            "cold_exchanger_sensor",
            TemperatureSensorLocation.COLD_EXCHANGER,
        ),
        IdealTemperatureSensor(
            "hot_exchanger_sensor",
            TemperatureSensorLocation.HOT_EXCHANGER,
        ),
    )
    observations = []
    input_observations = []
    voltage_observations = []
    row_times = []
    for row_number, row in enumerate(reader, start=2):
        time = _finite_float(
            row["time_s"], field_name="time_s", row_number=row_number
        )
        current = _finite_float(
            row["current_A"], field_name="current_A", row_number=row_number
        )
        if time < 0.0:
            raise ValueError(f"row {row_number}: time_s must be nonnegative")
        if row_times and time <= row_times[-1]:
            raise ValueError("hardware CSV times must strictly increase")
        row_times.append(time)
        input_observations.append(HardwareInputObservation(time, current))
        for sensor, column in zip(
            sensors, ("cold_exchanger_K", "hot_exchanger_K")
        ):
            temperature = _optional_float(
                row[column], field_name=column, row_number=row_number
            )
            if temperature is None:
                continue
            if temperature <= 0.0:
                raise ValueError(
                    f"row {row_number}: absolute temperature must be positive"
                )
            observations.append(
                TemperatureObservation(
                    time=time,
                    sensor_name=sensor.name,
                    location=sensor.location,
                    temperature=temperature,
                    current=current,
                )
            )
        if "voltage_V" in reader.fieldnames:
            voltage = _optional_float(
                row.get("voltage_V"),
                field_name="voltage_V",
                row_number=row_number,
            )
            if voltage is not None:
                voltage_observations.append(HardwareVoltageObservation(time, voltage))
    if len(row_times) < 2:
        raise ValueError("hardware CSV must contain at least two time rows")
    if not observations:
        raise ValueError("hardware CSV contains no temperature readings")
    sampling_interval = median(
        right - left for left, right in zip(row_times, row_times[1:])
    )
    return HardwareDataset(
        temperatures=ObservationDataset(
            observations=tuple(observations),
            sensors=sensors,
            sampling_interval=sampling_interval,
        ),
        inputs=tuple(input_observations),
        voltage=tuple(voltage_observations),
    )


def load_hardware_csv(path: Union[str, Path]) -> HardwareDataset:
    """Load a hardware CSV without modifying it."""

    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8", newline="") as stream:
        return read_hardware_csv(stream)


def summarize_hardware_dataset(dataset: HardwareDataset) -> HardwareDatasetSummary:
    observations = dataset.temperatures.observations
    times = tuple(item.time for item in dataset.inputs)
    currents = tuple(item.current for item in dataset.inputs)
    temperatures = tuple(item.temperature for item in observations)
    cold_count = len(
        dataset.temperatures.observations_for("cold_exchanger_sensor")
    )
    hot_count = len(
        dataset.temperatures.observations_for("hot_exchanger_sensor")
    )
    return HardwareDatasetSummary(
        duration=times[-1] - times[0],
        row_count=len(times),
        temperature_record_count=len(observations),
        cold_temperature_count=cold_count,
        hot_temperature_count=hot_count,
        voltage_count=len(dataset.voltage),
        median_sampling_interval=dataset.temperatures.sampling_interval,
        minimum_current=min(currents),
        maximum_current=max(currents),
        minimum_temperature=min(temperatures),
        maximum_temperature=max(temperatures),
    )
