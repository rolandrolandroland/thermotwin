"""Self-contained provenance for virtual ThermoTwin observations.

The observation tables intentionally do not expose dense RK4 trajectories.
This module records the inputs needed to reproduce those trajectories and the
measurement-processing steps applied afterward.
"""

from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple, Union

from ..simulation.four_node_experiments import FourNodeContactExperiment
from ..physics.four_node import FourNodeContactThermalParameters
from ..core.controls import CurrentInput, PiecewiseConstantCurrent
from ..physics.thermoelectric import ThermoelectricParameters


MetadataValue = Union[str, int, float]


@dataclass(frozen=True)
class MetadataSetting:
    """One named, scalar configuration value stored with a dataset."""

    name: str
    value: MetadataValue

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("metadata setting name must be nonempty")
        object.__setattr__(self, "name", self.name.strip())
        if isinstance(self.value, bool) or not isinstance(
            self.value,
            (str, int, float),
        ):
            raise ValueError("metadata setting value must be scalar")
        if isinstance(self.value, str):
            if not self.value.strip():
                raise ValueError("string metadata values must be nonempty")
            object.__setattr__(self, "value", self.value.strip())
        elif not math.isfinite(self.value):
            raise ValueError("numeric metadata values must be finite")


@dataclass(frozen=True)
class ObservationProcessStep:
    """One ordered transformation in the synthetic measurement pipeline."""

    name: str
    settings: Tuple[MetadataSetting, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("observation-process step name must be nonempty")
        object.__setattr__(self, "name", self.name.strip())
        try:
            settings = tuple(self.settings)
        except TypeError as error:
            raise ValueError(
                "observation-process settings must be metadata settings"
            ) from error
        if not all(isinstance(item, MetadataSetting) for item in settings):
            raise ValueError(
                "observation-process settings must be metadata settings"
            )
        names = tuple(item.name for item in settings)
        if len(set(names)) != len(names):
            raise ValueError(
                "observation-process setting names must be unique"
            )
        object.__setattr__(self, "settings", settings)


@dataclass(frozen=True)
class CurrentScheduleMetadata:
    """Serializable description of a scalar or piecewise current input."""

    kind: str
    transition_times: Tuple[float, ...]
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"scalar", "piecewise_constant"}:
            raise ValueError(
                "current schedule kind must be 'scalar' or "
                "'piecewise_constant'"
            )
        transition_times = tuple(float(value) for value in self.transition_times)
        values = tuple(float(value) for value in self.values)
        PiecewiseConstantCurrent(transition_times, values)
        if self.kind == "scalar" and (transition_times or len(values) != 1):
            raise ValueError("scalar current metadata must contain one value")
        object.__setattr__(self, "transition_times", transition_times)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_current(
        cls,
        current: CurrentInput,
    ) -> "CurrentScheduleMetadata":
        """Capture one supported current input without changing its kind."""

        if isinstance(current, PiecewiseConstantCurrent):
            return cls(
                kind="piecewise_constant",
                transition_times=current.transition_times,
                values=current.values,
            )
        value = float(current)
        if not math.isfinite(value):
            raise ValueError("current must be finite")
        return cls(kind="scalar", transition_times=(), values=(value,))

    def to_current(self) -> CurrentInput:
        """Reconstruct the recorded current input."""

        if self.kind == "scalar":
            return self.values[0]
        return PiecewiseConstantCurrent(
            transition_times=self.transition_times,
            values=self.values,
        )


@dataclass(frozen=True)
class ContactExperimentMetadata:
    """Complete reproducible configuration and synthetic physical truth."""

    experiment_name: str
    regime_name: str
    split: str
    thermoelectric_parameters: ThermoelectricParameters
    thermal_parameters: FourNodeContactThermalParameters
    initial_cold_face_temperature: float
    initial_hot_face_temperature: float
    initial_cold_exchanger_temperature: float
    initial_hot_exchanger_temperature: float
    duration: float
    integration_time_step: float
    current_schedule: CurrentScheduleMetadata
    cold_reservoir_temperature: float
    hot_reservoir_temperature: float
    cold_external_heat: float
    hot_external_heat: float

    def __post_init__(self) -> None:
        for field_name in ("experiment_name", "regime_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be nonempty")
            object.__setattr__(self, field_name, value.strip())
        if self.split not in {"unsplit", "train", "validation", "test"}:
            raise ValueError(
                "metadata split must be unsplit, train, validation, or test"
            )
        if not isinstance(
            self.thermoelectric_parameters,
            ThermoelectricParameters,
        ):
            raise ValueError("thermoelectric parameter truth is required")
        if not isinstance(
            self.thermal_parameters,
            FourNodeContactThermalParameters,
        ):
            raise ValueError("thermal parameter truth is required")
        if not isinstance(self.current_schedule, CurrentScheduleMetadata):
            raise ValueError("current schedule metadata is required")

        finite_values = (
            self.initial_cold_face_temperature,
            self.initial_hot_face_temperature,
            self.initial_cold_exchanger_temperature,
            self.initial_hot_exchanger_temperature,
            self.cold_reservoir_temperature,
            self.hot_reservoir_temperature,
            self.cold_external_heat,
            self.hot_external_heat,
        )
        if any(not math.isfinite(value) for value in finite_values):
            raise ValueError(
                "metadata temperatures and heat inputs must be finite"
            )
        if not math.isfinite(self.duration) or self.duration < 0.0:
            raise ValueError("metadata duration must be finite and nonnegative")
        if (
            not math.isfinite(self.integration_time_step)
            or self.integration_time_step <= 0.0
        ):
            raise ValueError(
                "metadata integration time step must be finite and positive"
            )

    @classmethod
    def from_experiment(
        cls,
        experiment: FourNodeContactExperiment,
        *,
        experiment_name: str,
        regime_name: str,
        split: str = "unsplit",
    ) -> "ContactExperimentMetadata":
        """Capture every input needed to rebuild a contact experiment."""

        return cls(
            experiment_name=experiment_name,
            regime_name=regime_name,
            split=split,
            thermoelectric_parameters=experiment.thermoelectric_parameters,
            thermal_parameters=experiment.thermal_parameters,
            initial_cold_face_temperature=(
                experiment.initial_cold_face_temperature
            ),
            initial_hot_face_temperature=(
                experiment.initial_hot_face_temperature
            ),
            initial_cold_exchanger_temperature=(
                experiment.initial_cold_exchanger_temperature
            ),
            initial_hot_exchanger_temperature=(
                experiment.initial_hot_exchanger_temperature
            ),
            duration=experiment.duration,
            integration_time_step=experiment.time_step,
            current_schedule=CurrentScheduleMetadata.from_current(
                experiment.current
            ),
            cold_reservoir_temperature=experiment.cold_reservoir_temperature,
            hot_reservoir_temperature=experiment.hot_reservoir_temperature,
            cold_external_heat=experiment.cold_external_heat,
            hot_external_heat=experiment.hot_external_heat,
        )

    def to_experiment(self) -> FourNodeContactExperiment:
        """Rebuild the recorded physical experiment exactly."""

        return FourNodeContactExperiment(
            thermoelectric_parameters=self.thermoelectric_parameters,
            thermal_parameters=self.thermal_parameters,
            initial_cold_face_temperature=(
                self.initial_cold_face_temperature
            ),
            initial_hot_face_temperature=self.initial_hot_face_temperature,
            initial_cold_exchanger_temperature=(
                self.initial_cold_exchanger_temperature
            ),
            initial_hot_exchanger_temperature=(
                self.initial_hot_exchanger_temperature
            ),
            duration=self.duration,
            time_step=self.integration_time_step,
            current=self.current_schedule.to_current(),
            cold_reservoir_temperature=self.cold_reservoir_temperature,
            hot_reservoir_temperature=self.hot_reservoir_temperature,
            cold_external_heat=self.cold_external_heat,
            hot_external_heat=self.hot_external_heat,
        )


@dataclass(frozen=True)
class DatasetProvenance:
    """Physical truth plus the ordered observation-processing history."""

    experiment: ContactExperimentMetadata
    observation_steps: Tuple[ObservationProcessStep, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ContactExperimentMetadata):
            raise ValueError("dataset provenance requires experiment metadata")
        try:
            steps = tuple(self.observation_steps)
        except TypeError as error:
            raise ValueError(
                "observation steps must be process-step objects"
            ) from error
        if not all(isinstance(step, ObservationProcessStep) for step in steps):
            raise ValueError("observation steps must be process-step objects")
        object.__setattr__(self, "observation_steps", steps)


def append_observation_process_step(
    provenance: Optional[DatasetProvenance],
    step: ObservationProcessStep,
) -> Optional[DatasetProvenance]:
    """Append provenance when present and preserve metadata-free test data."""

    if provenance is None:
        return None
    if not isinstance(step, ObservationProcessStep):
        raise ValueError("step must be an observation-process step")
    return replace(
        provenance,
        observation_steps=provenance.observation_steps + (step,),
    )
