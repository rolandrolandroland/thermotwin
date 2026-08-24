"""Contact-resistance inference with readings missing near turn-off."""

import math
from typing import NamedTuple, Optional, Tuple

from ...inference.contact_resistance import (
    FITTED_SENSOR_NAMES,
    ContactResistanceRegimeDataset,
    reference_contact_resistance_dataset_split,
)
from .robustness import (
    ContactResistanceRobustnessCaseResult,
    format_contact_resistance_robustness_cases,
    map_contact_resistance_dataset_split,
    match_split_schema,
    run_contact_resistance_robustness_case,
)
from ...observations.missingness import (
    DeterministicTemperatureMissingness,
    TemperatureSensorOutage,
    apply_deterministic_temperature_missingness,
)


class ContactResistanceMissingnessCase(NamedTuple):
    """One named outage design for all three current regimes."""

    name: str
    turn_off_half_width: Optional[float]
    remove_pre_pulse_control: bool = False


class ContactResistanceMissingnessStudyResult(NamedTuple):
    """Frozen outage cases and their inference diagnostics."""

    cases: Tuple[ContactResistanceRobustnessCaseResult, ...]


def reference_contact_resistance_missingness_cases(
) -> Tuple[ContactResistanceMissingnessCase, ...]:
    """Return complete, control, instant, narrow, and wide outage cases."""

    return (
        ContactResistanceMissingnessCase(
            "complete_readings",
            None,
        ),
        ContactResistanceMissingnessCase(
            "pre_pulse_control_0_to_4s",
            None,
            remove_pre_pulse_control=True,
        ),
        ContactResistanceMissingnessCase(
            "turn_off_instants",
            0.0,
        ),
        ContactResistanceMissingnessCase(
            "turn_off_windows_plus_minus_2s",
            2.0,
        ),
        ContactResistanceMissingnessCase(
            "turn_off_windows_plus_minus_5s",
            5.0,
        ),
    )


def _nonzero_to_zero_transition_times(
    dataset: ContactResistanceRegimeDataset,
) -> Tuple[float, ...]:
    current = dataset.regime.current
    return tuple(
        transition_time
        for index, transition_time in enumerate(current.transition_times)
        if current.values[index] != 0.0
        and current.values[index + 1] == 0.0
    )


def missingness_for_contact_resistance_case(
    case: ContactResistanceMissingnessCase,
    dataset: ContactResistanceRegimeDataset,
) -> DeterministicTemperatureMissingness:
    """Build regime-aligned outages for both fitted cold sensors."""

    if not isinstance(case, ContactResistanceMissingnessCase):
        raise ValueError("case must be a contact-resistance outage case")
    if case.remove_pre_pulse_control:
        intervals = ((0.0, 4.0),)
    elif case.turn_off_half_width is None:
        intervals = ()
    else:
        half_width = case.turn_off_half_width
        try:
            half_width_is_finite = math.isfinite(half_width)
        except TypeError as error:
            raise ValueError(
                "turn-off half-width must be finite and nonnegative"
            ) from error
        if not half_width_is_finite or half_width < 0.0:
            raise ValueError(
                "turn-off half-width must be finite and nonnegative"
            )
        intervals = tuple(
            (
                max(0.0, time - half_width),
                time + half_width,
            )
            for time in _nonzero_to_zero_transition_times(dataset)
        )
    outages = tuple(
        TemperatureSensorOutage(
            sensor_name=sensor_name,
            start_time=start_time,
            end_time=end_time,
        )
        for sensor_name in FITTED_SENSOR_NAMES
        for start_time, end_time in intervals
    )
    return DeterministicTemperatureMissingness(outages=outages)


def run_contact_resistance_missingness_study(
) -> ContactResistanceMissingnessStudyResult:
    """Fit ideal data after progressively removing turn-off readings."""

    ideal = reference_contact_resistance_dataset_split()
    results = []
    for case in reference_contact_resistance_missingness_cases():
        incomplete = map_contact_resistance_dataset_split(
            ideal,
            lambda dataset, selected_case=case: (
                apply_deterministic_temperature_missingness(
                    dataset.observations,
                    missingness_for_contact_resistance_case(
                        selected_case,
                        dataset,
                    ),
                ).dataset
            ),
        )
        visible_truth = match_split_schema(ideal, incomplete)
        results.append(
            run_contact_resistance_robustness_case(
                case.name,
                incomplete,
                visible_truth,
                fitted_sensor_names=FITTED_SENSOR_NAMES,
            )
        )
    return ContactResistanceMissingnessStudyResult(cases=tuple(results))


def format_contact_resistance_missingness_study_report(
    result: ContactResistanceMissingnessStudyResult,
) -> str:
    """Format the turn-off missingness comparison."""

    return format_contact_resistance_robustness_cases(
        "cold contact resistance turn-off missingness study",
        result.cases,
    )


def main() -> None:
    """Run and print the dependency-free missingness study."""

    print(
        format_contact_resistance_missingness_study_report(
            run_contact_resistance_missingness_study()
        )
    )


if __name__ == "__main__":
    main()
