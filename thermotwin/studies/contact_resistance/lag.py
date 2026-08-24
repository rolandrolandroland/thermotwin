"""Contact-resistance inference under first-order temperature-sensor lag."""

from typing import NamedTuple, Tuple

from ...inference.contact_resistance import (
    FITTED_SENSOR_NAMES,
    reference_contact_resistance_dataset_split,
)
from .robustness import (
    ContactResistanceRobustnessCaseResult,
    format_contact_resistance_robustness_cases,
    lag_contact_resistance_dataset_split,
    run_contact_resistance_robustness_case,
)
from ...observations.lag import FirstOrderTemperatureLag


class ContactResistanceLagCase(NamedTuple):
    """One named first-order sensor-lag pattern."""

    name: str
    lag_model: FirstOrderTemperatureLag


class ContactResistanceLagStudyResult(NamedTuple):
    """Frozen lag patterns and their contact-resistance fits."""

    cases: Tuple[ContactResistanceRobustnessCaseResult, ...]


def reference_contact_resistance_lag_cases(
) -> Tuple[ContactResistanceLagCase, ...]:
    """Return zero, individual, common, and asymmetric cold lag cases."""

    return (
        ContactResistanceLagCase(
            "zero_lag",
            FirstOrderTemperatureLag(),
        ),
        ContactResistanceLagCase(
            "cold_face_tau_2s",
            FirstOrderTemperatureLag(
                sensor_time_constants=(("cold_face_sensor", 2.0),),
            ),
        ),
        ContactResistanceLagCase(
            "cold_exchanger_tau_2s",
            FirstOrderTemperatureLag(
                sensor_time_constants=(
                    ("cold_exchanger_sensor", 2.0),
                ),
            ),
        ),
        ContactResistanceLagCase(
            "cold_pair_common_tau_2s",
            FirstOrderTemperatureLag(
                sensor_time_constants=(
                    ("cold_face_sensor", 2.0),
                    ("cold_exchanger_sensor", 2.0),
                ),
            ),
        ),
        ContactResistanceLagCase(
            "cold_pair_asymmetric_tau_2s_0p5s",
            FirstOrderTemperatureLag(
                sensor_time_constants=(
                    ("cold_face_sensor", 2.0),
                    ("cold_exchanger_sensor", 0.5),
                ),
            ),
        ),
    )


def run_contact_resistance_lag_study(
) -> ContactResistanceLagStudyResult:
    """Fit resistance while the estimator incorrectly assumes zero lag."""

    ideal = reference_contact_resistance_dataset_split()
    results = []
    for case in reference_contact_resistance_lag_cases():
        lagged = lag_contact_resistance_dataset_split(
            ideal,
            case.lag_model,
        )
        results.append(
            run_contact_resistance_robustness_case(
                case.name,
                lagged,
                ideal,
                fitted_sensor_names=FITTED_SENSOR_NAMES,
            )
        )
    return ContactResistanceLagStudyResult(cases=tuple(results))


def format_contact_resistance_lag_study_report(
    result: ContactResistanceLagStudyResult,
) -> str:
    """Format the sensor-lag/contact-dynamics comparison."""

    return format_contact_resistance_robustness_cases(
        "cold contact resistance sensor-lag study",
        result.cases,
    )


def main() -> None:
    """Run and print the dependency-free sensor-lag study."""

    print(
        format_contact_resistance_lag_study_report(
            run_contact_resistance_lag_study()
        )
    )


if __name__ == "__main__":
    main()
