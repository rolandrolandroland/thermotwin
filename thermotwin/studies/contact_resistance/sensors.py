"""Contact-resistance inference with restricted temperature sensors."""

from typing import NamedTuple, Tuple

from ...inference.contact_resistance import (
    ALL_SENSOR_NAMES,
    FITTED_SENSOR_NAMES,
    reference_contact_resistance_dataset_split,
)
from .robustness import (
    ContactResistanceRobustnessCaseResult,
    format_contact_resistance_robustness_cases,
    map_contact_resistance_dataset_split,
    restrict_observation_dataset,
    run_contact_resistance_robustness_case,
    validate_sensor_names,
)


class ContactResistanceSensorCase(NamedTuple):
    """One named set of physically available temperature sensors."""

    name: str
    sensor_names: Tuple[str, ...]


class ContactResistanceSensorStudyResult(NamedTuple):
    """Frozen sensor sets and their resistance-inference diagnostics."""

    cases: Tuple[ContactResistanceRobustnessCaseResult, ...]


def reference_contact_resistance_sensor_cases(
) -> Tuple[ContactResistanceSensorCase, ...]:
    """Return cold-pair, single-cold, hot-pair, and all-sensor cases."""

    return (
        ContactResistanceSensorCase(
            "cold_pair",
            FITTED_SENSOR_NAMES,
        ),
        ContactResistanceSensorCase(
            "cold_face_only",
            ("cold_face_sensor",),
        ),
        ContactResistanceSensorCase(
            "cold_exchanger_only",
            ("cold_exchanger_sensor",),
        ),
        ContactResistanceSensorCase(
            "hot_pair_only",
            ("hot_face_sensor", "hot_exchanger_sensor"),
        ),
        ContactResistanceSensorCase(
            "all_four_sensors",
            ALL_SENSOR_NAMES,
        ),
    )


def run_contact_resistance_sensor_study(
) -> ContactResistanceSensorStudyResult:
    """Fit the cold contact from each frozen available-sensor set."""

    ideal = reference_contact_resistance_dataset_split()
    results = []
    for case in reference_contact_resistance_sensor_cases():
        sensor_names = validate_sensor_names(case.sensor_names)
        restricted = map_contact_resistance_dataset_split(
            ideal,
            lambda dataset, names=sensor_names: (
                restrict_observation_dataset(
                    dataset.observations,
                    names,
                )
            ),
        )
        results.append(
            run_contact_resistance_robustness_case(
                case.name,
                restricted,
                restricted,
                fitted_sensor_names=sensor_names,
            )
        )
    return ContactResistanceSensorStudyResult(cases=tuple(results))


def format_contact_resistance_sensor_study_report(
    result: ContactResistanceSensorStudyResult,
) -> str:
    """Format the available-sensor comparison."""

    return format_contact_resistance_robustness_cases(
        "cold contact resistance restricted-sensor study",
        result.cases,
    )


def main() -> None:
    """Run and print the dependency-free restricted-sensor study."""

    print(
        format_contact_resistance_sensor_study_report(
            run_contact_resistance_sensor_study()
        )
    )


if __name__ == "__main__":
    main()
