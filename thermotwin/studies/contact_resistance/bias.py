"""Cold contact-resistance inference under fixed sensor bias."""

from typing import NamedTuple, Tuple

from ...inference.contact_resistance import (
    FITTED_SENSOR_NAMES,
    reference_contact_resistance_dataset_split,
)
from .robustness import (
    ContactResistanceRobustnessCaseResult,
    format_contact_resistance_robustness_cases,
    map_contact_resistance_dataset_split,
    run_contact_resistance_robustness_case,
)
from ...observations.bias import (
    FixedTemperatureBias,
    apply_fixed_temperature_bias,
)


class ContactResistanceBiasCase(NamedTuple):
    """One named deterministic sensor-bias pattern."""

    name: str
    bias_model: FixedTemperatureBias


class ContactResistanceBiasStudyResult(NamedTuple):
    """Frozen bias patterns and their inference results."""

    cases: Tuple[ContactResistanceRobustnessCaseResult, ...]


def reference_contact_resistance_bias_cases(
) -> Tuple[ContactResistanceBiasCase, ...]:
    """Return zero, individual, common, and differential cold biases."""

    return (
        ContactResistanceBiasCase(
            "zero_bias",
            FixedTemperatureBias(),
        ),
        ContactResistanceBiasCase(
            "cold_face_plus_0p10_K",
            FixedTemperatureBias(
                sensor_biases=(("cold_face_sensor", 0.10),),
            ),
        ),
        ContactResistanceBiasCase(
            "cold_exchanger_plus_0p10_K",
            FixedTemperatureBias(
                sensor_biases=(("cold_exchanger_sensor", 0.10),),
            ),
        ),
        ContactResistanceBiasCase(
            "cold_pair_common_plus_0p10_K",
            FixedTemperatureBias(
                sensor_biases=(
                    ("cold_face_sensor", 0.10),
                    ("cold_exchanger_sensor", 0.10),
                ),
            ),
        ),
        ContactResistanceBiasCase(
            "cold_pair_differential_0p10_K",
            FixedTemperatureBias(
                sensor_biases=(
                    ("cold_face_sensor", 0.05),
                    ("cold_exchanger_sensor", -0.05),
                ),
            ),
        ),
    )


def run_contact_resistance_bias_study(
) -> ContactResistanceBiasStudyResult:
    """Fit the frozen cold contact under five deterministic bias patterns."""

    ideal = reference_contact_resistance_dataset_split()
    results = []
    for case in reference_contact_resistance_bias_cases():
        biased = map_contact_resistance_dataset_split(
            ideal,
            lambda dataset, model=case.bias_model: (
                apply_fixed_temperature_bias(
                    dataset.observations,
                    model,
                ).dataset
            ),
        )
        results.append(
            run_contact_resistance_robustness_case(
                case.name,
                biased,
                ideal,
                fitted_sensor_names=FITTED_SENSOR_NAMES,
            )
        )
    return ContactResistanceBiasStudyResult(cases=tuple(results))


def format_contact_resistance_bias_study_report(
    result: ContactResistanceBiasStudyResult,
) -> str:
    """Format the frozen fixed-bias comparison."""

    return format_contact_resistance_robustness_cases(
        "cold contact resistance fixed-bias study",
        result.cases,
    )


def main() -> None:
    """Run and print the dependency-free fixed-bias study."""

    print(
        format_contact_resistance_bias_study_report(
            run_contact_resistance_bias_study()
        )
    )


if __name__ == "__main__":
    main()
