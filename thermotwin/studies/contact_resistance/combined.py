"""Repeated contact-resistance inference with combined imperfections."""

import argparse
from dataclasses import dataclass, field
import math
from typing import NamedTuple, Optional, Tuple

from ...inference.contact_resistance import (
    REFERENCE_COLD_CONTACT_RESISTANCE,
    ContactResistanceDatasetSplit,
    ContactResistanceSearchConfig,
    reference_contact_resistance_dataset_split,
)
from .missingness import (
    ContactResistanceMissingnessCase,
    missingness_for_contact_resistance_case,
)
from .noise import (
    ContactResistanceNoiseStudySummary,
    ContactResistanceNoiseTrialResult,
    contact_resistance_noise_seeds,
    noisy_contact_resistance_dataset_split,
    summarize_contact_resistance_noise_trials,
)
from .robustness import (
    ROBUSTNESS_SEARCH_CONFIG,
    lag_contact_resistance_dataset_split,
    map_contact_resistance_dataset_split,
    match_split_schema,
    restrict_observation_dataset,
    run_contact_resistance_robustness_case,
    validate_sensor_names,
)
from ...observations.bias import (
    FixedTemperatureBias,
    apply_fixed_temperature_bias,
)
from ...observations.lag import FirstOrderTemperatureLag
from ...observations.missingness import (
    apply_deterministic_temperature_missingness,
)


def _default_combined_bias() -> FixedTemperatureBias:
    return FixedTemperatureBias(
        sensor_biases=(("cold_face_sensor", 0.10),),
    )


def _default_combined_lag() -> FirstOrderTemperatureLag:
    return FirstOrderTemperatureLag(
        sensor_time_constants=(("cold_face_sensor", 2.0),),
    )


@dataclass(frozen=True)
class ContactResistanceCombinedStudyConfig:
    """Frozen combined measurement pipeline and Monte Carlo settings."""

    noise_standard_deviation: float = 0.05
    trial_count: int = 100
    first_seed: int = 2026
    bias_model: FixedTemperatureBias = field(
        default_factory=_default_combined_bias
    )
    lag_model: FirstOrderTemperatureLag = field(
        default_factory=_default_combined_lag
    )
    turn_off_half_width: Optional[float] = 2.0
    available_sensor_names: Tuple[str, ...] = (
        "cold_face_sensor",
        "cold_exchanger_sensor",
    )
    search: ContactResistanceSearchConfig = ROBUSTNESS_SEARCH_CONFIG

    def __post_init__(self) -> None:
        if isinstance(self.noise_standard_deviation, bool):
            raise ValueError(
                "noise standard deviation must be finite and nonnegative"
            )
        try:
            noise_is_finite = math.isfinite(
                self.noise_standard_deviation
            )
        except TypeError as error:
            raise ValueError(
                "noise standard deviation must be finite and nonnegative"
            ) from error
        if not noise_is_finite or self.noise_standard_deviation < 0.0:
            raise ValueError(
                "noise standard deviation must be finite and nonnegative"
            )
        if (
            not isinstance(self.trial_count, int)
            or isinstance(self.trial_count, bool)
            or self.trial_count <= 0
        ):
            raise ValueError("trial count must be a positive integer")
        if (
            not isinstance(self.first_seed, int)
            or isinstance(self.first_seed, bool)
            or self.first_seed < 0
        ):
            raise ValueError("first seed must be a nonnegative integer")
        if not isinstance(self.bias_model, FixedTemperatureBias):
            raise ValueError("bias model must be fixed temperature bias")
        if not isinstance(self.lag_model, FirstOrderTemperatureLag):
            raise ValueError("lag model must be first-order temperature lag")
        if isinstance(self.turn_off_half_width, bool):
            raise ValueError(
                "turn-off half-width must be finite and nonnegative"
            )
        if self.turn_off_half_width is not None:
            try:
                half_width_is_finite = math.isfinite(
                    self.turn_off_half_width
                )
            except TypeError as error:
                raise ValueError(
                    "turn-off half-width must be finite and nonnegative"
                ) from error
            if (
                not half_width_is_finite
                or self.turn_off_half_width < 0.0
            ):
                raise ValueError(
                    "turn-off half-width must be finite and nonnegative"
                )
        if not isinstance(self.search, ContactResistanceSearchConfig):
            raise ValueError("search must be a contact-resistance config")
        object.__setattr__(
            self,
            "noise_standard_deviation",
            float(self.noise_standard_deviation),
        )
        if self.turn_off_half_width is not None:
            object.__setattr__(
                self,
                "turn_off_half_width",
                float(self.turn_off_half_width),
            )
        object.__setattr__(
            self,
            "available_sensor_names",
            validate_sensor_names(self.available_sensor_names),
        )


class ContactResistanceCombinedStudyResult(NamedTuple):
    """Configuration, individual fits, and empirical combined summary."""

    config: ContactResistanceCombinedStudyConfig
    trials: Tuple[ContactResistanceNoiseTrialResult, ...]
    summary: ContactResistanceNoiseStudySummary


def _combined_deterministic_base(
    config: ContactResistanceCombinedStudyConfig,
) -> Tuple[ContactResistanceDatasetSplit, ContactResistanceDatasetSplit]:
    ideal = reference_contact_resistance_dataset_split()
    lagged = lag_contact_resistance_dataset_split(
        ideal,
        config.lag_model,
    )
    biased = map_contact_resistance_dataset_split(
        lagged,
        lambda dataset: apply_fixed_temperature_bias(
            dataset.observations,
            config.bias_model,
        ).dataset,
    )
    return ideal, biased


def _remove_turnoff_and_restrict(
    datasets: ContactResistanceDatasetSplit,
    config: ContactResistanceCombinedStudyConfig,
) -> ContactResistanceDatasetSplit:
    missingness_case = ContactResistanceMissingnessCase(
        name="combined_turn_off_window",
        turn_off_half_width=config.turn_off_half_width,
    )
    incomplete = map_contact_resistance_dataset_split(
        datasets,
        lambda dataset: apply_deterministic_temperature_missingness(
            dataset.observations,
            missingness_for_contact_resistance_case(
                missingness_case,
                dataset,
            ),
        ).dataset,
    )
    return map_contact_resistance_dataset_split(
        incomplete,
        lambda dataset: restrict_observation_dataset(
            dataset.observations,
            config.available_sensor_names,
        ),
    )


def run_contact_resistance_combined_study(
    config: ContactResistanceCombinedStudyConfig = (
        ContactResistanceCombinedStudyConfig()
    ),
) -> ContactResistanceCombinedStudyResult:
    """Run lag, bias, noise, missingness, and sensor restriction."""

    ideal, biased = _combined_deterministic_base(config)
    visible_schema = _remove_turnoff_and_restrict(biased, config)
    visible_truth = match_split_schema(ideal, visible_schema)
    trials = []
    for trial_index in range(config.trial_count):
        seeds = contact_resistance_noise_seeds(
            config.first_seed,
            trial_index,
        )
        noisy = noisy_contact_resistance_dataset_split(
            biased,
            noise_standard_deviation=config.noise_standard_deviation,
            seeds=seeds,
        )
        observations = _remove_turnoff_and_restrict(noisy, config)
        case = run_contact_resistance_robustness_case(
            f"combined_trial_{trial_index}",
            observations,
            visible_truth,
            fitted_sensor_names=config.available_sensor_names,
            search=config.search,
        )
        inferred = case.inferred_cold_contact_resistance
        bound_tolerance = config.search.resistance_tolerance
        reached_bound = (
            inferred - config.search.lower_bound <= bound_tolerance
            or config.search.upper_bound - inferred <= bound_tolerance
        )
        trials.append(
            ContactResistanceNoiseTrialResult(
                trial_index=trial_index,
                seeds=seeds,
                inferred_cold_contact_resistance=inferred,
                signed_parameter_error=case.signed_parameter_error,
                absolute_parameter_error=case.absolute_parameter_error,
                reached_search_bound=reached_bound,
                search_evaluations=case.search_evaluations,
                training_observation_rmse=(
                    case.training_observation_rmse
                ),
                validation_observation_rmse=(
                    case.validation_observation_rmse
                ),
                test_observation_rmse=case.test_observation_rmse,
                training_truth_rmse=case.training_truth_rmse,
                validation_truth_rmse=case.validation_truth_rmse,
                test_truth_rmse=case.test_truth_rmse,
            )
        )
    trial_tuple = tuple(trials)
    return ContactResistanceCombinedStudyResult(
        config=config,
        trials=trial_tuple,
        summary=summarize_contact_resistance_noise_trials(
            trial_tuple,
            noise_standard_deviation=config.noise_standard_deviation,
        ),
    )


def format_contact_resistance_combined_study_report(
    result: ContactResistanceCombinedStudyResult,
) -> str:
    """Format the combined empirical parameter and error statistics."""

    config = result.config
    summary = result.summary
    sensor_names = ",".join(config.available_sensor_names)
    outage_description = (
        "none"
        if config.turn_off_half_width is None
        else f"{config.turn_off_half_width:.3f} s"
    )
    return "\n".join(
        (
            "cold contact resistance combined-imperfections study",
            "pipeline: dense lag -> sample -> bias -> noise -> "
            "turn-off missingness -> restrict sensors",
            f"trials: {summary.trial_count}",
            (
                "temperature noise standard deviation: "
                f"{summary.noise_standard_deviation:.6f} K"
            ),
            f"available sensors: {sensor_names}",
            (
                "turn-off outage half-width: "
                f"{outage_description}"
            ),
            (
                "mean inferred resistance: "
                f"{summary.mean_inferred_resistance:.9f} K/W"
            ),
            (
                "sample parameter standard deviation: "
                f"{summary.parameter_standard_deviation:.9f} K/W"
            ),
            (
                "mean parameter bias: "
                f"{summary.mean_parameter_bias:+.9f} K/W"
            ),
            f"parameter RMSE: {summary.parameter_rmse:.9f} K/W",
            (
                "empirical 5th--95th percentile interval: "
                f"[{summary.percentile_05_resistance:.9f}, "
                f"{summary.percentile_95_resistance:.9f}] K/W"
            ),
            f"search bound hits: {summary.search_bound_hits}",
            (
                "mean observation RMSE train/validation/test: "
                f"{summary.mean_training_observation_rmse:.6f} / "
                f"{summary.mean_validation_observation_rmse:.6f} / "
                f"{summary.mean_test_observation_rmse:.6f} K"
            ),
            (
                "mean truth RMSE train/validation/test: "
                f"{summary.mean_training_truth_rmse:.6f} / "
                f"{summary.mean_validation_truth_rmse:.6f} / "
                f"{summary.mean_test_truth_rmse:.6f} K"
            ),
        )
    )


def main() -> None:
    """Run the frozen study or a smaller command-line sample."""

    parser = argparse.ArgumentParser(
        description="Repeat contact inference with combined imperfections."
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--noise-standard-deviation",
        type=float,
        default=0.05,
    )
    parser.add_argument("--first-seed", type=int, default=2026)
    arguments = parser.parse_args()
    config = ContactResistanceCombinedStudyConfig(
        trial_count=arguments.trials,
        noise_standard_deviation=arguments.noise_standard_deviation,
        first_seed=arguments.first_seed,
    )
    print(
        format_contact_resistance_combined_study_report(
            run_contact_resistance_combined_study(config)
        )
    )


if __name__ == "__main__":
    main()
