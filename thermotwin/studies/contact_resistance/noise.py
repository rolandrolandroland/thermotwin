"""Repeated cold contact-resistance inference under Gaussian noise."""

import argparse
from dataclasses import dataclass
import math
from statistics import fmean, stdev
from typing import NamedTuple, Sequence, Tuple

from ...inference.contact_resistance import (
    REFERENCE_COLD_CONTACT_RESISTANCE,
    ContactResistanceDatasetSplit,
    ContactResistanceRegimeDataset,
    ContactResistanceSearchConfig,
    evaluate_contact_resistance_regime,
    fit_cold_contact_resistance,
    reference_contact_resistance_dataset_split,
)
from ...observations.noise import (
    GaussianTemperatureNoise,
    apply_gaussian_temperature_noise,
)


@dataclass(frozen=True)
class ContactResistanceNoiseStudyConfig:
    """Noise, repetition, seed, and scalar-search settings."""

    noise_standard_deviation: float = 0.05
    trial_count: int = 100
    first_seed: int = 2026
    search: ContactResistanceSearchConfig = (
        ContactResistanceSearchConfig(
            resistance_tolerance=1e-6,
            max_iterations=64,
        )
    )

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
        if not isinstance(self.search, ContactResistanceSearchConfig):
            raise ValueError("search must be a contact-resistance config")
        object.__setattr__(
            self,
            "noise_standard_deviation",
            float(self.noise_standard_deviation),
        )


class ContactResistanceNoiseSeeds(NamedTuple):
    """Independent saved seeds for one trial's three dataset splits."""

    training: int
    validation: int
    test: int


class ContactResistanceNoiseTrialResult(NamedTuple):
    """One noisy fit and its observation-versus-truth errors."""

    trial_index: int
    seeds: ContactResistanceNoiseSeeds
    inferred_cold_contact_resistance: float
    signed_parameter_error: float
    absolute_parameter_error: float
    reached_search_bound: bool
    search_evaluations: int
    training_observation_rmse: float
    validation_observation_rmse: float
    test_observation_rmse: float
    training_truth_rmse: float
    validation_truth_rmse: float
    test_truth_rmse: float


class ContactResistanceNoiseStudySummary(NamedTuple):
    """Empirical parameter distribution and mean regime errors."""

    true_cold_contact_resistance: float
    noise_standard_deviation: float
    trial_count: int
    mean_inferred_resistance: float
    parameter_standard_deviation: float
    mean_parameter_bias: float
    parameter_rmse: float
    minimum_inferred_resistance: float
    percentile_05_resistance: float
    median_inferred_resistance: float
    percentile_95_resistance: float
    maximum_inferred_resistance: float
    search_bound_hits: int
    mean_training_observation_rmse: float
    mean_validation_observation_rmse: float
    mean_test_observation_rmse: float
    mean_training_truth_rmse: float
    mean_validation_truth_rmse: float
    mean_test_truth_rmse: float


class ContactResistanceNoiseStudyResult(NamedTuple):
    """Frozen configuration, individual trials, and empirical summary."""

    config: ContactResistanceNoiseStudyConfig
    trials: Tuple[ContactResistanceNoiseTrialResult, ...]
    summary: ContactResistanceNoiseStudySummary


def contact_resistance_noise_seeds(
    first_seed: int,
    trial_index: int,
) -> ContactResistanceNoiseSeeds:
    """Return three nonoverlapping deterministic split seeds for one trial."""

    if (
        not isinstance(first_seed, int)
        or isinstance(first_seed, bool)
        or first_seed < 0
    ):
        raise ValueError("first seed must be a nonnegative integer")
    if (
        not isinstance(trial_index, int)
        or isinstance(trial_index, bool)
        or trial_index < 0
    ):
        raise ValueError("trial index must be a nonnegative integer")
    first_trial_seed = first_seed + 3 * trial_index
    if first_trial_seed + 2 >= _ADDITIONAL_REGIME_SEED_NAMESPACE:
        raise ValueError("first seed and trial index exceed the seed namespace")
    return ContactResistanceNoiseSeeds(
        training=first_trial_seed,
        validation=first_trial_seed + 1,
        test=first_trial_seed + 2,
    )


_ADDITIONAL_REGIME_SEED_NAMESPACE = 1 << 128


def _regime_noise_seed(split_seed: int, regime_index: int) -> int:
    """Map a split seed and regime index to a collision-free random seed.

    The first regime deliberately retains the historical split seed so the
    established one-regime studies remain numerically reproducible. Additional
    regimes use a disjoint integer namespace and Cantor pairing, which makes
    every ``(split_seed, regime_index)`` pair unique.
    """

    if (
        not isinstance(split_seed, int)
        or isinstance(split_seed, bool)
        or not 0 <= split_seed < _ADDITIONAL_REGIME_SEED_NAMESPACE
    ):
        raise ValueError("split seed must be a nonnegative supported integer")
    if (
        not isinstance(regime_index, int)
        or isinstance(regime_index, bool)
        or regime_index < 0
    ):
        raise ValueError("regime index must be a nonnegative integer")
    if regime_index == 0:
        return split_seed
    paired_index = regime_index - 1
    paired = (
        (split_seed + paired_index)
        * (split_seed + paired_index + 1)
        // 2
        + paired_index
    )
    return _ADDITIONAL_REGIME_SEED_NAMESPACE + paired


def _noise_regime_group(
    datasets: Sequence[ContactResistanceRegimeDataset],
    *,
    standard_deviation: float,
    seed: int,
) -> Tuple[ContactResistanceRegimeDataset, ...]:
    noisy_datasets = []
    for regime_index, dataset in enumerate(datasets):
        noisy = apply_gaussian_temperature_noise(
            dataset.observations,
            GaussianTemperatureNoise(
                default_standard_deviation=standard_deviation,
                random_seed=_regime_noise_seed(seed, regime_index),
            ),
        )
        noisy_datasets.append(
            ContactResistanceRegimeDataset(
                regime=dataset.regime,
                observations=noisy.dataset,
            )
        )
    return tuple(noisy_datasets)


def noisy_contact_resistance_dataset_split(
    ideal_datasets: ContactResistanceDatasetSplit,
    *,
    noise_standard_deviation: float,
    seeds: ContactResistanceNoiseSeeds,
) -> ContactResistanceDatasetSplit:
    """Add independent noise to three whole-regime dataset groups."""

    if not isinstance(ideal_datasets, ContactResistanceDatasetSplit):
        raise ValueError("ideal datasets must be a contact-resistance split")
    return ContactResistanceDatasetSplit(
        train=_noise_regime_group(
            ideal_datasets.train,
            standard_deviation=noise_standard_deviation,
            seed=seeds.training,
        ),
        validation=_noise_regime_group(
            ideal_datasets.validation,
            standard_deviation=noise_standard_deviation,
            seed=seeds.validation,
        ),
        test=_noise_regime_group(
            ideal_datasets.test,
            standard_deviation=noise_standard_deviation,
            seed=seeds.test,
        ),
    )


def _single_group_fitted_pair_rmse(
    cold_contact_resistance: float,
    datasets: Sequence[ContactResistanceRegimeDataset],
) -> float:
    metrics = tuple(
        evaluate_contact_resistance_regime(
            cold_contact_resistance,
            dataset,
        )
        for dataset in datasets
    )
    if len(metrics) != 1:
        raise ValueError(
            "the frozen noise study expects one regime in each split"
        )
    return metrics[0].fitted_pair_rmse


def run_contact_resistance_noise_trial(
    trial_index: int,
    ideal_datasets: ContactResistanceDatasetSplit,
    config: ContactResistanceNoiseStudyConfig,
) -> ContactResistanceNoiseTrialResult:
    """Fit one noisy training regime and score all three regimes."""

    seeds = contact_resistance_noise_seeds(
        config.first_seed,
        trial_index,
    )
    noisy_datasets = noisy_contact_resistance_dataset_split(
        ideal_datasets,
        noise_standard_deviation=config.noise_standard_deviation,
        seeds=seeds,
    )
    fit = fit_cold_contact_resistance(
        noisy_datasets.train,
        config.search,
    )
    inferred = fit.inferred_cold_contact_resistance
    signed_error = inferred - REFERENCE_COLD_CONTACT_RESISTANCE
    bound_tolerance = config.search.resistance_tolerance
    reached_bound = (
        inferred - config.search.lower_bound <= bound_tolerance
        or config.search.upper_bound - inferred <= bound_tolerance
    )
    return ContactResistanceNoiseTrialResult(
        trial_index=trial_index,
        seeds=seeds,
        inferred_cold_contact_resistance=inferred,
        signed_parameter_error=signed_error,
        absolute_parameter_error=abs(signed_error),
        reached_search_bound=reached_bound,
        search_evaluations=len(fit.evaluations),
        training_observation_rmse=_single_group_fitted_pair_rmse(
            inferred,
            noisy_datasets.train,
        ),
        validation_observation_rmse=_single_group_fitted_pair_rmse(
            inferred,
            noisy_datasets.validation,
        ),
        test_observation_rmse=_single_group_fitted_pair_rmse(
            inferred,
            noisy_datasets.test,
        ),
        training_truth_rmse=_single_group_fitted_pair_rmse(
            inferred,
            ideal_datasets.train,
        ),
        validation_truth_rmse=_single_group_fitted_pair_rmse(
            inferred,
            ideal_datasets.validation,
        ),
        test_truth_rmse=_single_group_fitted_pair_rmse(
            inferred,
            ideal_datasets.test,
        ),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("at least one value is required")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must lie from zero to one")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    upper_weight = position - lower_index
    return (
        (1.0 - upper_weight) * ordered[lower_index]
        + upper_weight * ordered[upper_index]
    )


def summarize_contact_resistance_noise_trials(
    trials: Sequence[ContactResistanceNoiseTrialResult],
    *,
    noise_standard_deviation: float,
) -> ContactResistanceNoiseStudySummary:
    """Calculate empirical parameter and temperature-error statistics."""

    trials = tuple(trials)
    if not trials:
        raise ValueError("at least one noise trial is required")
    estimates = tuple(
        trial.inferred_cold_contact_resistance for trial in trials
    )
    errors = tuple(trial.signed_parameter_error for trial in trials)
    return ContactResistanceNoiseStudySummary(
        true_cold_contact_resistance=(
            REFERENCE_COLD_CONTACT_RESISTANCE
        ),
        noise_standard_deviation=noise_standard_deviation,
        trial_count=len(trials),
        mean_inferred_resistance=fmean(estimates),
        parameter_standard_deviation=(
            stdev(estimates) if len(estimates) > 1 else 0.0
        ),
        mean_parameter_bias=fmean(errors),
        parameter_rmse=math.sqrt(
            fmean(error * error for error in errors)
        ),
        minimum_inferred_resistance=min(estimates),
        percentile_05_resistance=_percentile(estimates, 0.05),
        median_inferred_resistance=_percentile(estimates, 0.50),
        percentile_95_resistance=_percentile(estimates, 0.95),
        maximum_inferred_resistance=max(estimates),
        search_bound_hits=sum(
            trial.reached_search_bound for trial in trials
        ),
        mean_training_observation_rmse=fmean(
            trial.training_observation_rmse for trial in trials
        ),
        mean_validation_observation_rmse=fmean(
            trial.validation_observation_rmse for trial in trials
        ),
        mean_test_observation_rmse=fmean(
            trial.test_observation_rmse for trial in trials
        ),
        mean_training_truth_rmse=fmean(
            trial.training_truth_rmse for trial in trials
        ),
        mean_validation_truth_rmse=fmean(
            trial.validation_truth_rmse for trial in trials
        ),
        mean_test_truth_rmse=fmean(
            trial.test_truth_rmse for trial in trials
        ),
    )


def run_contact_resistance_noise_study(
    config: ContactResistanceNoiseStudyConfig = (
        ContactResistanceNoiseStudyConfig()
    ),
) -> ContactResistanceNoiseStudyResult:
    """Run the frozen repeated-noise study on CPU."""

    ideal_datasets = reference_contact_resistance_dataset_split()
    trials = tuple(
        run_contact_resistance_noise_trial(
            trial_index,
            ideal_datasets,
            config,
        )
        for trial_index in range(config.trial_count)
    )
    summary = summarize_contact_resistance_noise_trials(
        trials,
        noise_standard_deviation=config.noise_standard_deviation,
    )
    return ContactResistanceNoiseStudyResult(
        config=config,
        trials=trials,
        summary=summary,
    )


def format_contact_resistance_noise_study_report(
    result: ContactResistanceNoiseStudyResult,
) -> str:
    """Format the empirical parameter and temperature statistics."""

    summary = result.summary
    return "\n".join(
        (
            "cold contact resistance Gaussian-noise study",
            f"trials: {summary.trial_count}",
            (
                "true cold contact resistance: "
                f"{summary.true_cold_contact_resistance:.9f} K/W"
            ),
            (
                "temperature noise standard deviation: "
                f"{summary.noise_standard_deviation:.6f} K"
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
                "mean fitted-pair observation RMSE "
                "(train/validation/test): "
                f"{summary.mean_training_observation_rmse:.6f} / "
                f"{summary.mean_validation_observation_rmse:.6f} / "
                f"{summary.mean_test_observation_rmse:.6f} K"
            ),
            (
                "mean fitted-pair truth RMSE "
                "(train/validation/test): "
                f"{summary.mean_training_truth_rmse:.6f} / "
                f"{summary.mean_validation_truth_rmse:.6f} / "
                f"{summary.mean_test_truth_rmse:.6f} K"
            ),
        )
    )


def main() -> None:
    """Run the frozen study or a smaller command-line trial count."""

    parser = argparse.ArgumentParser(
        description=(
            "Repeat cold contact-resistance inference under Gaussian noise."
        )
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--noise-standard-deviation", type=float, default=0.05)
    parser.add_argument("--first-seed", type=int, default=2026)
    arguments = parser.parse_args()
    config = ContactResistanceNoiseStudyConfig(
        trial_count=arguments.trials,
        noise_standard_deviation=arguments.noise_standard_deviation,
        first_seed=arguments.first_seed,
    )
    print(
        format_contact_resistance_noise_study_report(
            run_contact_resistance_noise_study(config)
        )
    )


if __name__ == "__main__":
    main()
