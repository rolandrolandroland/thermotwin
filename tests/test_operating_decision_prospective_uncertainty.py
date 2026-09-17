from __future__ import annotations

from dataclasses import replace
import inspect
import json
import math
import unittest
from unittest.mock import Mock, patch

from thermotwin.observations.test_stand import regular_measurement_times
from thermotwin.studies.operating_decision import (
    FINAL_EVALUATION,
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    POLICY_NAMES,
    MarginInterval,
    NumericalFailure,
    OperatingRegime,
    default_fixed_policies,
    initial_acquisition_regime,
)
from thermotwin.studies.operating_decision_prospective_random_streams import (
    RUN_BIAS,
    WHITE_NOISE,
    ProspectiveRandomStreamNamespace,
    ProspectiveRandomStreamRegistry,
    prospective_observation_stream,
    prospective_parameter_stream,
    prospective_probe_stream,
)
from thermotwin.studies.operating_decision_realism import (
    OperatingDecisionRealismConfig,
    RealisticAcquisitionFitSet,
    RealisticCandidateFit,
    RealisticOperatingRun,
    RunInstrumentation,
    _decoded_parameters,
    _parameter_spec,
)
from thermotwin.studies.sensor_model_discrimination import (
    COLD_EXCHANGER,
    COLD_FACE,
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    HOT_EXCHANGER,
    MODEL_NAMES,
    ObservableRun,
    ObservableValue,
)
import thermotwin.studies.operating_decision_prospective_uncertainty as prospective


def _final_regime(config: OperatingDecisionRealismConfig) -> OperatingRegime:
    return OperatingRegime(
        "untouched_final_operating_schedule",
        FINAL_EVALUATION,
        config.final_current,
        (),
    )


def _initial_run(
    config: OperatingDecisionRealismConfig,
    *,
    value_shift: float = 0.0,
) -> RealisticOperatingRun:
    regime = initial_acquisition_regime()
    times = regular_measurement_times(80.0, config.sensor.sampling_interval)
    values = tuple(
        ObservableValue(channel, time, 300.0 + value_shift + channel_index)
        for time in times
        for channel_index, channel in enumerate((COLD_EXCHANGER, HOT_EXCHANGER))
    )
    return RealisticOperatingRun(
        regime,
        ObservableRun(regime.name, regime.current, values),
        RunInstrumentation(False),
    )


def _fit(
    model_name: str,
    config: OperatingDecisionRealismConfig,
    *,
    objective: float = 1.0,
    reached_bound: bool = False,
    converged: bool = True,
    covariance_scale: float = 0.01,
) -> RealisticCandidateFit:
    spec = _parameter_spec(model_name, False, config)
    offsets = (0.0,) * len(spec.names)
    physical, interface_mass, series_resistance, face_sensor = _decoded_parameters(
        model_name,
        offsets,
        spec,
    )
    covariance = tuple(
        tuple(
            covariance_scale if row == column else 0.0
            for column in range(len(spec.names))
        )
        for row in range(len(spec.names))
    )
    return RealisticCandidateFit(
        model_name=model_name,
        log_multipliers=offsets,
        parameter_names=spec.names,
        physical_values=physical,
        interface_mass=interface_mass,
        series_resistance=series_resistance,
        face_sensor=face_sensor,
        objective=objective,
        covariance=covariance,
        reached_bound=reached_bound,
        evaluation_count=7,
        converged=converged,
        termination_reason="scaled_projected_gradient_tolerance",
        completed_iterations=3,
        accepted_iterations=2,
        scaled_gradient_infinity_norm=0.0,
        last_step_infinity_norm=0.0,
        last_relative_objective_reduction=0.0,
    )


def _interval(model_name: str, lower: float, upper: float) -> MarginInterval:
    return MarginInterval(
        model_name,
        0.5 * (lower + upper),
        0.25 * (upper - lower),
        lower,
        upper,
    )


def _interval_for_fit(fit, *_args) -> MarginInterval:
    if fit.model_name == FOUR_STATE_MODEL:
        return _interval(FOUR_STATE_MODEL, -1.0, 0.25)
    return _interval(FIVE_STATE_MODEL, -0.25, 1.0)


def _evidence(
    config: OperatingDecisionRealismConfig,
    *,
    excluded_model: str | None = None,
    value_shift: float = 0.0,
):
    fits = tuple(
        _fit(
            model_name,
            config,
            reached_bound=model_name == excluded_model,
        )
        for model_name in MODEL_NAMES
    )
    fit_set = RealisticAcquisitionFitSet(fits, ())
    with (
        patch.object(
            prospective,
            "_multistart_fit_candidates",
            return_value=fit_set,
        ),
        patch(
            "thermotwin.studies.operating_decision_prospective."
            "forecast_realistic_margin_interval",
            side_effect=_interval_for_fit,
        ),
    ):
        return prospective.prepare_prospective_acquisition_evidence(
            _initial_run(config, value_shift=value_shift),
            _final_regime(config),
            config,
        )


def _candidate_outcomes(width: float = 0.5):
    half = 0.5 * width
    return tuple(
        sorted(
            (
                prospective.ProspectiveCandidateOutcome(
                    FOUR_STATE_MODEL,
                    "admissible",
                    1.0,
                    _interval(FOUR_STATE_MODEL, -half, half),
                ),
                prospective.ProspectiveCandidateOutcome(
                    FIVE_STATE_MODEL,
                    "admissible",
                    1.0,
                    _interval(FIVE_STATE_MODEL, -half, half),
                ),
            ),
            key=lambda item: item.model_name,
        )
    )


def _completed_draw(
    policy_name: str,
    generator_model: str,
    draw_index: int,
    width: float,
    *,
    stable: bool = True,
    generator_offsets=(0.0,),
) -> prospective.ProspectiveDrawOutcome:
    newly_excluded = () if stable else (FOUR_STATE_MODEL,)
    outcomes = _candidate_outcomes(width) if stable else (
        prospective.ProspectiveCandidateOutcome(
            FIVE_STATE_MODEL,
            "admissible",
            1.0,
            _interval(FIVE_STATE_MODEL, -0.5 * width, 0.5 * width),
        ),
        prospective.ProspectiveCandidateOutcome(
            FOUR_STATE_MODEL,
            "fit_reached_bound",
            2.0,
            None,
        ),
    )
    outcomes = tuple(sorted(outcomes, key=lambda item: item.model_name))
    return prospective.ProspectiveDrawOutcome(
        policy_name=policy_name,
        generator_model=generator_model,
        draw_index=draw_index,
        generator_log_offsets=tuple(generator_offsets),
        face_probe_log_offsets=(0.0, 0.0)
        if policy_name == FIXED_FACE_TEMPERATURE
        else None,
        candidate_outcomes=outcomes,
        initially_admissible_became_inadmissible=newly_excluded,
        initially_excluded_became_admissible=(),
        stable=stable,
        failed=False,
        failure_stage=None,
        failure_model=None,
        failure_type=None,
        raw_after_width=width,
        scored_after_width=width,
        synthetic_observation_digest="a" * 64,
    )


def _register_parameter_use(fit, namespace, draw_index, registry):
    stream = prospective_parameter_stream(
        namespace,
        generator_model=fit.model_name,
        draw_index=draw_index,
    )
    for action in (FIXED_THERMAL, FIXED_VOLTAGE, FIXED_FACE_TEMPERATURE):
        registry.register(
            stream,
            consumer=f"test/{fit.model_name}/{draw_index}/{action}/parameters",
            shared_for_action=action,
        )


def _register_probe_use(namespace, model_name, draw_index, registry):
    registry.register(
        prospective_probe_stream(
            namespace,
            generator_model=model_name,
            draw_index=draw_index,
        ),
        consumer=f"test/{model_name}/{draw_index}/probe",
    )


def _register_observation_uses(kwargs):
    policy = kwargs["policy"]
    generator_model = kwargs["generator_fit"].model_name
    draw_index = kwargs["draw_index"]
    namespace = kwargs["namespace"]
    registry = kwargs["registry"]
    for regime in policy.additional_regimes:
        for channel in regime.channels:
            for purpose in (RUN_BIAS, WHITE_NOISE):
                registry.register(
                    prospective_observation_stream(
                        namespace,
                        generator_model=generator_model,
                        draw_index=draw_index,
                        purpose=purpose,
                        action=policy.name,
                        run=regime.name,
                        channel=channel,
                    ),
                    consumer=(
                        f"test/{generator_model}/{draw_index}/{policy.name}/"
                        f"{regime.name}/{channel}/{purpose}"
                    ),
                )


class _SequenceGenerator:
    def __init__(self, values):
        self.values = iter(values)

    def gauss(self, _mean, _standard_deviation):
        return next(self.values)


class _SequenceStream:
    def __init__(self, values):
        self.values = tuple(values)

    def new_generator(self):
        return _SequenceGenerator(self.values)


class ProspectiveUncertaintyValidationTests(unittest.TestCase):
    def test_scoring_interface_cannot_accept_revealed_information(self):
        forbidden = {
            "truth_condition",
            "truth_family",
            "verification_run",
            "verification_score",
            "selected_policy_outcome",
            "true_margin",
            "final_response",
            "trial_index",
            "device_token",
        }
        prepare_parameters = set(
            inspect.signature(
                prospective.prepare_prospective_acquisition_evidence
            ).parameters
        )
        estimate_parameters = set(
            inspect.signature(
                prospective.estimate_prospective_action_uncertainty
            ).parameters
        )
        self.assertFalse(forbidden.intersection(prepare_parameters))
        self.assertFalse(forbidden.intersection(estimate_parameters))

    def test_config_is_strict_and_freezes_every_algorithm_label(self):
        for name, value in (
            ("draw_count", 0),
            ("draw_count", True),
            ("max_parameter_draw_attempts", 0),
            ("max_parameter_draw_attempts", True),
            ("minimum_stable_fraction", 0.0),
            ("minimum_stable_fraction", -0.01),
            ("minimum_stable_fraction", 1.01),
            ("minimum_stable_fraction", math.nan),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    prospective.ProspectiveUncertaintyConfig(**{name: value})
        with self.assertRaisesRegex(ValueError, "unsupported prospective estimator"):
            prospective.ProspectiveUncertaintyConfig(estimator="changed")
        with self.assertRaisesRegex(ValueError, "failure policy"):
            prospective.ProspectiveUncertaintyConfig(
                draw_failure_policy="drop_failed_draws"
            )

    def test_candidate_outcome_rejects_negative_objective(self):
        with self.assertRaises(ValueError):
            prospective.ProspectiveCandidateOutcome(
                FOUR_STATE_MODEL,
                "admissible",
                -1.0,
                _interval(FOUR_STATE_MODEL, -1.0, 1.0),
            )

    def test_candidate_outcome_rejects_inverted_or_nonfinite_interval(self):
        with self.assertRaises(ValueError):
            prospective.ProspectiveCandidateOutcome(
                FOUR_STATE_MODEL,
                "admissible",
                1.0,
                MarginInterval(FOUR_STATE_MODEL, 0.0, 1.0, 1.0, -1.0),
            )
        with self.assertRaises(ValueError):
            prospective.ProspectiveCandidateOutcome(
                FOUR_STATE_MODEL,
                "admissible",
                1.0,
                MarginInterval(
                    FOUR_STATE_MODEL,
                    0.0,
                    1.0,
                    -1.0,
                    math.inf,
                ),
            )

    def test_completed_draw_requires_both_candidate_statuses(self):
        with self.assertRaises(ValueError):
            prospective.ProspectiveDrawOutcome(
                policy_name=FIXED_VOLTAGE,
                generator_model=FOUR_STATE_MODEL,
                draw_index=0,
                generator_log_offsets=(0.0,),
                face_probe_log_offsets=None,
                candidate_outcomes=(),
                initially_admissible_became_inadmissible=(),
                initially_excluded_became_admissible=(),
                stable=True,
                failed=False,
                failure_stage=None,
                failure_model=None,
                failure_type=None,
                raw_after_width=1.0,
                scored_after_width=1.0,
                synthetic_observation_digest="a" * 64,
            )

    def test_completed_draw_requires_a_lowercase_sha256_observation_digest(self):
        values = dict(
            policy_name=FIXED_VOLTAGE,
            generator_model=FOUR_STATE_MODEL,
            draw_index=0,
            generator_log_offsets=(0.0,),
            face_probe_log_offsets=None,
            candidate_outcomes=_candidate_outcomes(),
            initially_admissible_became_inadmissible=(),
            initially_excluded_became_admissible=(),
            stable=True,
            failed=False,
            failure_stage=None,
            failure_model=None,
            failure_type=None,
            raw_after_width=1.0,
            scored_after_width=1.0,
        )
        for digest in ("not-a-digest", "A" * 64, "z" * 64):
            with self.subTest(digest=digest):
                with self.assertRaises(ValueError):
                    prospective.ProspectiveDrawOutcome(
                        **values,
                        synthetic_observation_digest=digest,
                    )

    def test_summary_counts_are_disjoint(self):
        with self.assertRaises(ValueError):
            prospective.ProspectiveModelActionSummary(
                FIXED_THERMAL,
                FOUR_STATE_MODEL,
                draw_count=1,
                stable_draw_count=1,
                failed_draw_count=1,
                mean_after_width=0.5,
                eligible=True,
            )

    def test_action_uncertainty_recomputes_worst_case_source_mean(self):
        summaries = tuple(
            prospective.ProspectiveModelActionSummary(
                FIXED_THERMAL,
                model_name,
                draw_count=2,
                stable_draw_count=2,
                failed_draw_count=0,
                mean_after_width=mean,
                eligible=True,
            )
            for model_name, mean in zip(sorted(MODEL_NAMES), (0.5, 1.5))
        )
        with self.assertRaises(ValueError):
            prospective.ProspectiveActionUncertainty(
                policy_name=FIXED_THERMAL,
                eligible=True,
                failure_reason=None,
                uncertainty_before=2.0,
                expected_uncertainty_after=0.5,
                prospective_draw_count=4,
                source_summaries=summaries,
            )

    def test_action_uncertainty_requires_matching_unique_source_summaries(self):
        mismatched = prospective.ProspectiveModelActionSummary(
            FIXED_VOLTAGE,
            FOUR_STATE_MODEL,
            draw_count=2,
            stable_draw_count=2,
            failed_draw_count=0,
            mean_after_width=0.5,
            eligible=True,
        )
        with self.assertRaises(ValueError):
            prospective.ProspectiveActionUncertainty(
                policy_name=FIXED_THERMAL,
                eligible=True,
                failure_reason=None,
                uncertainty_before=2.0,
                expected_uncertainty_after=0.5,
                prospective_draw_count=2,
                source_summaries=(mismatched,),
            )

        duplicated = prospective.ProspectiveModelActionSummary(
            FIXED_THERMAL,
            FOUR_STATE_MODEL,
            draw_count=1,
            stable_draw_count=1,
            failed_draw_count=0,
            mean_after_width=0.5,
            eligible=True,
        )
        with self.assertRaises(ValueError):
            prospective.ProspectiveActionUncertainty(
                policy_name=FIXED_THERMAL,
                eligible=True,
                failure_reason=None,
                uncertainty_before=2.0,
                expected_uncertainty_after=0.5,
                prospective_draw_count=2,
                source_summaries=(duplicated, duplicated),
            )

    def test_covariance_factorization_and_bounded_sampler_are_conservative(self):
        singular = prospective._positive_semidefinite_cholesky(
            ((1.0, 1.0), (1.0, 1.0))
        )
        self.assertEqual(singular[1][1], 0.0)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            prospective._positive_semidefinite_cholesky(
                ((1.0, 0.2), (0.1, 1.0))
            )
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            prospective._positive_semidefinite_cholesky(
                ((1.0, 0.0), (0.0, -0.1))
            )

        accepted = prospective._bounded_gaussian_draw(
            (0.0,),
            ((1.0,),),
            ((-0.5, 0.5),),
            _SequenceStream((2.0, 0.25)),
            2,
        )
        self.assertEqual(accepted, (0.25,))
        with self.assertRaisesRegex(ValueError, "exhausted"):
            prospective._bounded_gaussian_draw(
                (0.0,),
                ((1.0,),),
                ((-0.5, 0.5),),
                _SequenceStream((2.0, -2.0)),
                2,
            )
        self.assertEqual(
            prospective._bounded_gaussian_draw(
                (0.25,),
                ((0.0,),),
                ((-0.5, 0.5),),
                _SequenceStream((999.0,)),
                1,
            ),
            (0.25,),
        )


class ProspectiveUncertaintyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = OperatingDecisionRealismConfig()

    def test_factory_checks_exact_initial_grid_and_final_schedule_before_fit(self):
        good = _initial_run(self.config)
        sparse_values = good.observations.values[:-2]
        sparse = RealisticOperatingRun(
            good.regime,
            ObservableRun(
                good.observations.name,
                good.observations.current,
                sparse_values,
            ),
            good.instrumentation,
        )
        fit = Mock(side_effect=AssertionError("fit must not run"))
        with patch.object(prospective, "_multistart_fit_candidates", fit):
            with self.assertRaisesRegex(ValueError, "exact regular grid"):
                prospective.prepare_prospective_acquisition_evidence(
                    sparse,
                    _final_regime(self.config),
                    self.config,
                )

            wrong_final = OperatingRegime(
                "renamed_final_schedule",
                FINAL_EVALUATION,
                self.config.final_current,
                (),
            )
            with self.assertRaisesRegex(ValueError, "frozen untouched"):
                prospective.prepare_prospective_acquisition_evidence(
                    good,
                    wrong_final,
                    self.config,
                )
        fit.assert_not_called()

    def test_factory_refits_only_the_common_initial_run_and_seals_evidence(self):
        fit_set = RealisticAcquisitionFitSet(
            tuple(_fit(name, self.config) for name in MODEL_NAMES),
            (),
        )
        with (
            patch.object(
                prospective,
                "_multistart_fit_candidates",
                return_value=fit_set,
            ) as refit,
            patch(
                "thermotwin.studies.operating_decision_prospective."
                "forecast_realistic_margin_interval",
                side_effect=_interval_for_fit,
            ),
        ):
            evidence = prospective.prepare_prospective_acquisition_evidence(
                _initial_run(self.config),
                _final_regime(self.config),
                self.config,
            )
        self.assertEqual(refit.call_args.args[0], (evidence.common_initial_run,))
        self.assertEqual(len(evidence.evidence_digest), 64)
        with self.assertRaisesRegex(ValueError, "acquisition factory"):
            prospective.ProspectiveAcquisitionEvidence(
                common_initial_run=evidence.common_initial_run,
                fit_set=evidence.fit_set,
                snapshot=evidence.snapshot,
                final_regime=evidence.final_regime,
                physical_protocol_digest=evidence.physical_protocol_digest,
                evidence_digest=evidence.evidence_digest,
                _construction_seal=object(),
            )

    def test_evidence_digest_binds_observations_and_validation_rejects_tampering(self):
        first = _evidence(self.config, value_shift=0.0)
        second = _evidence(self.config, value_shift=0.01)
        self.assertNotEqual(first.evidence_digest, second.evidence_digest)

        tampered = replace(first, evidence_digest="0" * 64)
        with (
            patch.object(
                prospective,
                "_multistart_fit_candidates",
                return_value=first.fit_set,
            ),
            patch(
                "thermotwin.studies.operating_decision_prospective."
                "forecast_realistic_margin_interval",
                side_effect=_interval_for_fit,
            ),
            self.assertRaisesRegex(ValueError, "digest is invalid"),
        ):
            prospective._validate_acquisition_evidence(tampered, self.config)

    def test_validation_reauthenticates_fit_provenance_not_just_its_digest(self):
        evidence = _evidence(self.config)
        forged_fits = list(evidence.fit_set.fits)
        forged_fits[0] = forged_fits[0]._replace(
            objective=forged_fits[0].objective + 0.25
        )
        forged_fit_set = RealisticAcquisitionFitSet(tuple(forged_fits), ())
        with patch(
            "thermotwin.studies.operating_decision_prospective."
            "forecast_realistic_margin_interval",
            side_effect=_interval_for_fit,
        ):
            forged_snapshot = prospective.build_prospective_acquisition_snapshot(
                forged_fit_set,
                evidence.final_regime,
                self.config,
            )
        forged_digest = prospective._acquisition_evidence_digest(
            evidence.common_initial_run,
            forged_fit_set,
            forged_snapshot,
            evidence.final_regime,
            self.config,
        )
        forged = replace(
            evidence,
            fit_set=forged_fit_set,
            snapshot=forged_snapshot,
            evidence_digest=forged_digest,
        )
        with (
            patch.object(
                prospective,
                "_multistart_fit_candidates",
                return_value=evidence.fit_set,
            ),
            patch(
                "thermotwin.studies.operating_decision_prospective."
                "forecast_realistic_margin_interval",
                side_effect=_interval_for_fit,
            ),
            self.assertRaises(ValueError),
        ):
            prospective._validate_acquisition_evidence(forged, self.config)

    def test_initial_fit_failure_is_retained_as_serializable_evidence(self):
        fit_set = RealisticAcquisitionFitSet(
            (_fit(FOUR_STATE_MODEL, self.config),),
            (
                NumericalFailure(
                    FIVE_STATE_MODEL,
                    "acquisition_fit",
                    "ValueError",
                ),
            ),
        )
        with patch.object(
            prospective,
            "_multistart_fit_candidates",
            return_value=fit_set,
        ):
            evidence = prospective.prepare_prospective_acquisition_evidence(
                _initial_run(self.config),
                _final_regime(self.config),
                self.config,
            )
            payload = prospective.prospective_acquisition_evidence_payload(
                evidence,
                self.config,
            )
        self.assertEqual(
            evidence.snapshot.selection_failure_reason,
            "acquisition_fit_failure",
        )
        self.assertEqual(
            payload["snapshot"]["selection_failure_reason"],
            "acquisition_fit_failure",
        )
        json.dumps(payload, allow_nan=False, sort_keys=True)

    def test_multistart_runs_all_frozen_starts_and_selects_best_objective(self):
        run = _initial_run(self.config)

        def starts(spec):
            return tuple((float(index),) * len(spec.names) for index in range(3))

        objectives = {
            FOUR_STATE_MODEL: (3.0, 1.0, 2.0),
            FIVE_STATE_MODEL: (5.0, None, 4.0),
        }

        def fit_candidate(model_name, _runs, _config, *, initial_log_multipliers):
            index = int(initial_log_multipliers[0])
            objective = objectives[model_name][index]
            if objective is None:
                raise ValueError("disposable failed start")
            return _fit(model_name, self.config, objective=objective)

        with (
            patch.object(prospective, "_fit_start_vectors", side_effect=starts),
            patch.object(
                prospective,
                "fit_realistic_candidate",
                side_effect=fit_candidate,
            ) as fit_call,
        ):
            fit_set = prospective._multistart_fit_candidates((run,), self.config)

        self.assertEqual(fit_call.call_count, 6)
        self.assertEqual(
            {fit.model_name: fit.objective for fit in fit_set.fits},
            {FOUR_STATE_MODEL: 1.0, FIVE_STATE_MODEL: 4.0},
        )
        self.assertEqual(fit_set.failures, ())


class ProspectiveUncertaintyScoringTests(unittest.TestCase):
    def setUp(self):
        self.config = OperatingDecisionRealismConfig()
        self.evidence = _evidence(self.config)
        self.namespace = ProspectiveRandomStreamNamespace(
            campaign="prospective-disposable-2026-09",
            partition="step2-development",
            block=3,
            acquisition_evidence_digest=self.evidence.evidence_digest,
        )

    def _mocked_result(self, *, draw_count=1):
        scoring = prospective.ProspectiveUncertaintyConfig(
            draw_count=draw_count,
            minimum_stable_fraction=1.0,
        )

        def parameter_draw(fit, namespace, draw_index, _physical, _scoring, registry):
            _register_parameter_use(fit, namespace, draw_index, registry)
            return fit.log_multipliers

        def probe_draw(namespace, model_name, draw_index, _physical, _scoring, registry):
            _register_probe_use(namespace, model_name, draw_index, registry)
            return (0.0, 0.0)

        def evaluate(**kwargs):
            _register_observation_uses(kwargs)
            return _completed_draw(
                kwargs["policy"].name,
                kwargs["generator_fit"].model_name,
                kwargs["draw_index"],
                0.5,
                generator_offsets=kwargs["generator_offsets"],
            )

        with (
            patch.object(prospective, "_validate_acquisition_evidence"),
            patch.object(
                prospective,
                "_generator_parameter_draw",
                side_effect=parameter_draw,
            ),
            patch.object(
                prospective,
                "_face_probe_draw",
                side_effect=probe_draw,
            ),
            patch.object(
                prospective,
                "_evaluate_hypothetical_draw",
                side_effect=evaluate,
            ),
        ):
            return prospective.estimate_prospective_action_uncertainty(
                self.evidence,
                self.namespace,
                self.config,
                scoring,
            )

    def test_action_simulation_uses_exact_packages_and_loads_only_face_action(self):
        generator_fit = next(
            fit
            for fit in self.evidence.fit_set.fits
            if fit.model_name == FOUR_STATE_MODEL
        )
        policies = {
            item.name: item for item in default_fixed_policies() if item.additional_regimes
        }

        def synthetic_run(**kwargs):
            regime = kwargs["regime"]
            values = tuple(
                ObservableValue(channel, 0.0, 300.0)
                for channel in regime.channels
            )
            return RealisticOperatingRun(
                regime,
                ObservableRun(regime.name, regime.current, values),
                RunInstrumentation(COLD_FACE in regime.channels),
            )

        for policy_name in (FIXED_THERMAL, FIXED_VOLTAGE, FIXED_FACE_TEMPERATURE):
            with self.subTest(policy_name=policy_name):
                simulate = Mock(return_value=object())
                with (
                    patch.object(
                        prospective,
                        "_simulate_realistic_observables",
                        simulate,
                    ),
                    patch.object(
                        prospective,
                        "_synthetic_observed_run",
                        side_effect=synthetic_run,
                    ),
                ):
                    runs = prospective._simulate_action_runs(
                        policies[policy_name],
                        generator_fit,
                        generator_fit.log_multipliers,
                        (0.1, -0.1)
                        if policy_name == FIXED_FACE_TEMPERATURE
                        else None,
                        0,
                        self.namespace,
                        self.config,
                        ProspectiveRandomStreamRegistry(),
                    )
                self.assertEqual(
                    tuple(run.regime for run in runs),
                    policies[policy_name].additional_regimes,
                )
                self.assertEqual(
                    simulate.call_count,
                    len(policies[policy_name].additional_regimes),
                )
                for call in simulate.call_args_list:
                    face_sensor = call.args[6]
                    if policy_name == FIXED_FACE_TEMPERATURE:
                        self.assertIsNotNone(face_sensor)
                    else:
                        self.assertIsNone(face_sensor)

    def test_new_candidate_attrition_cannot_create_positive_information_value(self):
        policy = next(
            item for item in default_fixed_policies() if item.name == FIXED_VOLTAGE
        )
        generator_fit = next(
            fit
            for fit in self.evidence.fit_set.fits
            if fit.model_name == FOUR_STATE_MODEL
        )
        refits = RealisticAcquisitionFitSet(
            tuple(
                _fit(
                    model_name,
                    self.config,
                    reached_bound=model_name == FOUR_STATE_MODEL,
                )
                for model_name in MODEL_NAMES
            ),
            (),
        )
        surviving_interval = _interval(FIVE_STATE_MODEL, -0.1, 0.1)
        baseline = (
            self.evidence.snapshot.provisional_margin_envelope.upper
            - self.evidence.snapshot.provisional_margin_envelope.lower
        )
        with (
            patch.object(prospective, "_simulate_action_runs", return_value=()),
            patch.object(
                prospective,
                "_multistart_fit_candidates",
                return_value=refits,
            ),
            patch.object(
                prospective,
                "forecast_realistic_margin_interval",
                return_value=surviving_interval,
            ),
        ):
            outcome = prospective._evaluate_hypothetical_draw(
                policy=policy,
                generator_fit=generator_fit,
                generator_offsets=generator_fit.log_multipliers,
                face_probe_offsets=None,
                draw_index=0,
                evidence=self.evidence,
                namespace=self.namespace,
                physical_config=self.config,
                registry=ProspectiveRandomStreamRegistry(),
                baseline_width=baseline,
            )
        self.assertFalse(outcome.failed)
        self.assertFalse(outcome.stable)
        self.assertEqual(
            outcome.initially_admissible_became_inadmissible,
            (FOUR_STATE_MODEL,),
        )
        self.assertEqual(outcome.raw_after_width, 0.2)
        self.assertEqual(outcome.scored_after_width, baseline)

    def test_failed_refit_is_retained_and_imputed_to_the_common_baseline(self):
        policy = next(
            item for item in default_fixed_policies() if item.name == FIXED_THERMAL
        )
        generator_fit = self.evidence.fit_set.fits[0]
        surviving_model = next(
            model for model in MODEL_NAMES if model != FOUR_STATE_MODEL
        )
        refits = RealisticAcquisitionFitSet(
            (_fit(surviving_model, self.config),),
            (NumericalFailure(FOUR_STATE_MODEL, "acquisition_fit", "ValueError"),),
        )
        baseline = (
            self.evidence.snapshot.provisional_margin_envelope.upper
            - self.evidence.snapshot.provisional_margin_envelope.lower
        )
        with (
            patch.object(prospective, "_simulate_action_runs", return_value=()),
            patch.object(
                prospective,
                "_multistart_fit_candidates",
                return_value=refits,
            ),
            patch.object(
                prospective,
                "forecast_realistic_margin_interval",
                return_value=_interval(surviving_model, -0.1, 0.1),
            ),
        ):
            outcome = prospective._evaluate_hypothetical_draw(
                policy=policy,
                generator_fit=generator_fit,
                generator_offsets=generator_fit.log_multipliers,
                face_probe_offsets=None,
                draw_index=0,
                evidence=self.evidence,
                namespace=self.namespace,
                physical_config=self.config,
                registry=ProspectiveRandomStreamRegistry(),
                baseline_width=baseline,
            )
        self.assertTrue(outcome.failed)
        self.assertFalse(outcome.stable)
        self.assertEqual(outcome.failure_stage, "prospective_refit")
        self.assertEqual(outcome.scored_after_width, baseline)
        self.assertIsNone(outcome.raw_after_width)

    def test_recovered_candidate_reenters_the_after_action_envelope(self):
        evidence = _evidence(self.config, excluded_model=FIVE_STATE_MODEL)
        namespace = replace(
            self.namespace,
            acquisition_evidence_digest=evidence.evidence_digest,
        )
        policy = next(
            item for item in default_fixed_policies() if item.name == FIXED_VOLTAGE
        )
        generator_fit = next(
            fit for fit in evidence.fit_set.fits if fit.model_name == FOUR_STATE_MODEL
        )
        refits = RealisticAcquisitionFitSet(
            tuple(_fit(model_name, self.config) for model_name in MODEL_NAMES),
            (),
        )

        def after_interval(fit, *_args):
            if fit.model_name == FOUR_STATE_MODEL:
                return _interval(FOUR_STATE_MODEL, -0.25, 0.25)
            return _interval(FIVE_STATE_MODEL, 0.5, 1.0)

        with (
            patch.object(prospective, "_simulate_action_runs", return_value=()),
            patch.object(
                prospective,
                "_multistart_fit_candidates",
                return_value=refits,
            ),
            patch.object(
                prospective,
                "forecast_realistic_margin_interval",
                side_effect=after_interval,
            ),
        ):
            outcome = prospective._evaluate_hypothetical_draw(
                policy=policy,
                generator_fit=generator_fit,
                generator_offsets=generator_fit.log_multipliers,
                face_probe_offsets=None,
                draw_index=0,
                evidence=evidence,
                namespace=namespace,
                physical_config=self.config,
                registry=ProspectiveRandomStreamRegistry(),
                baseline_width=1.25,
            )
        self.assertEqual(
            outcome.initially_excluded_became_admissible,
            (FIVE_STATE_MODEL,),
        )
        self.assertTrue(outcome.stable)
        self.assertEqual(outcome.raw_after_width, 1.25)

    def test_aggregation_uses_retained_denominator_worst_case_and_per_source_floor(self):
        baseline = 2.0
        draws = []
        source_models = tuple(sorted(MODEL_NAMES))
        for policy_name in (FIXED_THERMAL, FIXED_VOLTAGE, FIXED_FACE_TEMPERATURE):
            draws.extend(
                (
                    _completed_draw(policy_name, source_models[0], 0, 0.2),
                    _completed_draw(policy_name, source_models[0], 1, 0.4),
                    _completed_draw(policy_name, source_models[1], 0, 0.8),
                    prospective._failed_draw(
                        policy_name=policy_name,
                        generator_model=source_models[1],
                        draw_index=1,
                        generator_offsets=(0.0,),
                        face_probe_offsets=(0.0, 0.0)
                        if policy_name == FIXED_FACE_TEMPERATURE
                        else None,
                        baseline_width=baseline,
                        stage="prospective_refit",
                        error_type="ValueError",
                    ),
                )
            )
        draws.sort(
            key=lambda item: (
                item.generator_model,
                item.policy_name,
                item.draw_index,
            )
        )

        actions = prospective._summarize_uncertainty_draws(
            draws,
            baseline,
            source_models,
            prospective.ProspectiveUncertaintyConfig(
                draw_count=2,
                minimum_stable_fraction=0.5,
            ),
        )
        self.assertEqual(tuple(item.policy_name for item in actions), POLICY_NAMES)
        for action in actions[1:]:
            self.assertTrue(action.eligible)
            self.assertEqual(action.prospective_draw_count, 4)
            self.assertAlmostEqual(action.source_summaries[0].mean_after_width, 0.3)
            self.assertAlmostEqual(action.source_summaries[1].mean_after_width, 1.4)
            self.assertAlmostEqual(action.expected_uncertainty_after, 1.4)

        stricter = prospective._summarize_uncertainty_draws(
            draws,
            baseline,
            source_models,
            prospective.ProspectiveUncertaintyConfig(
                draw_count=2,
                minimum_stable_fraction=0.75,
            ),
        )
        for action in stricter[1:]:
            self.assertFalse(action.eligible)
            self.assertEqual(
                action.failure_reason,
                "insufficient_stable_prospective_draws",
            )

    def test_aggregation_rejects_draws_outside_declared_source_matrix(self):
        draws = [
            _completed_draw(policy_name, FOUR_STATE_MODEL, 0, 0.5)
            for policy_name in (
                FIXED_THERMAL,
                FIXED_VOLTAGE,
                FIXED_FACE_TEMPERATURE,
            )
        ]
        draws.append(
            _completed_draw(FIXED_VOLTAGE, FIVE_STATE_MODEL, 0, 0.5)
        )
        draws.sort(
            key=lambda item: (
                item.generator_model,
                item.policy_name,
                item.draw_index,
            )
        )
        with self.assertRaisesRegex(ValueError, "draw matrix"):
            prospective._summarize_uncertainty_draws(
                draws,
                2.0,
                (FOUR_STATE_MODEL,),
                prospective.ProspectiveUncertaintyConfig(draw_count=1),
            )

    def test_estimator_pairs_one_parameter_draw_across_all_three_actions(self):
        config = prospective.ProspectiveUncertaintyConfig(
            draw_count=2,
            minimum_stable_fraction=1.0,
        )
        parameter_calls = []
        probe_calls = []
        evaluation_calls = []

        def parameter_draw(fit, namespace, draw_index, _physical, _scoring, registry):
            _register_parameter_use(fit, namespace, draw_index, registry)
            value = tuple(0.01 * draw_index for _ in fit.log_multipliers)
            parameter_calls.append((fit.model_name, draw_index, value))
            return value

        def probe_draw(namespace, model_name, draw_index, _physical, _scoring, registry):
            _register_probe_use(namespace, model_name, draw_index, registry)
            value = (0.1 + 0.05 * draw_index, -0.1 - 0.05 * draw_index)
            probe_calls.append((model_name, draw_index, value))
            return value

        def evaluate(**kwargs):
            _register_observation_uses(kwargs)
            policy_name = kwargs["policy"].name
            evaluation_calls.append(
                (
                    kwargs["generator_fit"].model_name,
                    kwargs["draw_index"],
                    policy_name,
                    tuple(kwargs["generator_offsets"]),
                    kwargs["face_probe_offsets"],
                )
            )
            return _completed_draw(
                policy_name,
                kwargs["generator_fit"].model_name,
                kwargs["draw_index"],
                0.5,
                generator_offsets=kwargs["generator_offsets"],
            )

        with (
            patch.object(prospective, "_validate_acquisition_evidence"),
            patch.object(
                prospective,
                "_generator_parameter_draw",
                side_effect=parameter_draw,
            ),
            patch.object(
                prospective,
                "_face_probe_draw",
                side_effect=probe_draw,
            ),
            patch.object(
                prospective,
                "_evaluate_hypothetical_draw",
                side_effect=evaluate,
            ),
        ):
            result = prospective.estimate_prospective_action_uncertainty(
                self.evidence,
                self.namespace,
                self.config,
                config,
            )

        source_count = len(self.evidence.snapshot.admissible_candidate_models)
        self.assertEqual(len(parameter_calls), source_count * config.draw_count)
        self.assertEqual(len(probe_calls), source_count * config.draw_count)
        self.assertEqual(
            len(evaluation_calls),
            source_count * config.draw_count * 3,
        )
        for model_name, draw_index, value in parameter_calls:
            selected = [
                item
                for item in evaluation_calls
                if item[0] == model_name and item[1] == draw_index
            ]
            self.assertEqual(len(selected), 3)
            self.assertEqual({item[3] for item in selected}, {value})
            face = next(
                item for item in selected if item[2] == FIXED_FACE_TEMPERATURE
            )
            nonface = tuple(
                item for item in selected if item[2] != FIXED_FACE_TEMPERATURE
            )
            expected_probe = next(
                item[2]
                for item in probe_calls
                if item[0] == model_name and item[1] == draw_index
            )
            self.assertEqual(face[4], expected_probe)
            self.assertTrue(all(item[4] is None for item in nonface))
        self.assertEqual(
            tuple(item.policy_name for item in result.action_uncertainties),
            POLICY_NAMES,
        )
        self.assertTrue(result.stream_audit.clean)

    def test_parameter_sampler_exhaustion_imputes_all_actions_for_that_draw(self):
        evidence = _evidence(self.config, excluded_model=FIVE_STATE_MODEL)
        namespace = replace(
            self.namespace,
            acquisition_evidence_digest=evidence.evidence_digest,
        )

        def exhausted_parameter(
            fit,
            stream_namespace,
            draw_index,
            _physical,
            _scoring,
            registry,
        ):
            _register_parameter_use(
                fit,
                stream_namespace,
                draw_index,
                registry,
            )
            raise prospective._ProspectiveDrawSamplingError(
                "bounded prospective parameter sampler exhausted its attempts"
            )

        with (
            patch.object(prospective, "_validate_acquisition_evidence"),
            patch.object(
                prospective,
                "_generator_parameter_draw",
                side_effect=exhausted_parameter,
            ),
            patch.object(prospective, "_face_probe_draw") as probe,
            patch.object(prospective, "_evaluate_hypothetical_draw") as evaluate,
        ):
            result = prospective.estimate_prospective_action_uncertainty(
                evidence,
                namespace,
                self.config,
                prospective.ProspectiveUncertaintyConfig(
                    draw_count=1,
                    minimum_stable_fraction=1.0,
                ),
            )
        probe.assert_not_called()
        evaluate.assert_not_called()
        self.assertEqual(len(result.draw_outcomes), 3)
        self.assertEqual(
            {item.policy_name for item in result.draw_outcomes},
            {FIXED_THERMAL, FIXED_VOLTAGE, FIXED_FACE_TEMPERATURE},
        )
        self.assertTrue(all(item.failed for item in result.draw_outcomes))
        self.assertTrue(
            all(
                item.failure_stage == "prospective_parameter_draw"
                for item in result.draw_outcomes
            )
        )

    def test_probe_sampler_exhaustion_fails_only_the_face_action(self):
        evidence = _evidence(self.config, excluded_model=FIVE_STATE_MODEL)
        namespace = replace(
            self.namespace,
            acquisition_evidence_digest=evidence.evidence_digest,
        )
        evaluated = []

        def evaluate(**kwargs):
            _register_observation_uses(kwargs)
            policy_name = kwargs["policy"].name
            evaluated.append(policy_name)
            return prospective.ProspectiveDrawOutcome(
                policy_name=policy_name,
                generator_model=kwargs["generator_fit"].model_name,
                draw_index=kwargs["draw_index"],
                generator_log_offsets=tuple(kwargs["generator_offsets"]),
                face_probe_log_offsets=None,
                candidate_outcomes=tuple(
                    sorted(
                        (
                            prospective.ProspectiveCandidateOutcome(
                                FOUR_STATE_MODEL,
                                "admissible",
                                1.0,
                                _interval(FOUR_STATE_MODEL, -0.25, 0.25),
                            ),
                            prospective.ProspectiveCandidateOutcome(
                                FIVE_STATE_MODEL,
                                "fit_reached_bound",
                                1.0,
                                None,
                            ),
                        ),
                        key=lambda item: item.model_name,
                    )
                ),
                initially_admissible_became_inadmissible=(),
                initially_excluded_became_admissible=(),
                stable=True,
                failed=False,
                failure_stage=None,
                failure_model=None,
                failure_type=None,
                raw_after_width=0.5,
                scored_after_width=0.5,
                synthetic_observation_digest="a" * 64,
            )

        def parameter_draw(
            fit,
            stream_namespace,
            draw_index,
            _physical,
            _scoring,
            registry,
        ):
            _register_parameter_use(
                fit,
                stream_namespace,
                draw_index,
                registry,
            )
            return fit.log_multipliers

        def exhausted_probe(
            stream_namespace,
            model_name,
            draw_index,
            _physical,
            _scoring,
            registry,
        ):
            _register_probe_use(
                stream_namespace,
                model_name,
                draw_index,
                registry,
            )
            raise prospective._ProspectiveDrawSamplingError(
                "bounded prospective parameter sampler exhausted its attempts"
            )

        with (
            patch.object(prospective, "_validate_acquisition_evidence"),
            patch.object(
                prospective,
                "_generator_parameter_draw",
                side_effect=parameter_draw,
            ),
            patch.object(
                prospective,
                "_face_probe_draw",
                side_effect=exhausted_probe,
            ),
            patch.object(
                prospective,
                "_evaluate_hypothetical_draw",
                side_effect=evaluate,
            ),
        ):
            result = prospective.estimate_prospective_action_uncertainty(
                evidence,
                namespace,
                self.config,
                prospective.ProspectiveUncertaintyConfig(
                    draw_count=1,
                    minimum_stable_fraction=1.0,
                ),
            )
        self.assertEqual(set(evaluated), {FIXED_THERMAL, FIXED_VOLTAGE})
        outcomes = {item.policy_name: item for item in result.draw_outcomes}
        self.assertFalse(outcomes[FIXED_THERMAL].failed)
        self.assertFalse(outcomes[FIXED_VOLTAGE].failed)
        self.assertTrue(outcomes[FIXED_FACE_TEMPERATURE].failed)
        self.assertEqual(
            outcomes[FIXED_FACE_TEMPERATURE].failure_stage,
            "prospective_probe_draw",
        )

    def test_structurally_invalid_generator_covariance_aborts_scorecard(self):
        evidence = _evidence(self.config, excluded_model=FIVE_STATE_MODEL)
        fits = list(evidence.fit_set.fits)
        source_index = next(
            index
            for index, fit in enumerate(fits)
            if fit.model_name == FOUR_STATE_MODEL
        )
        size = len(fits[source_index].covariance)
        invalid = tuple(
            tuple(
                -1.0 if row == column == 0 else (1.0 if row == column else 0.0)
                for column in range(size)
            )
            for row in range(size)
        )
        fits[source_index] = fits[source_index]._replace(covariance=invalid)
        forged = replace(
            evidence,
            fit_set=RealisticAcquisitionFitSet(tuple(fits), ()),
        )
        namespace = replace(
            self.namespace,
            acquisition_evidence_digest=forged.evidence_digest,
        )
        with (
            patch.object(prospective, "_validate_acquisition_evidence"),
            self.assertRaisesRegex(ValueError, "positive semidefinite"),
        ):
            prospective.estimate_prospective_action_uncertainty(
                forged,
                namespace,
                self.config,
                prospective.ProspectiveUncertaintyConfig(draw_count=1),
            )

    def test_result_payload_rejects_config_not_bound_to_protocol_digest(self):
        result = self._mocked_result(draw_count=1)
        with self.assertRaises(ValueError):
            replace(
                result,
                config=replace(result.config, draw_count=2),
            )

    def test_result_rejects_missing_or_foreign_stream_manifest(self):
        result = self._mocked_result(draw_count=1)
        empty_audit = ProspectiveRandomStreamRegistry().audit()
        with self.assertRaisesRegex(ValueError, "random-stream uses"):
            replace(
                result,
                stream_uses=(),
                stream_audit=empty_audit,
            )

        first = result.stream_uses[0]
        foreign_key = replace(
            first.stream.key,
            acquisition_evidence_digest="b" * 64,
        )
        foreign_use = replace(
            first,
            stream=replace(first.stream, key=foreign_key),
        )
        foreign_uses = (foreign_use, *result.stream_uses[1:])
        registry = ProspectiveRandomStreamRegistry()
        for use in foreign_uses:
            registry.register(
                use.stream,
                consumer=use.consumer,
                shared_for_action=use.shared_for_action,
            )
        with self.assertRaisesRegex(ValueError, "one namespace"):
            replace(
                result,
                stream_uses=foreign_uses,
                stream_audit=registry.audit(),
            )

    def test_result_rederives_width_from_candidate_intervals(self):
        result = self._mocked_result(draw_count=1)
        draw = result.draw_outcomes[0]
        baseline = (
            result.acquisition_evidence.snapshot.provisional_margin_envelope.upper
            - result.acquisition_evidence.snapshot.provisional_margin_envelope.lower
        )
        with self.assertRaisesRegex(ValueError, "candidate outcomes"):
            prospective._validate_draw_against_acquisition(
                replace(draw, raw_after_width=0.01, scored_after_width=0.01),
                result.acquisition_evidence.snapshot,
                baseline,
            )

    def test_protocol_and_result_digests_bind_config_physics_and_contents(self):
        scoring = prospective.ProspectiveUncertaintyConfig(draw_count=2)
        digest = prospective.prospective_uncertainty_protocol_digest(
            self.config,
            scoring,
        )
        self.assertNotEqual(
            digest,
            prospective.prospective_uncertainty_protocol_digest(
                self.config,
                replace(scoring, draw_count=3),
            ),
        )
        changed_physics = replace(
            self.config,
            face_sensor_prior_log_standard_deviation=(
                self.config.face_sensor_prior_log_standard_deviation + 0.01
            ),
        )
        self.assertNotEqual(
            digest,
            prospective.prospective_uncertainty_protocol_digest(
                changed_physics,
                scoring,
            ),
        )

        result = self._mocked_result(draw_count=1)
        with patch.object(prospective, "_validate_acquisition_evidence"):
            payload = prospective.prospective_uncertainty_result_payload(result)
        json.dumps(payload, allow_nan=False, sort_keys=True)
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("declared_cost", rendered)
        self.assertNotIn("utility_per_cost", rendered)
        with (
            patch.object(prospective, "_validate_acquisition_evidence"),
            self.assertRaisesRegex(ValueError, "result digest is invalid"),
        ):
            prospective.prospective_uncertainty_result_payload(
                replace(result, result_digest="0" * 64)
            )


if __name__ == "__main__":
    unittest.main()
