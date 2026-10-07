from copy import deepcopy
import json
import math
import unittest

from scipy.stats import binomtest

from thermotwin.studies.operating_decision import APPROVE, INSUFFICIENT_EVIDENCE, REJECT
from thermotwin.studies.operating_decision_realism import STAGE3_TRUTH_CONDITIONS
from thermotwin.studies.operating_decision_prospective_phase_e import (
    phase_e_protocol_payload, validate_phase_e_protocol,
)
from thermotwin.studies.operating_decision_prospective_phase_e_calibration import (
    PHASE_E_PROCEDURES, POSITIVE_INFINITY, SELECTOR_PROCEDURE,
    apply_calibrated_decision, calibrated_outcome, calibrate_complete_procedures,
    case_nonconformity, decode_score, tolerance_rank, validate_calibration_result,
)
from thermotwin.studies.operating_decision_prospective_phase_e_analysis import (
    BOUND_ALPHA, analyze_primary_comparison, binomial_bounds,
    zero_error_approval_requirement,
)


def fixture_records(n=100):
    return [
        {"block": block, "truth_family": family, "procedure": procedure,
         "selected_action": "stop_now", "raw_interval": {"lower": -1., "upper": 1.},
         "development_offset_kelvin": 0., "true_margin": 1. + block / 100.,
         "verification_succeeded": True, "pipeline_failure": False,
         "realized_resources": {"total_diagnostic_energy": 60., "diagnostic_run_count": 2,
                                "extra_sensor_count": 0, "energized_schedule_time_seconds": 160.}}
        for block in range(n) for family in STAGE3_TRUTH_CONDITIONS
        for procedure in PHASE_E_PROCEDURES
    ]


class PhaseECalibrationTests(unittest.TestCase):
    def test_tolerance_rank_matches_independent_binomial_sum(self):
        self.assertEqual(tolerance_rank(100), 96)
        self.assertEqual(tolerance_rank(10), 11)
        for n in (29, 50, 100):
            rank = tolerance_rank(n)
            def confidence(k):
                return sum(math.comb(n, j) * .9**j * .1**(n-j) for j in range(min(k, n+1)))
            self.assertGreaterEqual(confidence(rank), .95)
            self.assertLess(confidence(rank-1), .95)

    def test_action_offset_is_applied_once(self):
        row = fixture_records(1)[0]
        row.update(raw_interval={"lower": 1., "upper": 2.}, true_margin=2.5,
                   development_offset_kelvin=.2)
        self.assertAlmostEqual(case_nonconformity(row), .3)

    def test_finite_score_corresponds_exactly_to_final_coverage(self):
        rows = fixture_records(5)
        for row in rows:
            # Exactly representable distances test the mathematical boundary
            # without comparing two different roundings of a decimal constant.
            row["true_margin"] = 1. + row["block"] / 32.
            for q in (0., .03125, .0625, .5):
                result = calibrated_outcome(row, q)
                self.assertEqual(result["interval_covered"], case_nonconformity(row) <= q)

    def test_required_score_never_rounds_down_at_cancellation_boundary(self):
        row = fixture_records(1)[0]
        row.update(raw_interval={"lower": 1., "upper": 2.}, true_margin=-1e-14)
        naive = max(0., 1. - row["true_margin"], row["true_margin"] - 2.)
        self.assertFalse(calibrated_outcome(row, naive)["interval_covered"])
        q = case_nonconformity(row)
        self.assertGreater(q, naive)
        self.assertTrue(calibrated_outcome(row, q)["interval_covered"])
        for margin in (-1e-20, -1e-14, -.25, 1., 1.5, 2., 2.1):
            row["true_margin"] = margin
            self.assertTrue(calibrated_outcome(row, case_nonconformity(row))["interval_covered"])

    def test_failures_are_infinite_and_cannot_become_valid_decisions(self):
        base = fixture_records(1)[0]
        variants = [dict(base, raw_interval=None), dict(base, verification_succeeded=False),
                    dict(base, pipeline_failure=True), dict(base, raw_interval={"lower": 2., "upper": 1.}),
                    dict(base, raw_interval={"lower": None, "upper": 1.}),
                    dict(base, raw_interval={"lower": -math.inf, "upper": 1.})]
        for row in variants:
            self.assertEqual(case_nonconformity(row), math.inf)
            out = calibrated_outcome(row, 100.)
            self.assertEqual(out["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(out["final_interval"])
            self.assertFalse(out["interval_covered"])

    def test_unknown_truth_is_an_incident_not_a_dropped_failure(self):
        with self.assertRaises(ValueError):
            case_nonconformity(dict(fixture_records(1)[0], true_margin=None))

    def test_block_maxima_rank_and_denominator(self):
        rows = fixture_records()
        # One family's failure invalidates the whole block for only its procedure.
        for row in rows:
            if row["procedure"] == SELECTOR_PROCEDURE and row["block"] >= 96 and row["truth_family"] == STAGE3_TRUTH_CONDITIONS[0]:
                row["raw_interval"] = None
        result = calibrate_complete_procedures(rows)
        self.assertEqual(result["order_statistic_rank"], 96)
        self.assertTrue(result["all_required_corrections_finite"])
        selected = result["procedures"][SELECTOR_PROCEDURE]
        self.assertEqual(len(selected["block_scores"]), 100)
        self.assertEqual(selected["infinite_block_count"], 4)
        self.assertAlmostEqual(selected["correction_kelvin"], .95)
        for row in rows:
            if row["procedure"] == SELECTOR_PROCEDURE and row["block"] == 95:
                row["verification_succeeded"] = False
        result = calibrate_complete_procedures(rows)
        self.assertEqual(result["procedures"][SELECTOR_PROCEDURE]["correction_kelvin"], POSITIVE_INFINITY)
        self.assertFalse(result["all_required_corrections_finite"])
        self.assertFalse(result["reserved_generation_authorized"])

    def test_procedures_have_separate_corrections_and_same_rank(self):
        rows = fixture_records()
        for row in rows:
            if row["procedure"] == SELECTOR_PROCEDURE:
                row["development_offset_kelvin"] = .2
        result = calibrate_complete_procedures(rows)
        self.assertAlmostEqual(result["procedures"][SELECTOR_PROCEDURE]["correction_kelvin"], .75)
        self.assertAlmostEqual(result["procedures"]["stop_now"]["correction_kelvin"], .95)
        self.assertFalse(result["confidence_is_joint_over_procedures"])

    def test_json_roundtrip_recomputation_and_tampering(self):
        rows = fixture_records()
        result = calibrate_complete_procedures(rows)
        restored = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(validate_calibration_result(restored, rows), result)
        restored["procedures"][SELECTOR_PROCEDURE]["correction_kelvin"] = 0.
        with self.assertRaises(ValueError):
            validate_calibration_result(restored, rows)

    def test_missing_duplicate_foreign_and_boolean_block_rejected(self):
        rows = fixture_records(2)
        for invalid in (rows[:-1], rows + [rows[0]], [dict(rows[0], block=True), *rows[1:]],
                        [dict(rows[0], procedure="foreign"), *rows[1:]]):
            with self.assertRaises(ValueError):
                calibrate_complete_procedures(invalid, block_count=2)

    def test_truth_free_final_decision_and_boundaries(self):
        row = {"raw_interval": {"lower": .1, "upper": .5}, "development_offset_kelvin": .1,
               "verification_succeeded": True, "pipeline_failure": False}
        self.assertEqual(apply_calibrated_decision(row, 0.)["decision"], APPROVE)
        self.assertEqual(apply_calibrated_decision(row, .001)["decision"], INSUFFICIENT_EVIDENCE)
        row["raw_interval"] = {"lower": -.5, "upper": -.1}
        self.assertEqual(apply_calibrated_decision(row, 0.)["decision"], INSUFFICIENT_EVIDENCE)
        row["development_offset_kelvin"] = .09
        self.assertEqual(apply_calibrated_decision(row, 0.)["decision"], REJECT)
        with self.assertRaises(ValueError):
            apply_calibrated_decision(dict(row, true_margin=.2), 0.)
        with self.assertRaises(ValueError):
            apply_calibrated_decision(dict(row, truth_family="hidden"), 0.)

    def test_infinite_q_and_secondary_clearance_abstain(self):
        row = dict(fixture_records(1)[0], raw_interval={"lower": .05, "upper": .2}, true_margin=.1)
        self.assertEqual(calibrated_outcome(row, POSITIVE_INFINITY)["decision"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(calibrated_outcome(row, 0., decision_clearance_kelvin=.1)["decision"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(row["selected_action"], "stop_now")


class PhaseEAnalysisTests(unittest.TestCase):
    def outcomes(self, n=100):
        rows = fixture_records(n)
        for row in rows:
            row["true_margin"] = .5 if row["block"] % 2 else -.5
            row["raw_interval"] = {"lower": row["true_margin"]-.1, "upper": row["true_margin"]+.1}
        return [calibrated_outcome(row, 0.) for row in rows]

    def test_exact_bounds_independently_match_scipy_binomial_test(self):
        for successes, trials in ((0, 33), (3, 50), (50, 100), (100, 100)):
            ours = binomial_bounds(successes, trials, alpha=.025)
            ref = binomtest(successes, trials).proportion_ci(confidence_level=.95, method="exact")
            self.assertAlmostEqual(ours["lower"], ref.low, places=10)
            self.assertAlmostEqual(ours["upper"], ref.high, places=10)
        self.assertGreater(binomial_bounds(0, 33)["upper"], 0.)
        self.assertIsNone(binomial_bounds(0, 0)["upper"])

    def test_precision_zero_error_sample_requirements(self):
        for risk in (.1, .02):
            n = zero_error_approval_requirement(risk)
            self.assertLessEqual(1 - BOUND_ALPHA**(1/n), risk)
            self.assertGreater(1 - BOUND_ALPHA**(1/(n-1)), risk)

    def test_no_energy_saving_against_qualifying_stop_prevents_success(self):
        result = analyze_primary_comparison(self.outcomes(), block_count=100)
        self.assertFalse(result["primary_success"])
        self.assertEqual(result["comparisons"]["stop_now"]["energy"]["mean_saving_joules"], 0.)
        self.assertEqual(result["independent_block_count"], 100)
        self.assertEqual(result["procedures"][SELECTOR_PROCEDURE]["overall"]["case_count"], 300)
        # Zero errors never produce a zero risk upper bound.
        self.assertGreater(result["comparisons"]["stop_now"]["risk_increase_upper"], 0.)

    def test_missing_energy_keeps_denominator_and_prohibits_energy_claim(self):
        rows = self.outcomes()
        rows[0]["realized_resources"]["total_diagnostic_energy"] = None
        result = analyze_primary_comparison(rows, block_count=100)
        self.assertFalse(result["comparisons"]["stop_now"]["energy"]["complete"])
        self.assertEqual(result["comparisons"]["stop_now"]["energy"]["paired_blocks"], 100)
        self.assertFalse(result["primary_success"])

    def test_empty_approval_groups_are_na_and_not_qualifying(self):
        rows = self.outcomes()
        for i, row in enumerate(rows):
            if row["procedure"] == SELECTOR_PROCEDURE:
                rows[i] = calibrated_outcome(dict(row, raw_interval={"lower": -1., "upper": 1.}), 0.)
        result = analyze_primary_comparison(rows, block_count=100)
        selected = result["procedures"][SELECTOR_PROCEDURE]
        self.assertIsNone(selected["population_mixture_risk_bounds"]["upper"])
        self.assertFalse(selected["qualifies_for_primary_comparison"])
        self.assertEqual(selected["overall"]["abstentions"], 300)

    def test_final_decision_or_error_tampering_is_rejected(self):
        for field, value in (("false_approval", True), ("decision", APPROVE), ("interval_covered", False)):
            rows = self.outcomes(2)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                analyze_primary_comparison(rows, block_count=2)

    def test_discordant_coverage_is_paired_and_all_comparators_reported(self):
        rows = self.outcomes(10)
        for i, row in enumerate(rows):
            if row["procedure"] == SELECTOR_PROCEDURE and row["block"] < 2:
                rows[i] = calibrated_outcome(dict(row, raw_interval={"lower": -1., "upper": 1.}), 0.)
        result = analyze_primary_comparison(rows, block_count=10)
        self.assertEqual(len(result["comparisons"]), 4)
        pairs = result["comparisons"]["fixed_voltage"]["coverage_pairs_by_family"]
        for pair in pairs.values():
            self.assertEqual(pair["fixed_only_decisions"], 2)
            self.assertEqual(pair["selector_only_decisions"], 0)
            self.assertEqual(pair["coverage_loss_estimate"], .2)

    def test_success_requires_every_qualifying_fixed_comparison(self):
        rows = self.outcomes(800)  # Enough approvals for the strict zero-error bound.
        for row in rows:
            if row["procedure"] != SELECTOR_PROCEDURE:
                row["realized_resources"]["total_diagnostic_energy"] = 70.
        result = analyze_primary_comparison(rows, block_count=800)
        self.assertTrue(result["primary_success"])
        self.assertEqual(len(result["qualifying_fixed_policies"]), 4)
        for row in rows:
            if row["procedure"] == "stop_now":
                row["realized_resources"]["total_diagnostic_energy"] = 60.
        result = analyze_primary_comparison(rows, block_count=800)
        self.assertFalse(result["primary_success"])
        self.assertEqual(result["qualifying_fixed_frontier"], ["stop_now"])
        self.assertNotIn("stop_now", result["named_comparisons_passing_all_checks"])


class PhaseEProtocolTests(unittest.TestCase):
    def test_protocol_binds_complete_design_and_unopened_partitions(self):
        p = phase_e_protocol_payload(source_revision="a"*40, source_manifest_digest="b"*64)
        self.assertEqual(validate_phase_e_protocol(p), p)
        self.assertEqual(p["calibration"]["order_statistic_rank"], 96)
        self.assertEqual(p["calibration_block_count"], 100)
        self.assertEqual(p["reserved_block_count"], 100)
        self.assertFalse(p["generation_gate"]["reserved_generation_authorized"])
        self.assertEqual(p["analysis"]["confidence"]["primary_bound_count"], 73)
        self.assertTrue(p["precision_and_compute"]["risk_comparison_not_promised_to_be_powered"])

    def test_outcome_directed_protocol_change_rejected(self):
        p = phase_e_protocol_payload(source_revision="a"*40, source_manifest_digest="b"*64)
        changed = deepcopy(p)
        changed["analysis"]["max_false_approval_risk_increase_against_fixed"] = .1
        with self.assertRaises(ValueError):
            validate_phase_e_protocol(changed)


if __name__ == "__main__":
    unittest.main()
