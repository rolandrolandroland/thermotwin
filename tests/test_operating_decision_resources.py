import math
import unittest
from dataclasses import replace

from thermotwin.studies.operating_decision import (
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    default_fixed_policies,
)
from thermotwin.studies.operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RunInstrumentation,
    build_realistic_blinded_case,
    nominal_realistic_schedule_energy,
    realism_truth_for_trial,
)
from thermotwin.studies.operating_decision_resources import (
    RealizedRunEnergy,
    nominal_selection_cost_proxy,
    realized_blinded_case_energy,
    realized_blinded_case_energy_from_truth,
    realized_regime_terminal_energy,
    realized_regime_terminal_energy_from_truth,
)


class OperatingDecisionResourceTests(unittest.TestCase):
    def setUp(self):
        self.config = OperatingDecisionRealismConfig()
        self.truth_condition = STAGE3_TRUTH_CONDITIONS[0]
        self.policies = {
            policy.name: policy for policy in default_fixed_policies()
        }

    def case(self, policy_name, trial_index=0):
        return build_realistic_blinded_case(
            self.truth_condition,
            trial_index,
            self.policies[policy_name],
            self.config,
        )

    def test_case_totals_are_exact_sums_of_immutable_run_records(self):
        case = self.case(FIXED_THERMAL)
        result = realized_blinded_case_energy(
            self.truth_condition,
            0,
            case,
            self.config,
        )

        self.assertEqual(
            result.acquisition_energy,
            sum(run.terminal_energy for run in result.acquisition_runs),
        )
        self.assertEqual(
            result.verification_energy,
            result.verification_run.terminal_energy,
        )
        self.assertEqual(
            result.total_diagnostic_energy,
            result.acquisition_energy + result.verification_energy,
        )
        self.assertEqual(
            tuple(run.regime for run in result.acquisition_runs),
            tuple(run.regime for run in case.acquisition_runs),
        )
        with self.assertRaisesRegex(Exception, "cannot assign"):
            result.total_diagnostic_energy = 0.0

    def test_realized_energy_is_deterministic_and_varies_by_device(self):
        case_zero = self.case(FIXED_VOLTAGE, 0)
        regime = case_zero.acquisition_runs[0].regime
        instrumentation = case_zero.acquisition_runs[0].instrumentation
        first = realized_regime_terminal_energy(
            self.truth_condition,
            0,
            regime,
            instrumentation,
            self.config,
        )
        repeated = realized_regime_terminal_energy(
            self.truth_condition,
            0,
            regime,
            instrumentation,
            self.config,
        )
        other_device = realized_regime_terminal_energy(
            self.truth_condition,
            1,
            regime,
            instrumentation,
            self.config,
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first.terminal_energy, other_device.terminal_energy)
        self.assertTrue(math.isfinite(first.terminal_energy))
        self.assertGreaterEqual(first.terminal_energy, 0.0)

    def test_face_probe_loading_uses_the_instrumented_truth_path(self):
        face_case = self.case(FIXED_FACE_TEMPERATURE)
        voltage_case = self.case(FIXED_VOLTAGE)
        face_run = face_case.acquisition_runs[1]
        voltage_run = voltage_case.acquisition_runs[1]
        self.assertEqual(face_run.regime.current, voltage_run.regime.current)

        loaded = realized_regime_terminal_energy(
            self.truth_condition,
            0,
            face_run.regime,
            face_run.instrumentation,
            self.config,
        )
        unloaded = realized_regime_terminal_energy(
            self.truth_condition,
            0,
            voltage_run.regime,
            voltage_run.instrumentation,
            self.config,
        )

        self.assertTrue(loaded.instrumentation.temporary_face_sensor)
        self.assertFalse(unloaded.instrumentation.temporary_face_sensor)
        self.assertNotEqual(loaded.terminal_energy, unloaded.terminal_energy)

    def test_nominal_selection_proxy_matches_the_legacy_nominal_metric(self):
        case = self.case(FIXED_FACE_TEMPERATURE)
        run = case.acquisition_runs[1]
        proxy = nominal_selection_cost_proxy(
            run.regime,
            run.instrumentation,
            self.config,
        )
        legacy = nominal_realistic_schedule_energy(
            run.regime.current,
            self.config,
            temporary_face_sensor=True,
        )
        realized = realized_regime_terminal_energy(
            self.truth_condition,
            0,
            run.regime,
            run.instrumentation,
            self.config,
        )

        self.assertEqual(proxy.terminal_energy, legacy)
        self.assertNotEqual(proxy.terminal_energy, realized.terminal_energy)

    def test_explicit_truth_path_matches_legacy_draw_and_accepts_bound_identity(self):
        case = self.case(FIXED_VOLTAGE)
        truth = realism_truth_for_trial(self.config, 0)
        explicit = realized_blinded_case_energy_from_truth(
            self.truth_condition,
            0,
            case,
            truth,
            self.config,
            expected_case_id=case.case_id,
        )
        legacy = realized_blinded_case_energy(
            self.truth_condition,
            0,
            case,
            self.config,
        )
        self.assertEqual(explicit, legacy)
        run = case.acquisition_runs[0]
        self.assertEqual(
            realized_regime_terminal_energy_from_truth(
                self.truth_condition,
                0,
                run.regime,
                run.instrumentation,
                truth,
                self.config,
            ),
            realized_regime_terminal_energy(
                self.truth_condition,
                0,
                run.regime,
                run.instrumentation,
                self.config,
            ),
        )

    def test_identity_regime_and_energy_validation_rejects_bad_records(self):
        case = self.case(FIXED_VOLTAGE)
        wrong_id = case.case_id._replace(device_token="wrong-device")
        mismatched = replace(case, case_id=wrong_id)
        with self.assertRaisesRegex(ValueError, "identity"):
            realized_blinded_case_energy(
                self.truth_condition,
                0,
                mismatched,
                self.config,
            )

        run = case.acquisition_runs[0]
        with self.assertRaisesRegex(ValueError, "face-probe state"):
            realized_regime_terminal_energy(
                self.truth_condition,
                0,
                run.regime,
                RunInstrumentation(True),
                self.config,
            )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            RealizedRunEnergy(
                truth_condition=self.truth_condition,
                trial_index=0,
                regime=run.regime,
                instrumentation=run.instrumentation,
                terminal_energy=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
