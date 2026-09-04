import unittest

from thermotwin.inference.joint_thermal_parameters import JointThermalFitConfig
from thermotwin.reports.adaptive_experiment_campaign import (
    format_adaptive_campaign_report,
)
from thermotwin.simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
)
from thermotwin.simulation.interface_mass_mismatch import (
    InterfaceMassMismatch,
    integrate_interface_mass_truth,
)
from thermotwin.studies.adaptive_experiment_campaign import (
    AdaptiveCampaignConfig,
    run_adaptive_experiment_campaign,
)


class InterfaceMassTruthTests(unittest.TestCase):
    def test_zero_current_shared_equilibrium_remains_constant(self):
        reference = constant_current_contact_reference_experiment()
        trajectory = integrate_interface_mass_truth(
            reference.thermoelectric_parameters,
            reference.thermal_parameters,
            InterfaceMassMismatch(),
            initial_temperature=300.0,
            duration=5.0,
            time_step=0.5,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        for history in trajectory[1:]:
            self.assertTrue(all(value == 300.0 for value in history))

    def test_mismatch_configuration_rejects_nonphysical_values(self):
        with self.assertRaises(ValueError):
            InterfaceMassMismatch(thermal_capacitance=0.0)
        with self.assertRaises(ValueError):
            InterfaceMassMismatch(face_side_resistance_fraction=1.0)


class AdaptiveExperimentCampaignTests(unittest.TestCase):
    @staticmethod
    def small_config():
        return AdaptiveCampaignConfig(
            trial_count=1,
            experiment_count=1,
            total_energy_budget=30.0,
            heuristic_candidate_names=("0.6A_20s",),
            fit=JointThermalFitConfig(
                dense_time_step=0.5,
                gauss_newton_iterations=2,
                initial_log_multipliers=((0.0, 0.0, 0.0),),
            ),
        )

    def test_campaign_is_paired_and_mismatch_exposes_hidden_state(self):
        result = run_adaptive_experiment_campaign(self.small_config())

        self.assertEqual(len(result.steps), 6)
        matched = tuple(
            item for item in result.steps if item.truth_condition == "matched_model"
        )
        mismatch = tuple(
            item
            for item in result.steps
            if item.truth_condition == "extra_interface_mass"
        )
        self.assertEqual(len(matched), 3)
        self.assertEqual(len(mismatch), 3)
        self.assertLess(
            max(item.heldout_hidden_face_rmse for item in matched),
            min(item.heldout_hidden_face_rmse for item in mismatch),
        )
        self.assertTrue(any(item.false_confidence for item in mismatch))

    def test_report_states_virtual_and_model_mismatch_boundaries(self):
        text = format_adaptive_campaign_report(
            run_adaptive_experiment_campaign(self.small_config())
        )

        self.assertIn("extra state", text)
        self.assertIn("does not count prototypes", text)
        self.assertIn("not hardware validation", text)

    def test_heuristic_budget_is_checked(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            run_adaptive_experiment_campaign(
                AdaptiveCampaignConfig(
                    trial_count=1,
                    experiment_count=1,
                    total_energy_budget=5.0,
                    heuristic_candidate_names=("0.8A_20s",),
                    fit=JointThermalFitConfig(
                        dense_time_step=0.5,
                        gauss_newton_iterations=1,
                        initial_log_multipliers=((0.0, 0.0, 0.0),),
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
