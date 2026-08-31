from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from thermotwin.reports.distributed_properties import (
    DistributedInversePropertyValidation,
    format_distributed_property_study,
    run_distributed_property_study,
    save_distributed_property_figure,
)


class DistributedPropertyReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_distributed_property_study(
            run_forward_pinn=False,
            quick=True,
        )

    def test_report_states_the_local_identifiability_boundary(self):
        text = format_distributed_property_study(self.result)
        self.assertIn("joint: rank", text)
        self.assertIn("training gate", text)
        self.assertIn("not trained", text)

    def test_figure_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "distributed.png"
            returned = save_distributed_property_figure(self.result, output)
            self.assertEqual(returned, output.resolve())
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)

    def test_report_compares_each_inverse_property_family_separately(self):
        validation = DistributedInversePropertyValidation(
            property_name="thermal_conductivity",
            truth_multipliers=(0.96, 1.03, 1.08),
            conventional_multipliers=(0.961, 1.029, 1.079),
            inverse_pinn_multipliers=(0.97, 1.02, 1.06),
            inverse_pinn_initial_loss=10.0,
            inverse_pinn_final_loss=0.5,
            observation_channels=(
                "cold_face_temperature",
                "hot_face_temperature",
                "voltage",
                "cold_side_heat",
                "hot_side_heat",
            ),
        )
        result = replace(
            self.result,
            inverse_property_validations=(validation,),
        )
        text = format_distributed_property_study(result)
        self.assertIn("Independent one-function inverse validations", text)
        self.assertIn("kappa(T)", text)
        self.assertIn("cold_side_heat", text)
        self.assertIn("conventional=0.001000", text)
        self.assertIn("PINN=0.020000", text)
        self.assertIn("only the named curve", text)

    def test_empty_inverse_property_selection_is_rejected_before_running(self):
        with self.assertRaisesRegex(ValueError, "at least one inverse property"):
            run_distributed_property_study(
                run_inverse_pinn=True,
                inverse_properties=(),
                quick=True,
            )

    def test_inverse_validation_requires_three_matching_knots(self):
        with self.assertRaisesRegex(ValueError, "three matching knots"):
            DistributedInversePropertyValidation(
                property_name="seebeck_coefficient",
                truth_multipliers=(1.0, 1.0),
                conventional_multipliers=(1.0, 1.0),
                inverse_pinn_multipliers=(1.0, 1.0),
                inverse_pinn_initial_loss=1.0,
                inverse_pinn_final_loss=0.5,
            )


if __name__ == "__main__":
    unittest.main()
