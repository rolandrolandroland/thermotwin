import subprocess
import sys
import unittest


class PackageArchitectureTests(unittest.TestCase):
    def test_core_import_does_not_load_optional_dependencies(self):
        command = (
            "import sys, thermotwin; "
            "assert 'torch' not in sys.modules; "
            "assert 'matplotlib' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_low_level_compatibility_paths_preserve_object_identity(self):
        from thermotwin.controls import PiecewiseConstantCurrent as legacy_control
        from thermotwin.core.controls import (
            PiecewiseConstantCurrent as core_control,
        )
        from thermotwin.physics.thermoelectric import (
            ThermoelectricParameters as layered_parameters,
        )
        from thermotwin.simulation.controls import (
            PiecewiseConstantCurrent as simulation_compatibility_control,
        )
        from thermotwin.thermoelectric import (
            ThermoelectricParameters as legacy_parameters,
        )

        self.assertIs(legacy_control, core_control)
        self.assertIs(simulation_compatibility_control, core_control)
        self.assertIs(legacy_parameters, layered_parameters)

    def test_observation_and_inference_facades_preserve_identity(self):
        from thermotwin.measurement_noise import GaussianTemperatureNoise as legacy
        from thermotwin.observations.noise import GaussianTemperatureNoise as layered
        from thermotwin.sparse_sensor_inference import (
            fit_sparse_sensor_parameters as legacy_fit,
        )
        from thermotwin.inference.sparse_sensors import (
            fit_sparse_sensor_parameters as layered_fit,
        )

        self.assertIs(legacy, layered)
        self.assertIs(legacy_fit, layered_fit)

    def test_codesign_facade_preserves_split_implementation_identity(self):
        from thermotwin.design.codesign.evaluation import (
            evaluate_design_current as layered_evaluate,
        )
        from thermotwin.design.codesign.optimization import (
            expected_improvement as layered_improvement,
        )
        from thermotwin.material_geometry_codesign import (
            evaluate_design_current as legacy_evaluate,
            expected_improvement as legacy_improvement,
        )

        self.assertIs(legacy_evaluate, layered_evaluate)
        self.assertIs(legacy_improvement, layered_improvement)

    def test_report_facade_preserves_identity(self):
        from thermotwin.cop_operating_map_report import (
            save_cop_operating_map_report as legacy_save,
        )
        from thermotwin.reports.cop_map import (
            save_cop_operating_map_report as layered_save,
        )

        self.assertIs(legacy_save, layered_save)


if __name__ == "__main__":
    unittest.main()
