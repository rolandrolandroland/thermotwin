import math
import unittest

from thermotwin.control_comparison import ControlComparisonConfig
from thermotwin.pulse_operating_map import run_pulse_operating_map


class PulseOperatingMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ControlComparisonConfig(
            warmup_duration=160.0,
            evaluation_duration=40.0,
            time_step=0.4,
            target_cooling_rates=(2.0,),
            pulse_periods=(10.0,),
            pulse_duty_cycles=(0.75,),
            maximum_storage_drift=0.2,
        )
        cls.result = run_pulse_operating_map(cls.config)

    def test_continuous_transient_matches_steady_map(self):
        item = self.result.comparisons[0]
        self.assertLess(abs(item.steady_map_cop_error_percent), 2.0)

    def test_pulse_current_statistics_are_distinct(self):
        item = self.result.comparisons[0]
        self.assertAlmostEqual(
            item.pulse_mean_current,
            item.pulse_duty_cycle * item.pulse_peak_current,
        )
        self.assertAlmostEqual(
            item.pulse_rms_current,
            math.sqrt(item.pulse_duty_cycle) * item.pulse_peak_current,
        )
        self.assertGreater(item.pulse_rms_current, item.pulse_mean_current)

    def test_pulse_result_is_below_continuous_envelope(self):
        item = self.result.comparisons[0]
        self.assertLess(item.pulse_cop, item.continuous_cop)
        self.assertLess(item.pulse_cop_change_percent, 0.0)
        self.assertLess(item.pulse_cooling_change_at_equal_power_percent, 0.0)

    def test_steady_curve_spans_current_grid(self):
        self.assertEqual(len(self.result.steady_curve), 30)


if __name__ == "__main__":
    unittest.main()
