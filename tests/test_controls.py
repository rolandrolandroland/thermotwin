import unittest

from thermotwin import PiecewiseConstantCurrent


class PiecewiseConstantCurrentTests(unittest.TestCase):
    def test_step_is_right_continuous_at_transition(self):
        schedule = PiecewiseConstantCurrent.step(
            transition_time=2.0,
            before_current=0.0,
            after_current=1.5,
        )

        self.assertEqual(schedule.value_at(1.999), 0.0)
        self.assertEqual(schedule.value_at(2.0), 1.5)
        self.assertEqual(schedule.value_at(20.0), 1.5)

    def test_pulse_returns_to_baseline_at_end(self):
        schedule = PiecewiseConstantCurrent.pulse(
            start_time=1.0,
            end_time=3.0,
            pulse_current=2.0,
            baseline_current=-0.5,
        )

        self.assertEqual(schedule.value_at(0.0), -0.5)
        self.assertEqual(schedule.value_at(1.0), 2.0)
        self.assertEqual(schedule.value_at(2.0), 2.0)
        self.assertEqual(schedule.value_at(3.0), -0.5)
        self.assertEqual(schedule.next_transition_after(0.0), 1.0)
        self.assertEqual(schedule.next_transition_after(1.0), 3.0)
        self.assertIsNone(schedule.next_transition_after(3.0))

    def test_periodic_pulse_is_right_continuous(self):
        schedule = PiecewiseConstantCurrent.periodic_pulse(
            duration=25.0,
            period=10.0,
            duty_cycle=0.4,
            pulse_current=1.2,
        )

        self.assertEqual(
            schedule.transition_times,
            (4.0, 10.0, 14.0, 20.0, 24.0),
        )
        self.assertEqual(schedule.value_at(3.999), 1.2)
        self.assertEqual(schedule.value_at(4.0), 0.0)
        self.assertEqual(schedule.value_at(10.0), 1.2)
        self.assertEqual(schedule.value_at(25.0), 0.0)

    def test_periodic_pulse_rejects_invalid_shape(self):
        for keyword, value in (
            ("duration", -1.0),
            ("period", 0.0),
            ("duty_cycle", 0.0),
            ("duty_cycle", 1.0),
        ):
            arguments = dict(
                duration=20.0,
                period=10.0,
                duty_cycle=0.5,
                pulse_current=1.0,
            )
            arguments[keyword] = value
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaises(ValueError):
                    PiecewiseConstantCurrent.periodic_pulse(**arguments)

    def test_invalid_schedules_are_rejected(self):
        invalid_cases = (
            dict(transition_times=(1.0,), values=(0.0,)),
            dict(transition_times=(-1.0,), values=(0.0, 1.0)),
            dict(transition_times=(2.0, 1.0), values=(0.0, 1.0, 0.0)),
            dict(transition_times=(1.0, 1.0), values=(0.0, 1.0, 0.0)),
            dict(transition_times=(float("nan"),), values=(0.0, 1.0)),
            dict(transition_times=(), values=(float("inf"),)),
        )

        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    PiecewiseConstantCurrent(**values)


if __name__ == "__main__":
    unittest.main()
