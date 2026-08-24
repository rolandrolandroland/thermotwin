from io import StringIO
import unittest

from thermotwin.hardware_data import (
    read_hardware_csv,
    summarize_hardware_dataset,
)


class HardwareDataTests(unittest.TestCase):
    def test_reader_preserves_missing_temperature_and_optional_voltage(self):
        dataset = read_hardware_csv(
            StringIO(
                "time_s,current_A,cold_exchanger_K,hot_exchanger_K,voltage_V\n"
                "0,0,300.0,300.1,0.0\n"
                "1,0.8,,300.2,1.7\n"
                "2,0.8,299.8,300.4,1.8\n"
            )
        )
        summary = summarize_hardware_dataset(dataset)

        self.assertEqual(summary.row_count, 3)
        self.assertEqual(len(dataset.inputs), 3)
        self.assertEqual(summary.cold_temperature_count, 2)
        self.assertEqual(summary.hot_temperature_count, 3)
        self.assertEqual(summary.voltage_count, 3)
        self.assertEqual(summary.median_sampling_interval, 1.0)

    def test_reader_rejects_missing_units_column(self):
        with self.assertRaisesRegex(ValueError, "hot_exchanger_K"):
            read_hardware_csv(
                StringIO(
                    "time_s,current_A,cold_exchanger_K\n"
                    "0,0,300\n"
                    "1,1,299\n"
                )
            )

    def test_input_history_preserves_row_with_both_temperatures_missing(self):
        dataset = read_hardware_csv(
            StringIO(
                "time_s,current_A,cold_exchanger_K,hot_exchanger_K\n"
                "0,0,300,300\n"
                "1,0.8,,\n"
                "2,0,299.9,300.1\n"
            )
        )

        self.assertEqual(
            tuple(item.time for item in dataset.inputs),
            (0.0, 1.0, 2.0),
        )
        self.assertEqual(len(dataset.temperatures.observations), 4)
        self.assertEqual(summarize_hardware_dataset(dataset).row_count, 3)

    def test_reader_rejects_nonmonotonic_time(self):
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            read_hardware_csv(
                StringIO(
                    "time_s,current_A,cold_exchanger_K,hot_exchanger_K\n"
                    "1,0,300,300\n"
                    "1,1,299,301\n"
                )
            )


if __name__ == "__main__":
    unittest.main()
