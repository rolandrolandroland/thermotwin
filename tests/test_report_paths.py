import json
from pathlib import Path
import tempfile
import unittest

from thermotwin.reports.paths import (
    FIGURES_DIRECTORY,
    default_figure_path,
    experiment_artifact_directory,
    figure_data_path,
    figure_explanation_path,
    save_figure_data,
)


class ReportPathTests(unittest.TestCase):
    def test_walkthrough_directory_matches_markdown_stem(self):
        expected = FIGURES_DIRECTORY / "COP_OPERATING_MAP_EXPERIMENT"
        self.assertEqual(
            experiment_artifact_directory("COP_OPERATING_MAP_EXPERIMENT.md"),
            expected,
        )
        self.assertEqual(
            experiment_artifact_directory("COP_OPERATING_MAP_EXPERIMENT"),
            expected,
        )
        self.assertEqual(
            default_figure_path(
                "cop_operating_map.png", "COP_OPERATING_MAP_EXPERIMENT.md"
            ),
            expected / "cop_operating_map.png",
        )

    def test_paths_reject_nested_or_unrelated_names(self):
        for value in ("", "nested/REPORT.md", "REPORT.txt", "../REPORT.md"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    experiment_artifact_directory(value)
        for value in ("", "nested/figure.png"):
            with self.subTest(figure=value):
                with self.assertRaises(ValueError):
                    default_figure_path(value, "REPORT.md")

    def test_json_sidecar_is_colocated_and_machine_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            figure = Path(directory) / "figure.png"
            output = save_figure_data(
                {
                    "time": (0.0, 1.0),
                    "temperature": (300.0, float("nan")),
                    "source": Path("input.csv"),
                },
                figure,
            )
            self.assertEqual(output, figure_data_path(figure))
            self.assertEqual(output.name, "figure.json")
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["figure"], "figure.png")
            self.assertEqual(payload["data"]["time"], [0.0, 1.0])
            self.assertEqual(payload["data"]["temperature"][1], "NaN")
            self.assertEqual(payload["data"]["source"], "input.csv")
            explanation = figure_explanation_path(figure)
            self.assertTrue(explanation.is_file())
            text = explanation.read_text(encoding="utf-8")
            self.assertIn("What this figure shows", text)
            self.assertIn("figure.json", text)


if __name__ == "__main__":
    unittest.main()
