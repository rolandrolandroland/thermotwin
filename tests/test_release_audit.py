from types import SimpleNamespace
from pathlib import Path
import unittest
from zipfile import ZipFile

from thermotwin.reports.release_audit import (
    evaluate_release_evidence,
    format_release_evidence_audit,
)


def _fixtures(*, hidden_face_rmse=0.007105):
    engineering = SimpleNamespace(
        sparse_inference=SimpleNamespace(
            fit=SimpleNamespace(inferred_cold_contact_resistance=0.25103),
            withheld_validation=SimpleNamespace(accessible_sensor_rmse=0.00187),
        ),
        experiment_selection=SimpleNamespace(
            selected=SimpleNamespace(
                current_amplitude=0.8,
                pulse_duration=20.0,
                information_gain_nats=7.198,
            ),
            validation=SimpleNamespace(rmse_reduction_percent=82.2),
        ),
    )
    reconstruction = SimpleNamespace(
        summaries=(
            SimpleNamespace(
                mean_retained_noisy_observation_rmse=0.019433,
                mean_missing_exchanger_rmse=0.009696,
                mean_hidden_face_rmse=hidden_face_rmse,
                mean_energy_rate_closure_rms=0.132833,
                mean_absolute_final_cumulative_energy_error=1.952169,
            ),
            SimpleNamespace(
                mean_retained_noisy_observation_rmse=0.017471,
                mean_missing_exchanger_rmse=0.079878,
                mean_hidden_face_rmse=2.193724,
                mean_energy_rate_closure_rms=16.493587,
                mean_absolute_final_cumulative_energy_error=370.392719,
            ),
        ),
        physics_missing_exchanger_rmse_reduction_percent=87.86,
        physics_hidden_face_rmse_reduction_percent=99.68,
        physics_energy_rate_error_reduction_percent=99.19,
        physics_all_metric_advantage_count=5,
        physics_completion_gate_pass_count=5,
    )
    nonlinear = SimpleNamespace(
        selected_rmse_reduction_vs_naive_percent=81.46,
        selected_rmse_reduction_vs_resource_control_percent=11.77,
    )
    codesign = SimpleNamespace(
        robustness_results=(SimpleNamespace(feasible_fraction=0.553),)
    )
    return engineering, reconstruction, nonlinear, codesign


class ReleaseAuditTests(unittest.TestCase):
    def test_expected_release_evidence_passes(self):
        audit = evaluate_release_evidence(*_fixtures())

        self.assertTrue(audit.passed)
        self.assertEqual(len(audit.checks), 24)
        self.assertIn("overall: PASS", format_release_evidence_audit(audit))

    def test_changed_headline_value_fails_loudly(self):
        audit = evaluate_release_evidence(*_fixtures(hidden_face_rmse=0.02))

        self.assertFalse(audit.passed)
        failures = tuple(item.name for item in audit.checks if not item.passed)
        self.assertEqual(failures, ("physics hidden-face RMSE",))

    def test_public_release_documents_and_five_slide_deck_exist(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "thermotwin/TECHNICAL_SUMMARY.md",
            "thermotwin/DEMO_SCRIPT.md",
            "thermotwin/PORTFOLIO_BULLETS.md",
        ):
            self.assertGreater((root / relative).stat().st_size, 500)

        deck = root / "docs/thermotwin/ThermoTwin_Technical_Overview.pptx"
        self.assertEqual(deck.read_bytes()[:2], b"PK")
        with ZipFile(deck) as archive:
            slides = tuple(
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
            notes = tuple(
                name
                for name in archive.namelist()
                if name.startswith("ppt/notesSlides/notesSlide")
                and name.endswith(".xml")
            )
        self.assertEqual(len(slides), 5)
        self.assertEqual(len(notes), 5)


if __name__ == "__main__":
    unittest.main()
