from pathlib import Path
import tempfile
import unittest

from thermotwin.pinn.distributed_inverse import InverseDistributedHistory
from thermotwin.reports.distributed_pinn_training_audit import (
    format_distributed_pinn_training_audit_report,
    save_distributed_pinn_training_audit_figure,
)
from thermotwin.studies.distributed_inverse_robustness import (
    DistributedInverseRobustnessSeeds,
)
from thermotwin.studies.distributed_pinn_training_audit import (
    DistributedPINNTrainingAuditConfig,
    DistributedPINNTrainingAuditResult,
    DistributedPINNTrainingAuditTrial,
    distributed_training_checkpoint,
    summarize_distributed_pinn_training_audit,
)


def _result():
    config = DistributedPINNTrainingAuditConfig(
        trial_count=1, checkpoint_epochs=(1, 2)
    )
    history = InverseDistributedHistory(
        total_loss=(2.0, 1.0),
        physics_loss=(0.75, 0.12),
        observation_loss=(1.25, 0.88),
        smoothness_loss=(0.1, 0.05),
        shrinkage_loss=(0.0, 0.0),
        property_values=((1.05, 1.06, 1.05), (1.04, 1.07, 1.03)),
    )
    checkpoints = tuple(
        distributed_training_checkpoint(
            history,
            epoch=epoch,
            baseline_values=(1.0, 1.0, 1.0),
            reference_rate_rms=1.0,
            residual_rate_scale=1.0,
            config=config,
        )
        for epoch in (1, 2)
    )
    trial = DistributedPINNTrainingAuditTrial(
        trial_index=0,
        seeds=DistributedInverseRobustnessSeeds(10, (11, 12, 13, 14)),
        checkpoints=checkpoints,
        first_operational_epoch=2,
    )
    return DistributedPINNTrainingAuditResult(
        config=config,
        truth_multipliers=(1.04, 1.07, 1.03),
        reference_temperature_rate_rms=1.0,
        trials=(trial,),
        epoch_summaries=summarize_distributed_pinn_training_audit((trial,)),
    )


class DistributedPINNTrainingAuditReportTests(unittest.TestCase):
    def test_report_separates_operational_and_truth_known_checks(self):
        text = format_distributed_pinn_training_audit_report(_result())
        self.assertIn("Truth-blind operational gate", text)
        self.assertIn("Truth-known benchmark diagnostics", text)
        self.assertIn("first operational epoch=2", text)
        self.assertIn("physics-loss weight: 10.000", text)
        self.assertIn("physics RMS", text)
        self.assertIn("nearly flat curve", text)

    def test_figure_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training-audit.png"
            returned = save_distributed_pinn_training_audit_figure(
                _result(), output
            )
            self.assertEqual(returned, output.resolve())
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
