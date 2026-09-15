import unittest
from unittest.mock import patch

from thermotwin.studies.operating_decision_random_streams import (
    CALIBRATION_STREAM,
    DEVICE_TRUTH_STREAM,
    OBSERVATION_STREAM,
    RANDOM_STREAM_PROTOCOL_VERSION,
    RandomStreamKey,
    RandomStreamRegistry,
    StreamUse,
    audit_stream_uses,
    calibration_stream,
    device_truth_stream,
    observation_stream,
    seed_from_stream_key,
)


class OperatingDecisionRandomStreamTests(unittest.TestCase):
    @staticmethod
    def observation(**overrides):
        arguments = {
            "campaign": "corrected-stage-6",
            "partition": "development",
            "block": 7,
            "purpose": "sensor_error",
            "family": "temperature-dependent-contact",
            "run": "initial_0.8A_20s",
            "channel": "cold_exchanger",
        }
        arguments.update(overrides)
        return observation_stream(**arguments)

    def test_seed_and_generator_are_deterministic(self):
        first = self.observation()
        second = self.observation()

        self.assertEqual(first, second)
        self.assertEqual(first.seed, second.seed)
        self.assertEqual(first.seed, seed_from_stream_key(first.key))
        expected_seed = int(
            "480319518083938133156235429978017923100"
            "18552387849195580836761325934522970437"
        )
        self.assertEqual(
            first.seed,
            expected_seed,
        )
        first_generator = first.new_generator()
        second_generator = second.new_generator()
        self.assertEqual(
            [first_generator.random() for _ in range(2)],
            [second_generator.random() for _ in range(2)],
        )
        self.assertGreater(first.seed.bit_length(), 240)

    def test_campaign_partition_protocol_purpose_and_family_are_namespaces(self):
        streams = (
            self.observation(),
            self.observation(campaign="corrected-stage-7"),
            self.observation(partition="calibration"),
            self.observation(protocol_version="future-rng-v3"),
            self.observation(purpose="run_bias"),
            self.observation(family="five-state-contact"),
        )

        self.assertEqual(len({stream.key for stream in streams}), len(streams))
        self.assertEqual(len({stream.seed for stream in streams}), len(streams))

    def test_channel_run_and_block_coordinates_never_alias(self):
        streams = tuple(
            self.observation(block=block, run=run, channel=channel)
            for block in range(3)
            for run in ("initial", "thermal", "verification")
            for channel in ("cold_exchanger", "hot_exchanger", "voltage")
        )

        self.assertEqual(len({stream.key for stream in streams}), len(streams))
        self.assertEqual(len({stream.seed for stream in streams}), len(streams))

    def test_device_truth_api_supports_shared_and_family_specific_truth(self):
        shared = device_truth_stream(
            campaign="corrected-stage-6",
            partition="development",
            block=7,
            purpose="physical_parameters",
        )
        family_specific = device_truth_stream(
            campaign="corrected-stage-6",
            partition="development",
            block=7,
            purpose="contact_beta",
            family="temperature-dependent-contact",
        )

        self.assertEqual(shared.key.stream_kind, DEVICE_TRUTH_STREAM)
        self.assertIsNone(shared.key.family)
        self.assertIsNone(shared.key.run)
        self.assertIsNone(shared.key.channel)
        self.assertNotEqual(shared.seed, family_specific.seed)

    def test_calibration_and_bootstrap_streams_have_their_own_namespace(self):
        gate = calibration_stream(
            campaign="corrected-stage-6",
            partition="gate-development",
            purpose="noise-reference/fixed-voltage",
        )
        bootstrap = calibration_stream(
            campaign="corrected-stage-6",
            partition="bootstrap",
            purpose="balanced-loss",
        )
        self.assertEqual(gate.key.stream_kind, CALIBRATION_STREAM)
        self.assertNotEqual(gate.seed, bootstrap.seed)
        self.assertNotEqual(gate.seed, self.observation().seed)

    def test_policy_pairing_is_intentional_and_auditable(self):
        common_measurement_key = self.observation().key
        common_truth = device_truth_stream(
            campaign="corrected-stage-6",
            partition="development",
            block=7,
            purpose="physical_parameters",
        )
        registry = RandomStreamRegistry()
        for policy in ("stop", "fixed-thermal", "adaptive"):
            common_measurement = self.observation()
            returned = registry.register(
                common_measurement,
                consumer=f"{policy}/initial/cold-exchanger",
                pairing_member=policy,
                pairing_id="block-7-common-initial",
            )
            self.assertIs(returned, common_measurement)
            self.assertEqual(returned.key, common_measurement_key)
            registry.register(
                common_truth,
                consumer=f"{policy}/device-truth",
                pairing_member=policy,
                pairing_id="block-7-common-device",
            )

        audit = registry.audit()
        self.assertTrue(audit.ok)
        self.assertEqual(audit.use_count, 6)
        self.assertEqual(audit.unique_key_count, 2)
        self.assertEqual(audit.unique_seed_count, 2)
        self.assertEqual(len(audit.declared_pairings), 2)
        self.assertEqual(
            {pairing.pairing_id for pairing in audit.declared_pairings},
            {"block-7-common-initial", "block-7-common-device"},
        )
        self.assertTrue(
            all(
                pairing.members == ("stop", "fixed-thermal", "adaptive")
                for pairing in audit.declared_pairings
            )
        )
        audit.assert_clean()

    def test_different_keys_with_a_derived_seed_collision_fail_the_audit(self):
        uses = (
            StreamUse(self.observation(channel="cold_exchanger"), "cold"),
            StreamUse(self.observation(channel="hot_exchanger"), "hot"),
        )

        with patch(
            "thermotwin.studies.operating_decision_random_streams."
            "seed_from_stream_key",
            return_value=1,
        ):
            audit = audit_stream_uses(uses)

        self.assertFalse(audit.ok)
        self.assertEqual(audit.unique_key_count, 2)
        self.assertEqual(audit.unique_seed_count, 1)
        self.assertIn("different stream keys", audit.unintended_reuses[0].reason)

    def test_unannounced_or_inconsistent_reuse_fails_the_audit(self):
        stream = self.observation()
        unannounced = audit_stream_uses(
            (
                StreamUse(stream, "stop/initial", pairing_member="stop"),
                StreamUse(stream, "adaptive/initial", pairing_member="adaptive"),
            )
        )
        inconsistent = audit_stream_uses(
            (
                StreamUse(
                    stream,
                    "stop/initial",
                    pairing_member="stop",
                    pairing_id="pair-a",
                ),
                StreamUse(
                    stream,
                    "adaptive/initial",
                    pairing_member="adaptive",
                    pairing_id="pair-b",
                ),
            )
        )
        duplicate_policy = audit_stream_uses(
            (
                StreamUse(
                    stream,
                    "adaptive/fit",
                    pairing_member="adaptive",
                    pairing_id="pair-a",
                ),
                StreamUse(
                    stream,
                    "adaptive/replay",
                    pairing_member="adaptive",
                    pairing_id="pair-a",
                ),
            )
        )

        for audit in (unannounced, inconsistent, duplicate_policy):
            self.assertFalse(audit.ok)
            self.assertEqual(len(audit.unintended_reuses), 1)
            with self.assertRaisesRegex(ValueError, "random-stream audit failed"):
                audit.assert_clean()

    def test_malformed_keys_and_pairing_declarations_are_rejected(self):
        common = {
            "protocol_version": RANDOM_STREAM_PROTOCOL_VERSION,
            "campaign": "corrected-stage-6",
            "partition": "development",
            "block": 0,
            "purpose": "sensor_error",
        }
        with self.assertRaisesRegex(ValueError, "campaign"):
            self.observation(campaign=" ")
        with self.assertRaisesRegex(TypeError, "block"):
            self.observation(block=True)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.observation(block=-1)
        with self.assertRaisesRegex(ValueError, "require family"):
            RandomStreamKey(
                **common,
                stream_kind=OBSERVATION_STREAM,
                run="initial",
                channel="voltage",
            )
        with self.assertRaisesRegex(ValueError, "cannot carry"):
            RandomStreamKey(
                **common,
                stream_kind=DEVICE_TRUTH_STREAM,
                run="initial",
            )
        with self.assertRaisesRegex(ValueError, "stream_kind"):
            RandomStreamKey(**common, stream_kind="optimizer")
        with self.assertRaisesRegex(ValueError, "requires a pairing member"):
            StreamUse(
                self.observation(),
                "initial/cold-exchanger",
                pairing_id="common-initial",
            )


if __name__ == "__main__":
    unittest.main()
