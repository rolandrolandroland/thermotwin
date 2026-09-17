import random
import unittest
from unittest.mock import patch

from thermotwin.studies.operating_decision import (
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
)
from thermotwin.studies.operating_decision_prospective_random_streams import (
    PARAMETER_DRAW,
    PROBE_DRAW,
    RUN_BIAS,
    WHITE_NOISE,
    ProspectiveRandomStream,
    ProspectiveRandomStreamKey,
    ProspectiveRandomStreamNamespace,
    ProspectiveRandomStreamRegistry,
    prospective_observation_stream,
    prospective_parameter_stream,
    prospective_probe_stream,
)
from thermotwin.studies.sensor_model_discrimination import (
    COLD_EXCHANGER,
    COLD_FACE,
    FOUR_STATE_MODEL,
    FIVE_STATE_MODEL,
    VOLTAGE,
)


_DIGEST = "a" * 64


def _namespace(**changes):
    values = {
        "campaign": "prospective-disposable-2026-09",
        "partition": "step2-development",
        "block": 3,
        "acquisition_evidence_digest": _DIGEST,
    }
    values.update(changes)
    return ProspectiveRandomStreamNamespace(**values)


class OperatingDecisionProspectiveRandomStreamTests(unittest.TestCase):
    def test_namespace_is_strict_and_digest_bound(self):
        namespace = _namespace()
        self.assertEqual(namespace.block, 3)
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            _namespace(acquisition_evidence_digest="A" * 64)
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            _namespace(acquisition_evidence_digest="a" * 63)
        with self.assertRaisesRegex(TypeError, "block must be an integer"):
            _namespace(block=True)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _namespace(block=-1)
        with self.assertRaisesRegex(ValueError, "trimmed"):
            _namespace(partition=" step2-development")

    def test_parameter_probe_and_observation_keys_have_distinct_shapes(self):
        namespace = _namespace()
        parameter = prospective_parameter_stream(
            namespace,
            generator_model=FOUR_STATE_MODEL,
            draw_index=2,
        )
        self.assertEqual(parameter.key.purpose, PARAMETER_DRAW)
        self.assertIsNone(parameter.key.action)
        self.assertIsNone(parameter.key.run)
        self.assertIsNone(parameter.key.channel)

        probe = prospective_probe_stream(
            namespace,
            generator_model=FOUR_STATE_MODEL,
            draw_index=2,
        )
        self.assertEqual(probe.key.purpose, PROBE_DRAW)
        self.assertEqual(probe.key.action, FIXED_FACE_TEMPERATURE)
        self.assertIsNone(probe.key.run)
        self.assertIsNone(probe.key.channel)

        bias = prospective_observation_stream(
            namespace,
            generator_model=FOUR_STATE_MODEL,
            draw_index=2,
            purpose=RUN_BIAS,
            action=FIXED_VOLTAGE,
            run="repeat_0.8A_20s_voltage",
            channel=VOLTAGE,
        )
        noise = prospective_observation_stream(
            namespace,
            generator_model=FOUR_STATE_MODEL,
            draw_index=2,
            purpose=WHITE_NOISE,
            action=FIXED_VOLTAGE,
            run="repeat_0.8A_20s_voltage",
            channel=VOLTAGE,
        )
        self.assertNotEqual(bias.seed, noise.seed)

    def test_key_rejects_semantically_invalid_optional_coordinates(self):
        namespace = _namespace()
        fields = dict(
            protocol_version=namespace.protocol_version,
            campaign=namespace.campaign,
            partition=namespace.partition,
            block=namespace.block,
            acquisition_evidence_digest=namespace.acquisition_evidence_digest,
            generator_model=FOUR_STATE_MODEL,
            draw_index=0,
        )
        with self.assertRaisesRegex(ValueError, "must omit action"):
            ProspectiveRandomStreamKey(
                **fields,
                purpose=PARAMETER_DRAW,
                action=FIXED_THERMAL,
            )
        with self.assertRaisesRegex(ValueError, "face action"):
            ProspectiveRandomStreamKey(
                **fields,
                purpose=PROBE_DRAW,
                action=FIXED_VOLTAGE,
            )
        with self.assertRaisesRegex(ValueError, "require action, run, and channel"):
            ProspectiveRandomStreamKey(
                **fields,
                purpose=RUN_BIAS,
                action=FIXED_VOLTAGE,
            )
        with self.assertRaisesRegex(ValueError, "run does not belong"):
            ProspectiveRandomStreamKey(
                **fields,
                purpose=RUN_BIAS,
                action=FIXED_VOLTAGE,
                run="thermal_0.6A_30s",
                channel=COLD_EXCHANGER,
            )
        with self.assertRaisesRegex(ValueError, "channel does not belong"):
            ProspectiveRandomStreamKey(
                **fields,
                purpose=WHITE_NOISE,
                action=FIXED_THERMAL,
                run="thermal_0.6A_30s",
                channel=COLD_FACE,
            )
        with self.assertRaisesRegex(ValueError, "unknown generator"):
            prospective_parameter_stream(
                namespace,
                generator_model="hidden_truth_family",
                draw_index=0,
            )

    def test_semantic_coordinates_change_the_seed(self):
        namespace = _namespace()
        streams = (
            prospective_parameter_stream(
                namespace,
                generator_model=FOUR_STATE_MODEL,
                draw_index=0,
            ),
            prospective_parameter_stream(
                namespace,
                generator_model=FIVE_STATE_MODEL,
                draw_index=0,
            ),
            prospective_parameter_stream(
                namespace,
                generator_model=FOUR_STATE_MODEL,
                draw_index=1,
            ),
            prospective_parameter_stream(
                _namespace(block=4),
                generator_model=FOUR_STATE_MODEL,
                draw_index=0,
            ),
            prospective_parameter_stream(
                _namespace(acquisition_evidence_digest="b" * 64),
                generator_model=FOUR_STATE_MODEL,
                draw_index=0,
            ),
        )
        self.assertEqual(len({item.key for item in streams}), len(streams))
        self.assertEqual(len({item.seed for item in streams}), len(streams))

    def test_draw_prefix_is_stable_and_independent_of_global_random_state(self):
        namespace = _namespace()

        def sequence(count):
            return tuple(
                prospective_parameter_stream(
                    namespace,
                    generator_model=FOUR_STATE_MODEL,
                    draw_index=index,
                ).new_generator().random()
                for index in range(count)
            )

        random.seed(17)
        short = sequence(4)
        for _ in range(100):
            random.random()
        long = sequence(8)
        self.assertEqual(short, long[:4])
        self.assertEqual(short, sequence(4))

    def test_declared_parameter_sharing_across_actions_is_clean(self):
        stream = prospective_parameter_stream(
            _namespace(),
            generator_model=FOUR_STATE_MODEL,
            draw_index=0,
        )
        registry = ProspectiveRandomStreamRegistry()
        for action in (FIXED_THERMAL, FIXED_VOLTAGE, FIXED_FACE_TEMPERATURE):
            registry.register(
                stream,
                consumer=f"{action}/parameter-draw",
                shared_for_action=action,
            )
        audit = registry.audit()
        audit.assert_clean()
        self.assertTrue(audit.clean)
        self.assertEqual(audit.use_count, 3)
        self.assertEqual(audit.unique_key_count, 1)
        self.assertEqual(audit.unique_seed_count, 1)
        self.assertEqual(len(audit.declared_parameter_sharing), 1)
        self.assertEqual(
            audit.declared_parameter_sharing[0].actions,
            (FIXED_THERMAL, FIXED_VOLTAGE, FIXED_FACE_TEMPERATURE),
        )

    def test_exact_reuse_without_a_valid_sharing_declaration_fails(self):
        stream = prospective_parameter_stream(
            _namespace(),
            generator_model=FOUR_STATE_MODEL,
            draw_index=0,
        )
        registry = ProspectiveRandomStreamRegistry()
        registry.register(stream, consumer="first")
        registry.register(stream, consumer="second")
        audit = registry.audit()
        self.assertFalse(audit.clean)
        self.assertEqual(audit.unintended_reuse[0].reason, "undeclared_exact_key_reuse")
        with self.assertRaisesRegex(ValueError, "undeclared_exact_key_reuse"):
            audit.assert_clean()

        duplicated_action = ProspectiveRandomStreamRegistry()
        duplicated_action.register(
            stream,
            consumer="thermal/first",
            shared_for_action=FIXED_THERMAL,
        )
        duplicated_action.register(
            stream,
            consumer="thermal/second",
            shared_for_action=FIXED_THERMAL,
        )
        self.assertFalse(duplicated_action.audit().clean)

    def test_observation_and_probe_reuse_cannot_be_declared_as_shared(self):
        namespace = _namespace()
        probe = prospective_probe_stream(
            namespace,
            generator_model=FOUR_STATE_MODEL,
            draw_index=0,
        )
        with self.assertRaisesRegex(ValueError, "only parameter draws"):
            ProspectiveRandomStreamRegistry().register(
                probe,
                consumer="face/probe",
                shared_for_action=FIXED_FACE_TEMPERATURE,
            )

        noise = prospective_observation_stream(
            namespace,
            generator_model=FOUR_STATE_MODEL,
            draw_index=0,
            purpose=WHITE_NOISE,
            action=FIXED_FACE_TEMPERATURE,
            run="repeat_0.8A_20s_face_temperature",
            channel=COLD_FACE,
        )
        registry = ProspectiveRandomStreamRegistry()
        registry.register(noise, consumer="face/noise/first")
        registry.register(noise, consumer="face/noise/second")
        self.assertFalse(registry.audit().clean)

    def test_distinct_keys_with_the_same_derived_seed_are_a_collision(self):
        first = prospective_parameter_stream(
            _namespace(),
            generator_model=FOUR_STATE_MODEL,
            draw_index=0,
        )
        second = prospective_parameter_stream(
            _namespace(),
            generator_model=FIVE_STATE_MODEL,
            draw_index=0,
        )
        registry = ProspectiveRandomStreamRegistry()
        registry.register(first, consumer="four-state")
        registry.register(second, consumer="five-state")
        with patch(
            "thermotwin.studies.operating_decision_prospective_random_streams."
            "seed_from_prospective_key",
            return_value=7,
        ):
            audit = registry.audit()
        self.assertFalse(audit.clean)
        self.assertEqual(audit.unique_key_count, 2)
        self.assertEqual(audit.unique_seed_count, 1)
        self.assertEqual(audit.unintended_reuse[0].reason, "derived_seed_collision")

    def test_stream_and_namespace_types_are_not_interchangeable(self):
        namespace = _namespace()
        with self.assertRaisesRegex(TypeError, "namespace"):
            prospective_parameter_stream(  # type: ignore[arg-type]
                "not-a-namespace",
                generator_model=FOUR_STATE_MODEL,
                draw_index=0,
            )
        key = prospective_parameter_stream(
            namespace,
            generator_model=FOUR_STATE_MODEL,
            draw_index=0,
        ).key
        with self.assertRaisesRegex(TypeError, "prospective key"):
            ProspectiveRandomStream(key="not-a-key")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
