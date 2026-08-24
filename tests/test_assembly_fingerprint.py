import unittest

from thermotwin.assembly_fingerprint import (
    AssemblySpecification,
    run_assembly_fingerprint_study,
)


class AssemblyFingerprintTests(unittest.TestCase):
    def test_fingerprint_separates_reference_and_elevated_loss(self):
        result = run_assembly_fingerprint_study(
            (
                AssemblySpecification("reference", 0.25),
                AssemblySpecification("elevated", 0.45),
            )
        )

        reference, elevated = result.fingerprints
        self.assertEqual(reference.classification, "reference_band")
        self.assertEqual(elevated.classification, "elevated_interface_loss")
        self.assertLess(
            reference.inferred_cold_contact_resistance,
            elevated.inferred_cold_contact_resistance,
        )
        self.assertTrue(reference.truth_covered)
        self.assertTrue(elevated.truth_covered)

    def test_duplicate_assembly_names_are_rejected(self):
        with self.assertRaises(ValueError):
            run_assembly_fingerprint_study(
                (
                    AssemblySpecification("duplicate", 0.2),
                    AssemblySpecification("duplicate", 0.3),
                )
            )


if __name__ == "__main__":
    unittest.main()
