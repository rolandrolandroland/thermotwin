import unittest

from thermotwin.small_matrix import (
    gram_matrix,
    inverse_and_determinant,
    matrix_multiply,
)


class SmallMatrixTests(unittest.TestCase):
    def test_inverse_and_determinant(self):
        inverse, determinant = inverse_and_determinant(((4.0, 7.0), (2.0, 6.0)))

        self.assertAlmostEqual(determinant, 10.0)
        identity = matrix_multiply(((4.0, 7.0), (2.0, 6.0)), inverse)
        self.assertAlmostEqual(identity[0][0], 1.0)
        self.assertAlmostEqual(identity[0][1], 0.0)
        self.assertAlmostEqual(identity[1][0], 0.0)
        self.assertAlmostEqual(identity[1][1], 1.0)

    def test_gram_matrix(self):
        gram = gram_matrix(((1.0, 2.0), (3.0, 4.0)))
        self.assertEqual(gram, ((10.0, 14.0), (14.0, 20.0)))

    def test_singular_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            inverse_and_determinant(((1.0, 2.0), (2.0, 4.0)))


if __name__ == "__main__":
    unittest.main()
