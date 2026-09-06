from __future__ import annotations

import unittest

import numpy as np

from experiments.state_tokenizer.build_molmoweb_text_qformer_inputs import (
    fixed_text_baseline,
)


class MolmoWebTextInputsTest(unittest.TestCase):
    def test_collapse_reference_is_deterministic_normalized_and_text_sensitive(self):
        first = fixed_text_baseline("Title Example price $12.50", slots=32)
        again = fixed_text_baseline("Title Example price $12.50", slots=32)
        other = fixed_text_baseline("A completely different page", slots=32)

        self.assertEqual(first.shape, (32, 512))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))
        occupied = np.any(first != 0, axis=1)
        np.testing.assert_allclose(np.linalg.norm(first[occupied], axis=1), 1.0)
        self.assertTrue(np.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
