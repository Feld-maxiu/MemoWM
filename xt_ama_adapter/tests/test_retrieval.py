"""CPU-only tests for x_t vector indexing and trace preservation."""

import unittest

import numpy as np

from xt_ama_adapter import XTRetrievalIndex, adapt_ama_trajectory


def _vector(value: float) -> np.ndarray:
    vector = np.zeros(4096, dtype=np.float32)
    vector[0] = value
    vector[1] = 1.0 - abs(value)
    return vector


class TestXTRetrievalIndex(unittest.TestCase):
    def test_returns_original_ama_steps_in_stable_score_order(self):
        records = adapt_ama_trajectory(
            (
                {"turn_idx": 2, "action": "left", "observation": "alpha"},
                {"turn_idx": 3, "action": "up", "observation": "beta"},
                {"turn_idx": 4, "action": "right", "observation": "gamma"},
            ),
            episode_id="demo",
            task="demo task",
        )
        index = XTRetrievalIndex(records, np.stack((_vector(0.1), _vector(0.9), _vector(0.5))))
        results = index.retrieve_vector(_vector(0.95), top_k=2)

        self.assertEqual([item.record.step_index for item in results], [3, 4])
        self.assertEqual([item.rank for item in results], [1, 2])
        context = XTRetrievalIndex.reader_context(results)
        self.assertIn("AMA Step 3:\nAction: up", context)
        self.assertIn("AMA Step 4:\nAction: right", context)
        self.assertNotIn("AMA Step 2", context)

    def test_rejects_wrong_vector_dimension(self):
        records = adapt_ama_trajectory(
            ({"turn_idx": 0, "action": "wait", "observation": "state"},),
        )
        with self.assertRaises(ValueError):
            XTRetrievalIndex(records, np.zeros((1, 12), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
