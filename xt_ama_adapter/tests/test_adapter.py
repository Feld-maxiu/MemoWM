"""Regression tests for the standalone AMA -> x_t data adapter."""

import json
import unittest
from pathlib import Path

from xt_ama_adapter import (
    AMA_XT_OBSERVATION_PROTOCOL,
    adapt_ama_step,
    adapt_ama_trajectory,
)


class TestAMAXTAdapter(unittest.TestCase):
    def test_step_keeps_action_and_observation_in_canonical_evidence(self):
        record = adapt_ama_step(
            {"turn_idx": 7, "action": "down", "observation": "Baba is here."},
            episode_id=12,
            task="Solve the grid puzzle.",
        )

        self.assertEqual(record.protocol, AMA_XT_OBSERVATION_PROTOCOL)
        self.assertEqual(record.episode_id, "12")
        self.assertEqual(record.step_index, 7)
        self.assertEqual(record.user_text, "AMA task:\nSolve the grid puzzle.")
        self.assertEqual(
            record.step_text,
            "AMA Step 7:\nAction: down\nObservation:\nBaba is here.",
        )
        self.assertEqual(
            record.worldmem_payload(),
            {
                "screenshot": None,
                "user_text": "AMA task:\nSolve the grid puzzle.",
                "captions": (record.step_text,),
                "image_ids": (),
            },
        )

    def test_trajectory_keeps_one_record_per_ama_step(self):
        trajectory = (
            {"turn_idx": 4, "action": "left", "observation": "first"},
            {"turn_idx": 5, "action": "right", "observation": "second"},
        )
        records = adapt_ama_trajectory(trajectory, episode_id="ep-a", task="task")

        self.assertEqual([record.step_index for record in records], [4, 5])
        self.assertEqual([record.action for record in records], ["left", "right"])
        self.assertEqual(
            set(records[0].trace_record()),
            {"protocol", "episode_id", "step_index", "action", "observation", "step_text"},
        )

    def test_real_ama_episode_has_a_stable_record_per_step(self):
        root = Path(__file__).resolve().parents[2]
        dataset = (root / "third_party" / "AMA-Bench" / "dataset" / "test"
                   / "open_end_qa_set.jsonl")
        with dataset.open(encoding="utf-8") as handle:
            episode = json.loads(next(handle))

        records = adapt_ama_trajectory(
            episode["trajectory"],
            episode_id=episode["episode_id"],
            task=episode["task"],
        )
        self.assertEqual(len(records), len(episode["trajectory"]))
        self.assertEqual(records[0].step_index, episode["trajectory"][0]["turn_idx"])
        self.assertEqual(records[0].action, episode["trajectory"][0]["action"])
        self.assertEqual(records[0].observation, episode["trajectory"][0]["observation"])


if __name__ == "__main__":
    unittest.main()
