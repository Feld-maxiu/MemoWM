"""Regression tests for HumanTrajs alignment and leakage controls."""
import unittest

from xt_ama_adapter.humantrajs import (
    align_trajectories, normalized_qa_key, sanitize_action, stable_split,
)


def inspector(path: str) -> dict:
    return {"exists": "missing" not in path, "low_information": "black" in path,
            "sha256": path.replace("duplicate", "same")}


def row(step, action_name="click", image=None, trajectory="traj-a"):
    return {"trajectory_id": trajectory, "step_idx": step,
            "action": {"action_str": f"do-{step}", "action_description": "visible action",
                       "action_output": {"action_name": action_name,
                                         "thought": "SECRET FUTURE ANSWER"}},
            "observation": {"url": f"https://example.test/{step}", "private": "drop"},
            "image_path": image or f"image-{step}.png", "instruction": "task"}


class TestHumanTrajsPreparation(unittest.TestCase):
    def test_action_is_aligned_to_next_screenshot_and_thought_is_removed(self):
        prepared, rejected = align_trajectories([row(1), row(2), row(3)], inspector)
        self.assertEqual([item["step_idx"] for item in prepared], [1, 2])
        self.assertEqual(prepared[0]["before_image_path"], "image-1.png")
        self.assertEqual(prepared[0]["image_path"], "image-2.png")
        self.assertNotIn("SECRET", str(prepared[0]["action"]))
        self.assertEqual(prepared[1]["previous_memory_step_idx"], 1)
        self.assertEqual(rejected[0]["reason"], "no_post_action_frame")

    def test_terminal_and_black_post_frames_are_rejected(self):
        rows = [row(1, "send_msg_to_user"), row(2), row(3, image="black.png")]
        prepared, rejected = align_trajectories(rows, inspector)
        self.assertEqual(prepared, [])
        self.assertEqual({item["reason"] for item in rejected},
                         {"terminal_action", "post_image_low_information", "no_post_action_frame"})

    def test_duplicate_states_become_multi_positive_labels(self):
        rows = [row(1), row(2, image="duplicate.png"), row(3, image="same.png"), row(4)]
        prepared, _ = align_trajectories(rows, inspector)
        self.assertEqual(prepared[0]["equivalent_step_ids"], [1, 2])
        self.assertEqual(prepared[1]["equivalent_step_ids"], [1, 2])

    def test_split_is_group_stable(self):
        self.assertEqual(stable_split("same-trajectory"), stable_split("same-trajectory"))

    def test_sanitize_action_has_allowlist(self):
        clean = sanitize_action(row(1)["action"])
        self.assertEqual(set(clean), {"action_name", "action_str", "action_description"})

    def test_page_dump_is_removed_from_action_description(self):
        action = row(1)["action"]
        action["action_description"] = "Click on HTML element with value '" + "page text " * 100
        clean = sanitize_action(action)
        self.assertEqual(clean["action_description"], "Click on HTML element")

    def test_qa_dedup_key_ignores_case_spacing_and_terminal_punctuation(self):
        self.assertEqual(normalized_qa_key("What  is shown?", " GitHub. "),
                         normalized_qa_key("what is shown", "github"))


if __name__ == "__main__":
    unittest.main()
