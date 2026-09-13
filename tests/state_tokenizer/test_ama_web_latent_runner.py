import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.state_tokenizer.run_ama_web_latent import (
    CACHE_PROTOCOL, CACHE_PROTOCOL_V1, TEXT_MEMORY_MODES,
    _episode_ids, _load_cache_payloads, _shuffled_episode_partners,
    shard_episodes, web_episodes,
)


class TestAMAWebLatentRunner(unittest.TestCase):
    def test_web_filter_and_shards_are_disjoint(self):
        rows = [
            {"episode_id": 3, "domain": "WEB"},
            {"episode_id": 1, "domain": "Game"},
            {"episode_id": 2, "domain": "web"},
            {"episode_id": 8, "domain": "WEB"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            web = web_episodes(path)
        self.assertEqual([row["episode_id"] for row in web], [2, 3, 8])
        shards = [shard_episodes(web, index, 3) for index in range(3)]
        self.assertEqual([[row["episode_id"] for row in shard] for shard in shards],
                         [[2], [3], [8]])

    def test_episode_id_parser(self):
        self.assertEqual(_episode_ids("184, 205"), {184, 205})
        self.assertEqual(_episode_ids(""), set())

    def test_shuffled_episode_partners_are_balanced_and_never_self(self):
        episode_ids = [11, 13, 17, 19, 23]
        partners = _shuffled_episode_partners(episode_ids, seed=35)
        self.assertEqual(set(partners), set(episode_ids))
        self.assertEqual(set(partners.values()), set(episode_ids))
        self.assertTrue(all(source != donor for source, donor in partners.items()))
        self.assertEqual(partners, _shuffled_episode_partners(episode_ids, seed=35))


class TestCachePayloadLoading(unittest.TestCase):
    def _payload(self, directory: Path, name: str, *, protocol: str,
                 with_texts: bool, episode_id: int = 7):
        payload = {
            "protocol": protocol,
            "metadata": {
                "episode_id": episode_id,
                "qformer_sha256": "q" * 64,
                "retrieval_head_sha256": "h" * 64,
            },
            "step_indices": torch.tensor([0, 1]),
        }
        if with_texts:
            payload["step_texts"] = ["AMA Step 0:", "AMA Step 1:"]
        target = directory / name
        torch.save(payload, target)
        return target

    def test_v2_cache_loads_for_text_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._payload(root, "episode-000007.pt", protocol=CACHE_PROTOCOL,
                          with_texts=True)
            payloads = _load_cache_payloads(
                sorted(root.glob("episode-*.pt")),
                qformer_hash="q" * 64, head_hash="h" * 64, needs_texts=True)
            self.assertEqual(payloads[7]["step_texts"][1], "AMA Step 1:")

    def test_v1_cache_is_rejected_for_text_modes_and_accepted_without(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._payload(root, "episode-000007.pt", protocol=CACHE_PROTOCOL_V1,
                          with_texts=False)
            files = sorted(root.glob("episode-*.pt"))
            with self.assertRaisesRegex(ValueError, "step_texts"):
                _load_cache_payloads(files, qformer_hash="q" * 64,
                                     head_hash="h" * 64, needs_texts=True)
            payloads = _load_cache_payloads(
                files, qformer_hash="q" * 64, head_hash="h" * 64,
                needs_texts=False)
            self.assertEqual(len(payloads), 1)

    def test_hash_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._payload(root, "episode-000007.pt", protocol=CACHE_PROTOCOL,
                          with_texts=True)
            with self.assertRaisesRegex(ValueError, "QFormer cache mismatch"):
                _load_cache_payloads(
                    sorted(root.glob("episode-*.pt")),
                    qformer_hash="x" * 64, head_hash="h" * 64, needs_texts=False)

    def test_text_modes_and_protocols(self):
        self.assertEqual(TEXT_MEMORY_MODES, ("text-only", "matched", "shuffled", "latent+anchor"))
        self.assertNotEqual(CACHE_PROTOCOL, CACHE_PROTOCOL_V1)


if __name__ == "__main__":
    unittest.main()
