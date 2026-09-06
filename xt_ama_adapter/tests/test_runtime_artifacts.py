import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from xt_ama_adapter.runtime import (
    QFORMER_PROTOCOL,
    QWEN_VL_FUSED_OBSERVATION_PROTOCOL,
    RETRIEVAL_HEAD_PROTOCOL,
    QFormerRuntimeConfig,
    validate_artifact_pair,
)


class TestRuntimeArtifacts(unittest.TestCase):
    def _artifacts(self, directory: str):
        root = Path(directory)
        model = root / "qwen35"
        query = root / "query"
        model.mkdir()
        query.mkdir()
        qformer = root / "qformer.pt"
        torch.save({
            "protocol": QFORMER_PROTOCOL,
            "metadata": {"queries": 32, "qk_norm": True,
                         "self_attention": False},
            "state_dict": {},
        }, qformer)
        artifact_hash = hashlib.sha256(qformer.read_bytes()).hexdigest()
        head = root / "head.pt"
        torch.save({
            "protocol": RETRIEVAL_HEAD_PROTOCOL,
            "metadata": {
                "teacher_protocol": QWEN_VL_FUSED_OBSERVATION_PROTOCOL,
                "qformer_artifact_sha256": artifact_hash,
                "qk_norm": True,
                "queries": 32,
            },
            "state_dict": {"projection.2.weight": torch.empty(4096, 1024)},
        }, head)
        return QFormerRuntimeConfig(
            residualmem_root=str(root), qwen35_model_path=str(model),
            qformer_checkpoint=str(qformer), retrieval_head_checkpoint=str(head),
            query_model_path=str(query),
        ), qformer

    def test_validates_exact_hash_and_qk_norm(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _qformer = self._artifacts(directory)
            report = validate_artifact_pair(config)
            self.assertEqual(report["qformer_queries"], 32)
            bad = QFormerRuntimeConfig(**{
                **config.__dict__, "qk_norm": False,
            })
            with self.assertRaisesRegex(ValueError, "qk_norm"):
                validate_artifact_pair(bad)

    def test_rejects_head_bound_to_a_different_qformer_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config, qformer = self._artifacts(directory)
            payload = torch.load(qformer, map_location="cpu", weights_only=True)
            payload["metadata"]["extra"] = "changes the artifact hash"
            torch.save(payload, qformer)
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_artifact_pair(config)


if __name__ == "__main__":
    unittest.main()
