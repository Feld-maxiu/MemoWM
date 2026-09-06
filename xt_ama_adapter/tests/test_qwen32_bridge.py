import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.train_qwen32_bridge import (
    HiddenAnchorCapture,
    SemanticReconstructionDecoder,
    _shuffle_partner,
    hidden_delta_whitening,
    paired_ce_margin,
    semantic_reconstruction_loss,
    semantic_relation_loss,
)
from xt_ama_adapter.qwen32_bridge import (
    Qwen32InputSoftTokenBridge, Qwen32LatentReader,
    Qwen32RMSCalibratedInputSoftTokenBridge, latent_position_ids,
    load_qwen32_bridge, render_ama_openend_parts, save_qwen32_bridge,
)


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return f"<user>{messages[0]['content']}</user><assistant>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [3 + ord(character) % 61 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(map(str, ids))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 5120)

    def get_input_embeddings(self):
        return self.embedding


class _Block(torch.nn.Module):
    def forward(self, values):
        return (values + 1.0,)


class _HookModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block() for _ in range(4)])

    def forward(self, values):
        for layer in self.model.layers:
            values = layer(values)[0]
        return values


class TestQwen32Bridge(unittest.TestCase):
    def test_masked_latents_do_not_consume_reader_positions(self):
        mask = torch.tensor([[True, True, False, False, True, True]])
        self.assertTrue(torch.equal(
            latent_position_ids(mask), torch.tensor([[0, 1, 0, 0, 2, 3]])
        ))

    def test_shuffled_control_never_uses_the_same_trajectory(self):
        indices = np.asarray([0, 1, 2, 3])
        trajectories = np.asarray(["a", "a", "b", "c"])
        partner = _shuffle_partner(indices, trajectories)
        self.assertEqual(set(partner), set(indices.tolist()))
        for source, other in partner.items():
            self.assertNotEqual(trajectories[source], trajectories[other])

    def test_direct_reader_treats_topk_as_memory_ranks_not_batch(self):
        bridge = Qwen32InputSoftTokenBridge(max_memory_ranks=2)
        reader = Qwen32LatentReader(_Model(), _Tokenizer(), bridge)
        xbar = torch.randn(2, 32, 512)
        valid = torch.ones(2, 32, dtype=torch.bool)
        valid[1, -3:] = False
        inputs = reader.build_inputs("Which label is visible?", xbar, valid)
        mask = inputs["attention_mask"]
        self.assertEqual(mask.shape[0], 1)
        self.assertEqual(int((~mask).sum()), 3)
        self.assertEqual(inputs["inputs_embeds"].shape[-1], 5120)

    def test_shape_and_mask(self):
        bridge = Qwen32InputSoftTokenBridge(max_memory_ranks=3)
        xbar = torch.randn(2, 2, 32, 512)
        valid = torch.ones(2, 2, 32, dtype=torch.bool)
        valid[0, 1, -2:] = False
        output, mask = bridge(xbar, valid)
        self.assertEqual(tuple(output.shape), (2, 64, 5120))
        self.assertTrue(torch.equal(mask, valid.flatten(1, 2)))
        self.assertTrue(torch.equal(output[0, -2:], torch.zeros_like(output[0, -2:])))

    def test_content_states_precede_slot_and_rank_embeddings(self):
        bridge = Qwen32InputSoftTokenBridge(max_memory_ranks=2)
        with torch.no_grad():
            bridge.slot_embedding.normal_()
            bridge.memory_rank_embedding.normal_()
        xbar = torch.randn(2, 2, 32, 512)
        valid = torch.ones(2, 2, 32, dtype=torch.bool)
        valid[0, 1, -2:] = False
        output, mask, content = bridge(xbar, valid, return_content=True)
        expected = content.reshape(2, 2, 32, 5120)
        expected = expected + bridge.slot_embedding[None, None]
        expected = expected + bridge.memory_rank_embedding[None, :, None]
        expected = expected.flatten(1, 2) * mask[..., None]
        self.assertTrue(torch.allclose(output, expected))
        self.assertTrue(torch.equal(content[~mask], torch.zeros_like(content[~mask])))

    def test_rms_calibration_matches_reader_scale(self):
        target = 0.021
        bridge = Qwen32RMSCalibratedInputSoftTokenBridge(
            target_rms=target, max_memory_ranks=2
        )
        xbar = torch.randn(2, 2, 32, 512)
        valid = torch.ones(2, 2, 32, dtype=torch.bool)
        valid[0, 1, -2:] = False
        output, mask = bridge(xbar, valid)
        measured = output.float().square().mean(-1).sqrt()
        self.assertTrue(torch.allclose(
            measured[mask], torch.full_like(measured[mask], target), atol=1e-5
        ))
        self.assertTrue(torch.equal(
            output[~mask], torch.zeros_like(output[~mask])
        ))

    def test_semantic_losses_are_finite_and_backpropagate(self):
        bridge = Qwen32InputSoftTokenBridge(max_memory_ranks=1)
        decoder = SemanticReconstructionDecoder(dropout=0.0)
        xbar = torch.randn(2, 1, 32, 512)
        valid = torch.ones(2, 1, 32, dtype=torch.bool)
        _, _, content = bridge(xbar, valid, return_content=True)
        reconstruction, cosine, _mse = semantic_reconstruction_loss(
            decoder, content, xbar, valid, noise_ratio=0.01
        )
        relation = semantic_relation_loss(content, xbar, valid)
        loss = reconstruction + relation
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(cosine))
        self.assertIsNotNone(bridge.input_projection.weight.grad)
        self.assertIsNotNone(decoder.projection.weight.grad)

    def test_relation_loss_accepts_an_isometric_expansion(self):
        xbar = torch.randn(3, 1, 8, 512)
        valid = torch.ones(3, 1, 8, dtype=torch.bool)
        content = torch.nn.functional.pad(xbar, (0, 5120 - 512)).flatten(1, 2)
        loss = semantic_relation_loss(content, xbar, valid)
        self.assertLess(float(loss), 1e-7)

    def test_hidden_anchor_capture_uses_requested_positions_and_layers(self):
        model = _HookModel()
        capture = HiddenAnchorCapture(model, (1, 3))
        values = torch.zeros(2, 5, 7)
        capture.begin([1, 4])
        model(values)
        anchors = capture.stacked()
        capture.close()
        self.assertEqual(tuple(anchors.shape), (2, 2, 7))
        self.assertTrue(torch.equal(anchors[:, 0], torch.ones(2, 7)))
        self.assertTrue(torch.equal(anchors[:, 1], torch.full((2, 7), 3.0)))

    def test_hidden_delta_whitening_centers_teacher_deltas(self):
        teacher, question = {}, {}
        indices = np.asarray([1, 2, 3])
        for index, offset in zip(indices, (-1.0, 0.0, 1.0)):
            question[int(index)] = torch.full((2, 5), 10.0)
            teacher[int(index)] = {
                "hidden_anchor": question[int(index)] + offset
                * torch.arange(1, 6).repeat(2, 1)
            }
        mean, scale, floor = hidden_delta_whitening(
            teacher, question, indices, floor_ratio=0.1
        )
        delta = torch.stack([
            teacher[int(index)]["hidden_anchor"] - question[int(index)]
            for index in indices
        ])
        whitened = (delta - mean) / scale
        self.assertTrue(torch.allclose(whitened.mean(0), torch.zeros_like(mean)))
        self.assertTrue(bool((scale > 0).all()))
        self.assertEqual(tuple(floor.shape), (2,))

    def test_margin_hinge_is_applied_before_batch_mean(self):
        matched = torch.tensor([1.0, 3.0])
        shuffled = torch.tensor([1.2, 2.0])
        loss = paired_ce_margin(matched, shuffled, margin=0.1)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.55)))

    def test_artifact_hash_binding(self):
        bridge = Qwen32InputSoftTokenBridge(max_memory_ranks=1)
        values = {name: str(index) * 64 for index, name in enumerate(
            ["qformer", "head", "reader", "data", "prompt"], start=1)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge.pt"
            save_qwen32_bridge(
                path, bridge, qformer_artifact_sha256=values["qformer"],
                retrieval_head_artifact_sha256=values["head"],
                reader_model_sha256=values["reader"],
                data_manifest_sha256=values["data"],
                prompt_sha256_value=values["prompt"],
            )
            loaded, metadata = load_qwen32_bridge(
                path, expected_qformer_sha256=values["qformer"],
                expected_retrieval_head_sha256=values["head"])
            self.assertEqual(loaded.slots, 32)
            self.assertEqual(metadata["model_dim"], 5120)
            with self.assertRaisesRegex(ValueError, "QFormer"):
                load_qwen32_bridge(path, expected_qformer_sha256="f" * 64)


class TestBuildInputsSegments(unittest.TestCase):
    def _reader(self, max_memory_ranks: int = 2):
        bridge = Qwen32InputSoftTokenBridge(max_memory_ranks=max_memory_ranks)
        return Qwen32LatentReader(_Model(), _Tokenizer(), bridge)

    def _prefix_length(self, reader, question):
        prefix, _suffix, _hash = render_ama_openend_parts(
            reader.tokenizer, question, enable_thinking=reader.enable_thinking
        )
        return len(reader.tokenizer.encode(prefix, add_special_tokens=False))

    def test_rank_order_is_carried_through_memory_rank_embeddings(self):
        reader = self._reader()
        question = "Which label is visible?"
        xbar_a, xbar_b = torch.randn(32, 512), torch.randn(32, 512)
        valid = torch.ones(32, dtype=torch.bool)
        offset = self._prefix_length(reader, question)
        latent_a_rank0 = reader.bridge(
            torch.stack([xbar_a])[None], valid[None, None])[0][0]
        latent_b_rank0 = reader.bridge(
            torch.stack([xbar_b])[None], valid[None, None])[0][0]
        inputs_ab, audit_ab = reader.build_inputs_segments(
            question, [("latent", (xbar_a, valid)), ("latent", (xbar_b, valid))])
        inputs_ba, _audit = reader.build_inputs_segments(
            question, [("latent", (xbar_b, valid)), ("latent", (xbar_a, valid))])
        self.assertEqual(audit_ab["latent_memories"], 2)
        self.assertTrue(torch.equal(
            inputs_ab["inputs_embeds"][0, offset:offset + 32], latent_a_rank0))
        self.assertTrue(torch.equal(
            inputs_ba["inputs_embeds"][0, offset:offset + 32], latent_b_rank0))

    def test_invalid_slots_do_not_advance_positions_or_gain_attention(self):
        reader = self._reader()
        valid = torch.ones(32, dtype=torch.bool)
        valid[-3:] = False
        inputs, _audit = reader.build_inputs_segments(
            "q?", [("latent", (torch.randn(32, 512), valid))])
        mask = inputs["attention_mask"]
        self.assertEqual(int((~mask).sum()), 3)
        position_ids = inputs["position_ids"]
        self.assertTrue(torch.equal(
            position_ids[mask], torch.arange(int(mask.sum()))))

    def test_budget_truncation_shortens_text_only(self):
        reader = self._reader()
        valid = torch.ones(32, dtype=torch.bool)
        latent_state = torch.randn(32, 512)
        question = "q?"
        offset = self._prefix_length(reader, question)
        full, audit_full = reader.build_inputs_segments(
            question, [("latent", (latent_state, valid)),
                       ("text", "word " * 400)],
            max_model_len=32000, max_new_tokens=8192)
        self.assertFalse(audit_full["prompt_truncated"])
        tight, audit = reader.build_inputs_segments(
            question, [("latent", (latent_state, valid)),
                       ("text", "word " * 400)],
            max_model_len=1000, max_new_tokens=50)
        self.assertTrue(audit["prompt_truncated"])
        self.assertLess(audit["text_tokens_after"], audit["text_tokens_before"])
        self.assertEqual(tight["attention_mask"].shape[1], audit["prompt_budget"])
        # The latent block is a fixed cost and must survive text truncation.
        self.assertTrue(torch.equal(
            tight["inputs_embeds"][0, offset:offset + 32],
            full["inputs_embeds"][0, offset:offset + 32]))

    def test_text_only_and_degenerate_segments(self):
        reader = self._reader()
        inputs, audit = reader.build_inputs_segments(
            "q?", [("text", "row one"), ("text", "row two")])
        self.assertEqual(audit["latent_memories"], 0)
        self.assertEqual(audit["text_rows"], 2)
        self.assertEqual(inputs["inputs_embeds"].shape[-1], 5120)
        with self.assertRaisesRegex(ValueError, "unknown memory segment kind"):
            reader.build_inputs_segments("q?", [("image", ())])
        with self.assertRaisesRegex(ValueError, "exceeds budget"):
            reader.build_inputs_segments("q?", [("text", "x")],
                                         max_model_len=120, max_new_tokens=50)

    def test_bridge_rank_clamp_is_enforced_by_the_bridge(self):
        reader = self._reader(max_memory_ranks=2)
        valid = torch.ones(32, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "maximum is 2"):
            reader.build_inputs_segments("q?", [
                ("latent", (torch.randn(32, 512), valid)) for _ in range(3)])


if __name__ == "__main__":
    unittest.main()
