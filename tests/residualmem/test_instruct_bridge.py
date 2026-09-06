import numpy as np
import torch

from residualmem.latent.instruct_bridge import (
    InputSoftTokenConnector,
    Layer16Connector,
    Layer16Restorer,
    MaskedAttentionRetrievalHead,
    cache_metadata_json,
)
from experiments.state_tokenizer.build_utility_retrieval_cache import build_views
from experiments.state_tokenizer.extract_qwen import _input_text
from experiments.state_tokenizer.train_retrieval_bridge import batch_loss, load_cache


def _state(batch=2):
    torch.manual_seed(7)
    xbar = torch.randn(batch, 64, 512)
    valid = torch.ones(batch, 64, dtype=torch.bool)
    valid[:, -5:] = False
    xbar[:, -5:] = 0
    return xbar, valid


def test_retrieval_head_normalizes_and_ignores_padding():
    xbar, valid = _state()
    head = MaskedAttentionRetrievalHead()
    first = head(xbar, valid)
    changed = xbar.clone()
    changed[:, -5:] = 1e6
    second = head(changed, valid)
    assert first.shape == (2, 4096)
    assert torch.allclose(first.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(first, second, atol=1e-6)


def test_xbar_only_bridge_cache_and_loss_need_no_a2(tmp_path):
    rng = np.random.default_rng(11)
    xbar = rng.normal(size=(3, 64, 512)).astype(np.float32)
    valid = np.ones((3, 64), dtype=np.bool_)
    teacher = rng.normal(size=(3, 4096)).astype(np.float32)
    teacher /= np.linalg.norm(teacher, axis=-1, keepdims=True)
    path = tmp_path / "xbar-only.npz"
    np.savez_compressed(
        path,
        xbar=xbar,
        valid=valid,
        teacher_fused_embedding=teacher,
        split=np.asarray(["train", "train", "validation"]),
        metadata=np.asarray(cache_metadata_json(representation="xbar")),
    )

    cache = load_cache(path)
    assert cache["metadata"]["representation"] == "xbar"
    assert "a2_xbar" not in cache
    loss, metrics = batch_loss(
        MaskedAttentionRetrievalHead(),
        torch.from_numpy(xbar),
        torch.from_numpy(valid),
        torch.from_numpy(teacher),
        0.05,
    )
    assert torch.isfinite(loss)
    assert set(metrics) == {"contrastive", "cosine", "xbar_cosine"}


def test_utility_gated_bridge_cache_and_loss_use_both_reconstructions(tmp_path):
    rng = np.random.default_rng(13)
    xbar = rng.normal(size=(3, 32, 512)).astype(np.float32)
    full = xbar + rng.normal(scale=0.01, size=xbar.shape).astype(np.float32)
    gated = full + rng.normal(scale=0.02, size=xbar.shape).astype(np.float32)
    valid = np.ones((3, 32), dtype=np.bool_)
    teacher = rng.normal(size=(3, 4096)).astype(np.float32)
    teacher /= np.linalg.norm(teacher, axis=-1, keepdims=True)
    path = tmp_path / "utility-gated.npz"
    np.savez_compressed(
        path,
        xbar=xbar,
        full_recon_xbar=full,
        gated_recon_xbar=gated,
        valid=valid,
        teacher_fused_embedding=teacher,
        split=np.asarray(["train", "train", "validation"]),
        metadata=np.asarray(cache_metadata_json(representation="utility_gated")),
    )

    cache = load_cache(path)
    assert cache["metadata"]["representation"] == "utility_gated"
    assert cache["full_recon_xbar"].shape == xbar.shape
    head = MaskedAttentionRetrievalHead()
    loss, metrics = batch_loss(
        head,
        torch.from_numpy(xbar),
        torch.from_numpy(valid),
        torch.from_numpy(teacher),
        0.05,
        full_recon=torch.from_numpy(full),
        gated_recon=torch.from_numpy(gated),
        consistency_weight=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in head.parameters())
    assert set(metrics) == {
        "contrastive", "cosine", "consistency",
        "full_recon_cosine", "gated_recon_cosine",
    }


def test_build_utility_views_uses_all_send_without_a_posterior():
    codes = np.ones((2, 2, 2), np.uint8)
    centroids = np.zeros((2, 2, 2, 2), np.float32)
    centroids[:, :, 1] = 1.0
    book = {
        "mean": np.zeros((2, 4), np.float32),
        "scale": np.ones((2, 4), np.float32),
        "centroids": centroids,
        "bases": None,
        "order": None,
    }
    posterior = {
        "target_indices": np.asarray([1]),
        "target_codes": codes[1:2],
        "entropy_bits": np.ones((1, 2, 2), np.float32),
        "wm_argmax": np.zeros((1, 2, 2), np.uint8),
    }
    full, gated, diagnostics = build_views(
        ["initial", "transition"], codes, ["initial", "transition"],
        {"initial": 0, "transition": 1}, posterior,
        utility=np.zeros(4), lam=0.001, book=book,
    )

    assert np.array_equal(full[0], gated[0])
    assert np.count_nonzero(full[1]) == full[1].size
    assert np.count_nonzero(gated[1]) == 0
    assert diagnostics["posterior_states"] == 1
    assert diagnostics["all_send_states"] == 1


def test_input_connector_has_exact_zero_padding():
    xbar, valid = _state()
    output = InputSoftTokenConnector()(xbar, valid)
    assert output.shape == (2, 64, 4096)
    assert torch.count_nonzero(output[:, -5:]) == 0


def test_layer16_zero_adapter_is_exact_restorer():
    restorer = Layer16Restorer(
        slot_mean=np.zeros((64, 512), np.float32),
        slot_scale=np.ones((64, 512), np.float32),
        pca_mean=np.zeros((4096,), np.float32),
        components=np.eye(4096, 512, dtype=np.float32),
    )
    connector = Layer16Connector(restorer)
    xbar, valid = _state(batch=1)
    assert torch.equal(connector(xbar, valid), restorer(xbar, valid))


def test_instruct_prompt_is_explicit_and_base_stays_unchanged():
    class Processor:
        vision_start_token = "<vs>"
        image_token = "<image>"
        vision_end_token = "</vs>"

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0] == {"type": "image"}
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "CHAT:" + messages[0]["content"][1]["text"]

    processor = Processor()
    base = _input_text(processor, "DOM", "TASK")
    instruct = _input_text(processor, "DOM", "TASK", "instruct")
    assert base.startswith("<vs><image></vs>\n")
    assert instruct.startswith("CHAT:")
    assert "DOM" in instruct and "TASK" in instruct
