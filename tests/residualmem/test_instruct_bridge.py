import numpy as np
import torch

from residualmem.latent.instruct_bridge import (
    InputSoftTokenConnector,
    Layer16Connector,
    Layer16Restorer,
    MaskedAttentionRetrievalHead,
    cache_metadata_json,
)
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
