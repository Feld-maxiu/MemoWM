import numpy as np
import torch

from residualmem.latent.instruct_bridge import (
    InputSoftTokenConnector,
    Layer16Connector,
    Layer16Restorer,
    MaskedAttentionRetrievalHead,
)
from experiments.state_tokenizer.extract_qwen import _input_text


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
