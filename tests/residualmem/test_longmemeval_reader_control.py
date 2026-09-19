from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.state_tokenizer.evaluate_longmemeval_reader_control import fixed_context, raw_messages
from experiments.state_tokenizer.longmemeval_reader import LME_WEB_SYSTEM_PROMPT
from residualmem.benchmarks.longmemeval_index import RetrievedObservation


def test_raw_control_preserves_question_image_observation_order_and_full_text(tmp_path):
    observations = []
    for index in range(2):
        path = tmp_path / f"state-{index}.png"
        Image.new("RGB", (8, 8)).save(path)
        observations.append(SimpleNamespace(
            trajectory_id="trajectory", anchor_text=f"anchor-{index}",
            record={"screenshot": str(path), "axtree": f"tree-{index}" + "x" * 16000 + "END"},
        ))
    messages, paths = raw_messages("question", observations, "question.png")
    assert paths == ["question.png", str(tmp_path / "state-0.png"), str(tmp_path / "state-1.png")]
    assert messages[0]["content"] == LME_WEB_SYSTEM_PROMPT
    content = messages[1]["content"]
    assert sum(item["type"] == "image" for item in content) == 3
    text = "".join(item.get("text", "") for item in content)
    assert text.index("tree-0") < text.index("anchor-0") < text.index("tree-1") < text.index("anchor-1")
    assert text.count("x" * 16000 + "END") == 2
    assert "### Question to answer:\nquestion" in text


def test_replayed_context_rejects_wrong_haystack_and_changed_context_count():
    row = RetrievedObservation(
        "trajectory", 0, 0.0, np.zeros((2, 3), np.float32),
        np.ones(2, bool), np.ones(4, np.float32), "", {},
    )
    index = SimpleNamespace(expanded_context=lambda hits, **kwargs: hits)
    lookup = {("trajectory", 0): row}
    baseline = {"hits": [{"trajectory_id": "trajectory", "record_index": 0, "score": 0.9}],
                "context_observations": 1}
    assert fixed_context(index, lookup, baseline, ["trajectory"])[0].score == 0.9
    with pytest.raises(ValueError, match="outside the official haystack"):
        fixed_context(index, lookup, baseline, ["other"])
    with pytest.raises(ValueError, match="context count"):
        fixed_context(index, lookup, {**baseline, "context_observations": 2}, ["trajectory"])


class _CompletionTokenizer:
    def encode(self, value, **kwargs):
        assert value == "</think>"
        return [99]

    def decode(self, values, **kwargs):
        return "".join({1: r"Considering \boxed{wrong}.", 2: r"\boxed{right}",
                        3: "plain answer"}[value] for value in values)


def test_thinking_scores_only_final_answer_not_boxed_reasoning():
    from experiments.state_tokenizer.longmemeval_reader import decode_reader_completion
    answer, diagnostics = decode_reader_completion(
        _CompletionTokenizer(), [1, 99, 2], enable_thinking=True, max_new_tokens=20000,
    )
    assert answer == r"\boxed{right}"
    assert diagnostics["reasoning_text"] == r"Considering \boxed{wrong}."
    assert diagnostics["thinking_completed"] is True
    assert diagnostics["answer_tokens"] == 1


def test_unfinished_thinking_falls_back_to_reasoning_text():
    from experiments.state_tokenizer.longmemeval_reader import decode_reader_completion
    answer, diagnostics = decode_reader_completion(
        _CompletionTokenizer(), [1], enable_thinking=True, max_new_tokens=1,
    )
    assert answer == r"Considering \boxed{wrong}."
    assert diagnostics["thinking_completed"] is False
    assert diagnostics["fallback_from_reasoning"] is True
    assert diagnostics["generation_limit_reached"] is True


def test_legacy_non_thinking_completion_is_unchanged():
    from experiments.state_tokenizer.longmemeval_reader import decode_reader_completion
    answer, diagnostics = decode_reader_completion(
        _CompletionTokenizer(), [3], enable_thinking=False, max_new_tokens=512,
    )
    assert answer == "plain answer"
    assert diagnostics["reasoning_tokens"] == 0


def test_runner_defaults_to_thinking_with_a_complete_generation_budget():
    from experiments.state_tokenizer.run_longmemeval_local import argument_parser
    required = [value for flag in ("--data-root", "--cache", "--model", "--checkpoint",
                                  "--embedding-model", "--output") for value in (flag, "unused")]
    args = argument_parser().parse_args(required)
    assert args.reader_enable_thinking is True
    assert args.max_new_tokens == 20000
    assert (args.reader_temperature, args.reader_top_p, args.reader_top_k) == (0.6, 0.95, 20)
    assert args.top_k == 8                                                       
    custom = argument_parser().parse_args(required + ["--reader-temperature", "0.7",
        "--reader-top-p", "0.9", "--reader-top-k", "12", "--top-k", "6"])
    assert (custom.reader_temperature, custom.reader_top_p, custom.reader_top_k, custom.top_k) == (0.7, 0.9, 12, 6)
    old = argument_parser().parse_args(required + ["--no-reader-enable-thinking", "--max-new-tokens", "512",
                                                  "--reader-temperature", "0"])
    assert old.reader_enable_thinking is False and old.max_new_tokens == 512
    assert old.reader_temperature == 0


def test_merge_rejects_mixed_generation_settings_and_legacy_rows():
    from experiments.state_tokenizer.merge_longmemeval_eval import shared_generation_config
    config = {"enable_thinking": True, "max_new_tokens": 20000, "do_sample": True,
              "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert shared_generation_config([{}, {}]) is None
    assert shared_generation_config([{"generation": config}] * 2) == config
    with pytest.raises(ValueError, match="generation configurations"):
        shared_generation_config([{}, {"generation": config}])
    for key, value in (("enable_thinking", False), ("do_sample", False), ("temperature", 0.7),
                       ("top_p", 0.9), ("top_k", 50)):
        with pytest.raises(ValueError, match="generation configurations"):
            shared_generation_config([{"generation": config}, {"generation": {**config, key: value}}])


@pytest.mark.parametrize("overrides", [
    {"temperature": -1}, {"temperature": float("nan")}, {"temperature": float("inf")},
    {"top_p": 0}, {"top_p": 1.1}, {"top_p": float("nan")}, {"top_k": -1}, {"top_k": 1.5},
])
def test_reader_rejects_invalid_sampling(overrides):
    from experiments.state_tokenizer.longmemeval_reader import reader_sampling_kwargs
    with pytest.raises(ValueError):
        reader_sampling_kwargs(**overrides)


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("temperature", [0.6, 0.0])
def test_actual_reader_generate_receives_sampling_and_preserves_image_path(tmp_path, with_image, temperature):
    from experiments.state_tokenizer.longmemeval_reader import LongMemEvalReader
    from residualmem.latent.instruct_bridge import MemorySegment

    class Tokenizer(_CompletionTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            return "rendered prompt"

        def __call__(self, text, **kwargs):
            ids = [77] if text == "<LME_MEMORY_SLOT>" else [10, 77, 11]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            assert messages[1]["content"][0] == {"type": "image"}
            return self.tokenizer.apply_chat_template(messages, **kwargs)

        def __call__(self, **kwargs):
            assert len(kwargs["images"]) == 1
            return {"input_ids": torch.tensor([[10, 60, 77, 11]]),
                    "attention_mask": torch.ones(1, 4),
                    "pixel_values": torch.ones(1, 3), "image_grid_thw": torch.ones(1, 3, dtype=torch.long)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(100, 4)
            self.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=30000))

        def get_input_embeddings(self):
            return self.embedding

        def generate(self, input_ids, **kwargs):
            self.received = kwargs
            assert input_ids.tolist() == ([[10, 60, 11]] if with_image else [[10, 11]])
            assert kwargs["inputs_embeds"].shape[1] == input_ids.shape[1] + 2
            assert kwargs["attention_mask"].shape[1] == kwargs["inputs_embeds"].shape[1]
            assert ("pixel_values" in kwargs) == with_image
            assert ("image_grid_thw" in kwargs) == with_image
            return torch.cat([input_ids, torch.tensor([[1, 99, 2]])], dim=1)

    model = Model()
    reader = LongMemEvalReader(model, Processor(), torch.nn.Identity(), mode="input")
    reader._encode_segments = lambda segments, device: ([torch.zeros(1, 2, 4)], [torch.ones(1, 2)], [])
    image_path = None
    if with_image:
        image_path = tmp_path / "question.png"
        Image.new("RGB", (8, 8)).save(image_path)
    answer = reader.answer("question", [MemorySegment(text="memory")],
                           question_image=image_path, temperature=temperature)
    assert answer == r"\boxed{right}"
    assert model.received["max_new_tokens"] == 20000
    assert model.received["do_sample"] is (temperature > 0)
    for key, value in {"temperature": 0.6, "top_p": 0.95, "top_k": 20}.items():
        if temperature > 0:
            assert model.received[key] == reader.last_generation[key] == value
        else:
            assert key not in model.received
    assert reader.last_generation["do_sample"] is (temperature > 0)
