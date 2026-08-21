from types import SimpleNamespace

from experiments.state_tokenizer.key_pooling import parse_dom_spans
from residualmem.benchmarks.worldmemarena_tokenizer import (
    fused_observation_text,
    genuine_user_text,
    observation_from_turn,
    synthetic_axtree,
)


def test_loader_inlined_caption_is_not_duplicated():
    caption = "A browser showing a blue settings page."
    turn = SimpleNamespace(
        role="user",
        text=f"real user text\n\nimage:\nimage_id: img_1\nimage_caption: {caption}",
        attachments=(SimpleNamespace(
            caption=caption, file_path="/tmp/screenshot.png", image_id="img_1"
        ),),
    )
    observation = observation_from_turn(turn)
    assert observation.user_text == "real user text"
    assert observation.captions == (caption,)
    fused = fused_observation_text(turn.text, observation.captions)
    assert fused.count(caption) == 1
    assert "image_id" not in fused


def test_assistant_turn_is_rejected():
    turn = SimpleNamespace(role="assistant", text="I can see... Action: {}", attachments=())
    try:
        observation_from_turn(turn)
    except ValueError as error:
        assert "assistant" in str(error)
    else:
        raise AssertionError("assistant output entered the tokenizer observation")


def test_synthetic_tree_is_valid_compact_axtree():
    turn = SimpleNamespace(
        role="user",
        text="search for papers",
        attachments=(SimpleNamespace(
            caption='A page with "quoted" text', file_path=None, image_id="img"
        ),),
    )
    tree = synthetic_axtree(observation_from_turn(turn))
    nodes = parse_dom_spans(tree)
    assert len(nodes) == 2
    assert all(node.tag == "textarea" for node in nodes)
    assert "search for papers" in tree


def test_genuine_user_text_keeps_plain_text():
    assert genuine_user_text("hello") == "hello"

