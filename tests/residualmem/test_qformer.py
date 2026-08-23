"""The Q-Former state tokenizer's structural invariants.

The pooled tokenizer these replace has three defects that all trace to the same
root -- causal attention silently nullifying something -- and none of them was
caught by a test, because nothing in ``tests/`` touched that path either:

1. image slots cannot see text that comes after them, so they are immune to
   serializer changes;
2. the context band is valid in exactly 3 of its 16 slots on every WorldMemArena
   observation;
3. the observation prompt sits last in the sequence and owns zero slots, so its
   causal influence on the 64 states is exactly zero.

``test_the_prompt_region_changes_the_state`` is the direct regression for (3):
it is the one property the pooled pipeline provably does not have, and it is
what the four anchor losses in the first draft of this design were buying.

No torch dependency beyond the module itself; runs under ``tests/run_tests.py``.
"""
from __future__ import annotations

import torch

from residualmem.latent.qformer import StateQFormer, qformer_hash

IMAGE, DOM, PROMPT = 0, 1, 2


def _module(seed: int = 0, **kwargs) -> StateQFormer:
    torch.manual_seed(seed)
    defaults = dict(num_queries=64, output_dim=512, input_dim=32,
                    hidden=16, heads=2, layers=2)
    return StateQFormer(**{**defaults, **kwargs}).eval()


def _inputs(batch: int = 1, tokens: int = 25, width: int = 32, seed: int = 1):
    """A stand-in for H_t laid out as the real pipeline lays it out."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, tokens, width, generator=generator)
    modality = torch.full((batch, tokens), DOM, dtype=torch.long)
    modality[:, : tokens // 2] = IMAGE
    modality[:, -5:] = PROMPT           # the instruction span, last as in _prompt()
    positions = torch.linspace(0, 1, tokens).expand(batch, tokens).contiguous()
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    return hidden, modality, positions, mask


def test_output_keeps_the_pipeline_shape_contract():
    """64x512 is not a preference -- three downstream modules hard-require it."""
    module = _module()
    hidden, modality, positions, mask = _inputs(batch=3, tokens=40)
    output = module(hidden, modality, positions, mask)
    assert output.shape == (3, 64, 512)
    assert torch.isfinite(output).all()


def test_a_variable_length_sequence_still_yields_64_states():
    module = _module()
    for tokens in (8, 25, 137):
        output = module(*_inputs(tokens=tokens))
        assert output.shape == (1, 64, 512), tokens


def test_the_prompt_region_changes_the_state():
    """The property the pooled tokenizer does not have.

    In ``build_static_key64`` the prompt band is ``PROMPT_SLOTS = 0`` and the
    instruction sits after both the image and the DOM, so no pooled slot can
    carry it -- ``OBSERVATION_PROMPT`` asks the model to preserve visible text,
    input values, control types, focus state and spatial relations, and none of
    that request reaches a slot. Cross-attention is not causal, so here it does.
    """
    module = _module()
    hidden, modality, positions, mask = _inputs(tokens=25)
    other = hidden.clone()
    other[:, -5:] = torch.randn(1, 5, hidden.shape[-1], generator=torch.Generator().manual_seed(9))
    assert (modality[:, -5:] == PROMPT).all(), "the changed span must be the instruction"
    assert torch.equal(hidden[:, :-5], other[:, :-5]), "only the instruction may differ"

    base = module(hidden, modality, positions, mask)
    changed = module(other, modality, positions, mask)
    assert not torch.allclose(base, changed, atol=1e-5), (
        "the instruction span reached no state token -- cross-attention is not "
        "covering the whole sequence"
    )


def test_padding_does_not_reach_the_output():
    """A short observation batched with a long one must score identically."""
    module = _module()
    short_tokens = 12
    hidden, modality, positions, mask = _inputs(batch=1, tokens=short_tokens)
    alone = module(hidden, modality, positions, mask)

    pad = 8
    padded_hidden = torch.cat((hidden, torch.randn(1, pad, hidden.shape[-1])), dim=1)
    padded_modality = torch.cat((modality, torch.zeros(1, pad, dtype=torch.long)), dim=1)
    padded_positions = torch.cat((positions, torch.zeros(1, pad)), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(1, pad, dtype=torch.bool)), dim=1)
    batched = module(padded_hidden, padded_modality, padded_positions, padded_mask)

    assert torch.allclose(alone, batched, atol=1e-5), (
        "padded positions changed the state; the mask or the zeroing is wrong"
    )


def test_gradients_reach_the_queries():
    """The queries are the learned part; if they get no gradient nothing trains."""
    module = _module()
    output = module(*_inputs())
    output.square().mean().backward()
    for name, parameter in (("queries", module.queries),
                            ("input_projection", module.input_projection.weight),
                            ("output_projection", module.output_projection.weight)):
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
        assert parameter.grad.abs().sum() > 0, f"{name} gradient is exactly zero"


def test_the_module_is_deterministic_so_it_can_be_frozen():
    """Report 5.2 freezes ``P_rho`` after training; dropout would break that."""
    module = _module()
    arguments = _inputs()
    assert torch.equal(module(*arguments), module(*arguments))


def test_modality_embedding_starts_neutral():
    """Zero init keeps step-zero behaviour a function of H_t alone.

    Same discipline as ``InputSoftTokenConnector.rank_embedding``, which is
    zero-initialized so slot identity is learned only if the objective asks.
    """
    module = _module()
    assert torch.count_nonzero(module.modality_embedding.weight) == 0


def test_shape_mismatches_are_rejected_rather_than_broadcast():
    module = _module()
    hidden, modality, positions, mask = _inputs(tokens=25)
    for bad in ("modality", "positions", "mask"):
        arguments = {"modality": modality, "positions": positions, "mask": mask}
        arguments[bad] = arguments[bad][:, :-1]
        try:
            module(hidden, arguments["modality"], arguments["positions"], arguments["mask"])
        except ValueError:
            continue
        raise AssertionError(f"a truncated {bad} was accepted")


def test_an_all_padding_row_is_rejected():
    module = _module()
    hidden, modality, positions, mask = _inputs(batch=2, tokens=20)
    mask[1] = False
    try:
        module(hidden, modality, positions, mask)
    except ValueError:
        return
    raise AssertionError("an observation with no tokens produced a state")


def test_the_hash_tracks_the_weights():
    """Replaces ``pca_sha256`` in the provenance chain, so it must move."""
    module = _module()
    before = qformer_hash(module)
    assert before == qformer_hash(_module()), "the hash is not reproducible"
    with torch.no_grad():
        module.queries[0, 0] += 1.0
    assert qformer_hash(module) != before, "a weight change left the hash unmoved"
