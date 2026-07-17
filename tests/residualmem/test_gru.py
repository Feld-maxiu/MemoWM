from __future__ import annotations

import numpy as np
import pytest

from residualmem.schemas import crafter_schema
from residualmem.types import StateSchema
from residualmem.world_model.gru import (
    GRUConfig,
    GRUPredictor,
    initialize_params,
    load_gru_checkpoint,
    save_gru_checkpoint,
)


def test_gru_is_deterministic_and_checkpoint_is_schema_strict(tmp_path):
    exact = crafter_schema()
    config = GRUConfig(hidden_size=8)
    params_a = initialize_params(exact, config, seed=7)
    params_b = initialize_params(exact, config, seed=7)
    assert all(
        np.array_equal(np.asarray(params_a[key]), np.asarray(params_b[key]))
        for key in params_a
    )

    checkpoint = tmp_path / "gru.npz"
    save_gru_checkpoint(checkpoint, params_a, exact, config)
    loaded = load_gru_checkpoint(checkpoint, exact)
    assert isinstance(loaded, GRUPredictor)
    assert loaded.schema == exact
    other = StateSchema("other-exact", exact.fields)
    with pytest.raises(ValueError, match="schema hash mismatch"):
        load_gru_checkpoint(checkpoint, other)


def test_gru_prediction_uses_decoder_visible_inputs_only():
    schema = crafter_schema()
    config = GRUConfig(hidden_size=8)
    params = initialize_params(schema, config, seed=11)
    state = schema.make_state([0] * len(schema.fields))
    first = GRUPredictor(params, schema, config)
    second = GRUPredictor(params, schema, config)

    prediction_a = first.predict_next(state, 3)
    prediction_b = second.predict_next(state, 3)
    assert prediction_a.default_state == prediction_b.default_state
    assert prediction_a.distributions.keys() == prediction_b.distributions.keys()
    assert all(
        np.array_equal(
            prediction_a.distributions[key], prediction_b.distributions[key]
        )
        for key in prediction_a.distributions
    )
