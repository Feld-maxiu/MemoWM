"""LatentPredictor: adapts the RSSM prior to the ``Predictor`` interface (seam 2).

The existing codec advances by calling ``predict_next(reconstructed, action)`` and
comparing the realized state to the predicted default. By exposing the RSSM prior
``p_theta(z_t | h_t)`` through this interface, the grouped categorical latent plugs
straight into the v0.3 exact codec (giving "latent-exact" for free) and, at Stage 2,
into the conditional entropy codec.

Carry is just the deterministic state ``h``; the previous latent ``z_{t-1}`` arrives
as the ``reconstructed`` argument each call, so encoder and decoder stay in lockstep
(and unsent codes never bypass the transition -- report Section 7.4 invariant).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..latent.types import DomainId
from ..types import CanonicalState, Prediction, Predictor, StateSchema
from . import rssm as R


class LatentPredictor(Predictor):
    def __init__(self, params, config: R.RSSMConfig, latent_schema: StateSchema,
                 domain: DomainId):
        if len(latent_schema.fields) != config.num_groups:
            raise ValueError("latent schema group count != config.num_groups")
        self.params = {key: jnp.asarray(value) for key, value in params.items()}
        self.config = config
        self.schema = latent_schema
        self.domain = domain
        self.predictor_id = "rssm-latent-v04"
        domain_idx = domain.index

        def _step(params, h, codes_prev, action):
            zemb_prev = R.embed_codes(params, codes_prev, config)
            h_new = R.transition(params, h, zemb_prev, action, 1.0, domain_idx, config)
            return h_new, R.prior_logits(params, h_new, config)

        self._step = jax.jit(_step)
        self._hash = R._content_hash(
            self.params, config, latent_schema.hash_bytes, domain.name)
        self.reset()

    def reset(self) -> None:
        self.h = jnp.zeros((self.config.hidden_size,), jnp.float32)

    def new_session(self) -> Predictor:
        return LatentPredictor(self.params, self.config, self.schema, self.domain)

    def predict_next(self, reconstructed: CanonicalState, action: int, dt: int = 1
                     ) -> Prediction:
        codes_prev = jnp.asarray(reconstructed.values, jnp.int32)
        self.h, logits = self._step(self.params, self.h, codes_prev, jnp.asarray(action))
        logits = np.asarray(logits, np.float32)          # (N, C)
        defaults = [int(logits[index].argmax()) for index in range(self.config.num_groups)]
        distributions = {self.schema.fields[index].name: logits[index]
                         for index in range(self.config.num_groups)}
        return Prediction(self.schema.make_state(defaults), distributions)

    @property
    def hash_bytes(self) -> bytes:
        return self._hash
