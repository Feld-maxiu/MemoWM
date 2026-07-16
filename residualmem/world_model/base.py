from __future__ import annotations

import hashlib

from ..types import CanonicalState, Prediction, Predictor


class PersistencePredictor(Predictor):
    predictor_id = "persistence-v1"

    def reset(self) -> None:
        pass

    def predict_next(
        self, reconstructed: CanonicalState, action: int, dt: int = 1
    ) -> Prediction:
        del action, dt
        return Prediction(reconstructed, {})


class IgnoreActionPredictor(Predictor):
    """Ablation wrapper that replaces every action with a fixed no-op action."""

    def __init__(self, predictor: Predictor, noop_action: int = 0):
        self.predictor = predictor
        self.noop_action = int(noop_action)
        self.predictor_id = f"ignore-action({predictor.predictor_id})"

    @property
    def hash_bytes(self) -> bytes:
        payload = (
            b"ignore-action\0"
            + self.predictor.hash_bytes
            + self.noop_action.to_bytes(4, "little", signed=True)
        )
        return hashlib.sha256(payload).digest()

    def reset(self) -> None:
        self.predictor.reset()

    def predict_next(
        self, reconstructed: CanonicalState, action: int, dt: int = 1
    ) -> Prediction:
        del action
        return self.predictor.predict_next(reconstructed, self.noop_action, dt)
