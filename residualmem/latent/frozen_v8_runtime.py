"""Runtime for the frozen v8/v9 state tokenizer.

The historical filename is retained because experiment scripts imported it.
``FrozenV9InstructTokenizer`` is the v9 entry point and selects the Instruct
chat template explicitly; v8 artifacts remain untouched.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from residualmem.benchmarks.worldmemarena_tokenizer import (
    WebObservation,
    synthetic_axtree,
    validate_observation,
)
from residualmem.encoders.normalization import GroupChannelNormalizer


@dataclasses.dataclass(frozen=True)
class FrozenTokenizerOutput:
    xbar: np.ndarray
    valid: np.ndarray
    a2_xbar: np.ndarray | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class FrozenV9InstructTokenizer:
    """Screenshot/text/caption -> frozen xbar (and optional A2 reconstruction)."""

    protocol = "qwen35_instruct_static_key64_v1"

    def __init__(
        self,
        *,
        model_path: str | Path,
        pca_path: str | Path,
        normalization_path: str | Path,
        a2_checkpoint: str | Path | None = None,
        device: str = "cuda:0",
        layer: int = 16,
        max_length: int = 8192,
        use_kernels: bool = True,
    ) -> None:
        from experiments.state_tokenizer.extract_qwen import _load_model

        self.device = torch.device(device)
        self.layer = int(layer)
        self.max_length = int(max_length)
        self.processor, self.model = _load_model(str(model_path), self.device, use_kernels)
        with np.load(pca_path, allow_pickle=False) as artifact:
            self.pca_mean = torch.as_tensor(
                np.asarray(artifact["mean"], np.float32), device=self.device
            )
            self.components = torch.as_tensor(
                np.asarray(artifact["components"], np.float32), device=self.device
            )
        if self.pca_mean.shape != (4096,) or self.components.shape != (4096, 512):
            raise ValueError("v9 PCA must contain mean (4096) and components (4096,512)")
        self.normalizer = GroupChannelNormalizer.from_npz(normalization_path)
        self._a2 = self._load_a2(a2_checkpoint) if a2_checkpoint else None

    @staticmethod
    def _load_a2(path: str | Path):
        from residualmem.latent.instruct_bridge import TorchA2Reconstructor
        return TorchA2Reconstructor.from_checkpoint(path)

    @staticmethod
    def _blank_image() -> Image.Image:
        return Image.new("RGB", (1280, 720), color=(255, 255, 255))

    def encode(self, observation: WebObservation) -> FrozenTokenizerOutput:
        from experiments.state_tokenizer.extract_qwen import (
            _StopAtLayer,
            _input_text,
            aligned_dom_token_offsets,
            modality_indices,
            prepare_inputs,
        )
        from experiments.state_tokenizer.fixed_prompt import OBSERVATION_PROMPT
        from experiments.state_tokenizer.static_key_pooling import build_static_key64

        validate_observation(observation)
        dom = synthetic_axtree(observation)
        if observation.screenshot:
            with Image.open(observation.screenshot) as handle:
                image = handle.convert("RGB")
        else:
            image = self._blank_image()
        inputs, truncated, kept_dom_chars, original_length = prepare_inputs(
            self.processor,
            image,
            dom,
            OBSERVATION_PROMPT,
            self.max_length,
            "instruct",
        )
        if truncated:
            raise ValueError(
                f"WorldMemArena observation exceeded max length: kept {kept_dom_chars}/{len(dom)} chars"
            )
        indices = modality_indices(self.processor, self.model, inputs["input_ids"])
        text = _input_text(self.processor, dom, OBSERVATION_PROMPT, "instruct")
        offsets = aligned_dom_token_offsets(
            self.processor, self.model, text, dom, inputs["input_ids"], indices[1]
        )
        image_grid = tuple(int(x) for x in inputs["image_grid_thw"][0].tolist())
        device_inputs = inputs.to(self.device)
        device_indices = tuple(index.to(self.device) for index in indices)
        layers = self.model.model.language_model.layers
        if not 1 <= self.layer <= len(layers):
            raise ValueError(f"layer {self.layer} outside model depth {len(layers)}")
        capture: dict[str, torch.Tensor] = {}

        def hook(_module, _args, value):
            capture["hidden"] = value[0] if isinstance(value, tuple) else value
            raise _StopAtLayer

        handle = layers[self.layer - 1].register_forward_hook(hook)
        try:
            try:
                with torch.inference_mode():
                    self.model.model(**device_inputs, use_cache=False, output_hidden_states=False)
            except _StopAtLayer:
                pass
        finally:
            handle.remove()
        hidden = capture.get("hidden")
        if hidden is None:
            raise RuntimeError("Qwen layer hook did not fire")
        image_hidden, dom_hidden, prompt_hidden = [
            hidden[0].index_select(0, index) for index in device_indices
        ]
        pooled = build_static_key64(
            image_hidden,
            dom_hidden,
            prompt_hidden,
            dom=dom,
            dom_token_offsets=offsets,
            image_grid_thw=image_grid,
            spatial_merge_size=int(self.model.config.vision_config.spatial_merge_size),
            instruction=observation.user_text or None,
        )
        tokens = pooled.tokens
        valid_t = pooled.valid
        pca = torch.zeros((64, 512), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            pca[valid_t] = (
                tokens[valid_t].float() - self.pca_mean
            ) @ self.components
        pca_np = pca.cpu().numpy()
        valid = valid_t.cpu().numpy()
        xbar = self.normalizer.normalize(pca_np, valid)
        a2_xbar = None
        if self._a2 is not None:
            self._a2.to(self.device)
            with torch.inference_mode():
                a2_xbar = self._a2(
                    torch.as_tensor(xbar[None], device=self.device),
                    torch.as_tensor(valid[None], device=self.device),
                )[0].cpu().numpy().astype(np.float32)
        return FrozenTokenizerOutput(
            xbar=xbar,
            valid=valid,
            a2_xbar=a2_xbar,
            metadata={
                "protocol": self.protocol,
                "original_sequence_length": original_length,
                "image_grid_thw": image_grid,
                "synthetic_axtree": dom,
            },
        )


# Compatibility name used by the first zero-training WorldMemArena adapter.
FrozenV8Tokenizer = FrozenV9InstructTokenizer
