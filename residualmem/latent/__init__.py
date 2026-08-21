"""Latent-side APIs with optional JAX/Torch dependencies loaded on demand."""
from __future__ import annotations

__all__ = [
    "DOMAINS",
    "DomainId",
    "LatentSpec",
    "StateTokens",
    "latent_state",
    "make_latent_schema",
    "MAGIC",
    "FORMAT_VERSION",
    "LEGACY_MAGICS",
    "PerceiverConfig",
    "initialize_perceiver",
    "resample",
    "BRIDGE_CACHE_PROTOCOL",
    "FUSED_OBSERVATION_TEACHER_PROTOCOL",
    "RETRIEVAL_BRIDGE_PROTOCOL",
    "READER_BRIDGE_PROTOCOL",
    "BROWSERGYM_TEACHER_TEXT_PROTOCOL",
    "MaskedAttentionRetrievalHead",
    "Qwen35LatentReader",
    "InputSoftTokenConnector",
    "Layer16Restorer",
    "Layer16Connector",
    "TorchA2Reconstructor",
]

_MODULE_BY_NAME = {
    "MAGIC": "format", "FORMAT_VERSION": "format", "LEGACY_MAGICS": "format",
    "PerceiverConfig": "tokenizer", "initialize_perceiver": "tokenizer", "resample": "tokenizer",
    "DOMAINS": "types", "DomainId": "types", "LatentSpec": "types",
    "StateTokens": "types", "latent_state": "types", "make_latent_schema": "types",
    "BRIDGE_CACHE_PROTOCOL": "instruct_bridge",
    "FUSED_OBSERVATION_TEACHER_PROTOCOL": "instruct_bridge",
    "RETRIEVAL_BRIDGE_PROTOCOL": "instruct_bridge",
    "READER_BRIDGE_PROTOCOL": "instruct_bridge",
    "BROWSERGYM_TEACHER_TEXT_PROTOCOL": "instruct_bridge",
    "MaskedAttentionRetrievalHead": "instruct_bridge",
    "Qwen35LatentReader": "instruct_bridge",
    "InputSoftTokenConnector": "instruct_bridge",
    "Layer16Restorer": "instruct_bridge",
    "Layer16Connector": "instruct_bridge",
    "TorchA2Reconstructor": "instruct_bridge",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    from importlib import import_module
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
