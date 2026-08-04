"""Reserved on-disk constants for the ResidualMem v0.4 latent stream.

The v0.3 exact stream keeps its own ``RSMEMV03`` magic (``codec/segment.py``);
the latent codec (Stage 2) writes a distinct ``RSMEMV04`` container so a decoder
never confuses the two paradigms. Defined here so both the codec and tests share
one source of truth.
"""
from __future__ import annotations

MAGIC = b"RSMEMV04"
FORMAT_VERSION = 4

# Rejected explicitly by the latent decoder with a LegacyFormatError.
LEGACY_MAGICS: tuple[bytes, ...] = (b"RSMEMV02", b"RSMEMV03")
