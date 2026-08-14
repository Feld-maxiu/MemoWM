"""Diagnostic-only experiment line for the v8 discrete world model.

Nothing in this package may write under ``outputs/world_model/v8/``. Every
artifact is stamped ``formal: false`` and every entry point is hard-wired to the
validation split, so no diagnostic can reach the locked test split or be
absorbed into the preregistered statistics.
"""

DEV_PROTOCOL = "v8_wm_dev_diagnostic_v1"
