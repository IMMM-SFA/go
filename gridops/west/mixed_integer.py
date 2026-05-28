"""
Placeholder for the U.S. Western Interconnection
mixed-integer programming (unit commitment) model.

Status: **Not yet implemented**.
"""

from __future__ import annotations


def build_west_mip(*args, **kwargs):
    """Raise :class:`NotImplementedError`; MIP model is not available."""
    raise NotImplementedError(
        "The U.S. Western Interconnection unit commitment MIP model has not been "
        "implemented yet. Use problem='linear' for economic dispatch."
    )