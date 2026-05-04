"""Core package for reward-constrained visual memory selection."""


def run_pipeline(*args, **kwargs):
    """Lazy import to avoid importing heavy optional deps at package import time."""
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


__all__ = ["run_pipeline"]
