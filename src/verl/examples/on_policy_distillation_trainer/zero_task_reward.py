"""Task reward used when MOPD trains from distillation loss only."""


def compute_score(*args, **kwargs) -> float:
    """Keep every routed sample while distillation supplies the learning signal."""
    return 0.0
