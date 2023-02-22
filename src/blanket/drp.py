from typing import Any, Callable

__all__ = ("get_drp_coverage",)


def get_drp_coverage(xs, thetas, sample_ref, metric: Callable[[Any, Any], Any]):
    """
    Estimates coverage with the distance to random point method.

    Reference:
        `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    Args:
        xs: ...

    Returns:
        ...
    """
    print("hello, world")
