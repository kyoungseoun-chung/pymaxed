#!/usr/bin/env python3
from pyapes.backend import DType

from pymaxed.tools import get_mono


def test_get_mono() -> None:
    dtype = DType("double")

    order, monomial = get_mono(3, 2, dtype)
    assert order == 10
    assert monomial.tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [0, 2],
        [1, 1],
        [2, 0],
        [0, 3],
        [1, 2],
        [2, 1],
        [3, 0],
    ]
