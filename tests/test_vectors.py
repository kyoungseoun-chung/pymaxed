#!/usr/bin/env python3
"""Test vector space related functions."""
import numpy as np
import pytest
import torch
from pyapes.core.backend import DType
from pyapes.core.geometry import Cylinder
from pyapes.core.mesh import Mesh
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pymaxed.tools import get_mono
from pymaxed.tools import gl_setup
from pymaxed.vectors import _poly_point
from pymaxed.vectors import _poly_target_point
from pymaxed.vectors import Vec


DTYPE = DType("double")


def test_vector() -> None:
    # Target to be tested
    target = [1, 0, 1, -0.27, 1.7178, 2, 8, 0, 2]

    # Test on 1D space
    mesh = Mesh(Cylinder[0:5, -5:5], None, [50, 100])
    mnts_order = 4

    vec = Vec(mesh, target, mnts_order, [25, 100])

    assert vec.dim == 2
    assert vec.coord == "rz"
    assert vec.mnts_order == mnts_order
    assert vec.p_order == 9

    scale_np = 1.0 / (
        np.prod(np.arange(1, 2 * mnts_order, 2)) ** (1.0 / (2 * mnts_order))
    )

    assert vec.alpha == pytest.approx(scale_np)

    mono = vec.mono
    mono_r = mono.repeat(vec.p_order, *[1 for _ in range(vec.dim)])

    basis_t = torch.zeros(tuple([vec.p_order] + list(vec.x[0].shape)))

    for k in range(vec.p_order):
        basis_t[k, :] = vec.x[0] ** mono[k, 0] * vec.x[1] ** mono[k, 1]

    assert_close(basis_t, vec.p)

    pass


def old_poly_point(coeffs: Tensor, basis: Tensor, p_k: Tensor) -> Tensor:
    """Get polynomial value at the grid."""
    p_order = p_k.shape[0]

    poly = torch.zeros(basis[0].shape)

    # Need to be careful here!
    for p in range(p_order):
        for k in range(p_order):
            poly += p_k[p, k] * coeffs[k] * basis[p, :]

    return poly


def old_poly_target_point(
    basis: Tensor, p1: Tensor, t1: int, p2: Tensor | None = None, t2: int | None = None
) -> Tensor:
    """Get target poly on the grid."""

    p_order = p1.shape[0]
    poly_target = torch.zeros(basis[0].shape)

    if p2 is None:
        for k in range(p_order):
            poly_target += p1[k, t1] * basis[k, :]

    else:
        # very crude implementation.
        for k in range(p_order):
            for l in range(p_order):
                poly_target += p1[k, t1] * basis[k, :] * p2[l, t2] * basis[l, :]

    return poly_target


def test_poly_construction() -> None:
    coeffs = torch.rand(5)
    p_k = torch.rand(5, 5)
    basis = torch.rand(5, 7, 14)

    old_ver = old_poly_point(coeffs, basis, p_k)
    new_ver = _poly_point(coeffs, basis, p_k)

    assert_close(old_ver, new_ver)

    k, _ = get_mono(15, 1, DType("double"))
    p1 = torch.rand(k, k)
    p2 = torch.rand(k, k)
    basis = torch.rand(k, 50, 100)

    t1 = int(torch.randint(0, k - 1, (1,)).item())
    t2 = int(torch.randint(0, k - 1, (1,)).item())

    old_ver = old_poly_target_point(basis, p1, t1)
    new_ver = _poly_target_point(basis, p1, t1)

    assert_close(old_ver, new_ver)

    import time

    tic = time.perf_counter()
    old_ver = old_poly_target_point(basis, p1, t1, p2, t2)
    sw1 = time.perf_counter() - tic

    tic = time.perf_counter()
    new_ver = _poly_target_point(basis, p1, t1, p2, t2)
    sw2 = time.perf_counter() - tic
    assert_close(old_ver, new_ver)

    gain = sw1 / sw2
    print(f"performance gain: {gain:.2f}x")
