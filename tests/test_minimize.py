#!/usr/bin/env python3
import torch
from pyapes.core.geometry import Box
from pyapes.core.geometry import Cylinder
from pyapes.core.mesh import Mesh
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pymaxed.maxed import Maxed
from pymaxed.minimize import minimize_bfgs
from pymaxed.vectors import Vec


N = 100
D = 7
M = 5
X = torch.randn(N, D)
Y = torch.randn(N, M)
trueB = torch.linalg.inv(X.T @ X) @ X.T @ Y


def objective_func(B: Tensor) -> Tensor:
    return torch.sum((Y - X @ B) ** 2)


def test_minimize() -> None:
    B0 = torch.zeros(D, M)
    res = minimize_bfgs(objective_func, B0)

    assert res["success"] == True
    assert_close(res["x"], trueB, atol=1e-4, rtol=1e-4)


def test_maxed_minimize() -> None:
    target = [1, 0, 1, -0.27, 1.7178]
    mesh = Mesh(Box[-5:5], None, [100])

    vec = Vec(mesh, target, 4, [100])

    maxed = Maxed(vec)

    maxed.solve()
    # Results from my old code
    coeffs_old = torch.tensor(
        [-1.60707008, 0.70435461, 1.21316491, -0.43266241, -0.54965037],
        dtype=vec.dtype.float,
        device=vec.device,
    )

    assert_close(maxed.coeffs, coeffs_old, atol=1e-4, rtol=1e-4)
    assert_close(vec.mnts, maxed.mnts_computed, atol=1e-4, rtol=1e-4)

    target = [1, 0, 1, -0.27, 1.7178, 2, 8, 0, 2]
    mesh = Mesh(Cylinder[0:10, -10:10], None, [128, 256])

    vec = Vec(mesh, target, 4, [50, 100])
    maxed = Maxed(vec)

    maxed.solve()
    assert_close(vec.mnts, maxed.mnts_computed, atol=1e-2, rtol=1e-2)
