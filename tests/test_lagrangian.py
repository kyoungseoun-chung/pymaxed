#!/usr/bin/env python3
"""Test lagrangian related functions."""
import torch
from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from torch import Tensor

from pymaxed.lagrangian import l_func
from pymaxed.lagrangian import l_hess
from pymaxed.lagrangian import l_jac
from pymaxed.lagrangian import med_integral
from pymaxed.lagrangian import med_ortho_polynomial
from pymaxed.vectors import Vec


def test_lagrangian() -> None:
    # 1D case
    mesh = Mesh(Box[-5:5], None, [50, 100])

    target = [1, 0, 1, -0.27, 1.7178]

    vec = Vec(mesh, target, 4, [100])

    coeff_init = torch.tensor([0, 0, -0.5, 0, 0], dtype=mesh.dtype.float)
    pk, gamma, err = med_ortho_polynomial(coeff_init, vec, vec.a, False)
    pass
