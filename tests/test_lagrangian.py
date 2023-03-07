#!/usr/bin/env python3
"""Test lagrangian related functions."""
from math import sqrt

import numpy as np
import pytest
import torch
from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from torch.testing import assert_close  # type: ignore

from pymaxed.lagrangian import l_func
from pymaxed.lagrangian import l_hess
from pymaxed.lagrangian import l_jac
from pymaxed.lagrangian import med_integral
from pymaxed.lagrangian import med_ortho_polynomial
from pymaxed.vectors import Vec


def test_auto_grad_prob() -> None:
    n_mnts = 4

    target = [1, 0, 1, -0.27, 1.7178]
    mesh = Mesh(Box[-10:10], None, [100])

    vec = Vec(mesh, target, n_mnts, [100])
    pk, gamma, _ = med_ortho_polynomial(vec.init, vec, vec.a)

    p_old = torch.tensor(
        [
            [
                6.31618785e-01,
                1.86092062e-18,
                -4.46624241e-01,
                -3.74645191e-18,
                3.86900019e-01,
            ],
            [
                0.00000000e00,
                6.31619017e-01,
                7.17212355e-18,
                -7.73610945e-01,
                -1.08793869e-17,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                4.46624568e-01,
                1.80027189e-18,
                -7.73838075e-01,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                2.57872286e-01,
                4.12526922e-18,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                1.28980243e-01,
            ],
        ],
        dtype=mesh.dtype.float,
        device=mesh.device,
    )
    gamma_old = torch.tensor(
        [-7.91616153e-01, 1.27121778e-17, -1.11950850e00, 0.00000000e00, 0.00000000e00],
        dtype=mesh.dtype.float,
        device=mesh.device,
    )

    assert_close(pk, p_old)
    assert_close(gamma, gamma_old)
    with torch.autograd.set_detect_anomaly(True):
        lag = l_func(gamma, vec, pk, vec.mnts)
        pass

    pass


def int_med(w, poly, poly_target) -> float:
    """Legacy code"""

    # suppress overflow warning!
    return (w * poly_target * np.exp(poly)).sum()


def test_lagrangian() -> None:
    # 1D case
    mesh = Mesh(Box[-5:5, -5:5], None, [50, 50])

    target = torch.nn.Parameter(
        torch.rand(15, dtype=mesh.dtype.float, requires_grad=True)
    )
    vec = Vec(mesh, target, 4, [100, 100])

    coeff_init = torch.rand(vec.p_order, dtype=mesh.dtype.float, requires_grad=True)

    for p in range(vec.p_order):
        target = int_med(
            vec.w,
            vec.get_poly(coeff_init.detach(), vec.a),
            vec.get_poly_target(vec.a, p),
        )
        test = med_integral(coeff_init, vec, vec.a, p)
        assert target == pytest.approx(test.detach())

    poly = ((vec.a @ coeff_init).repeat(*vec.p[0].shape, 1).T * vec.p).sum(dim=0)

    med = (vec.w * torch.exp(poly)).sum()
    pk, gamma, err = med_ortho_polynomial(coeff_init, vec, vec.a, False)
    lag = l_func(gamma, vec, pk, vec.mnts)
    jac = l_jac(gamma, vec, pk, vec.mnts)
    hes = l_hess(gamma, vec, pk, vec.mnts)
