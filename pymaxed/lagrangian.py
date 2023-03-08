#!/usr/bin/env python3
"""Lagrangian module to compute the Lagrange function, its Jacobian, and its Hessian."""
from math import sqrt

import torch
from torch import Tensor

from pymaxed.vectors import Vec


def quad_integral(
    coeffs: Tensor,
    vec: Vec,
    p1: Tensor,
    t1: int,
    p2: Tensor | None = None,
    t2: int | None = None,
) -> Tensor:
    r"""Perform the quadrature integration (using the Gauss-Legendre polynomial) of arbitrary function with maximum entropy
    distribution (MED) in N-dimension.

    Here, we are performing,

    $I(p_k(x)) = \int g(x) F_{MED}(p_k(x)) dx$

    where,

    $F_{MED}(x) = \exp{\sum_{i=0}^{K} \lambda_i x^i}$

    By using Gauss-Legendre quadrature method

    Args:
        coeffs (Tensor): coefficients used in MED
        vec (`Vec`): object contains relevant vector spaces.
        p1 (Tensor): base polynomial matrix
        t1 (int): target moment index
        p2 (Tensor, optional): if the polynomial is needed to be multiplied
            with other polynomials. Defaults to None.
        t2 (int, optional): if p2 exists, target moments index of p2.
            Defaults to None.

    Returns:
        float: result of integration.
    """

    # get relevant polynomials
    poly = vec.get_poly(coeffs, p1)
    poly_target = vec.get_poly_target(p1, t1, p2, t2)
    output = (vec.w * poly_target * torch.exp(poly)).sum()

    return output


def ortho_polynomial(gamma: Tensor, vec: Vec, a: Tensor) -> tuple[Tensor, Tensor, bool]:
    """Polynomial basis orthogonalization using Gram-Schmidt algorithm.
    Designed to be worked in multidimensional phase space (2D and 3D).

    Note:
        - According to Abramov's paper, with or without re-orthogonalization produces sufficiently orthogonal polynomials for the maximum entropy problems. Therefore, here we omits the re-orthogonalization step.

    Args:
        gamma (Tensor): Lagrangian multipliers.
        vec (object): vector space object.
        a (Tensor): polynomial basis.

    Returns:
        Tensor, Tensor, bool: orthogonalized basis, Lagrangian multiplier,
            and error flag.
    """

    a_init = a.clone()
    p = a.clone()
    gamma_init = gamma

    for k_mnts in range(vec.p_order):
        for m_mnts in range(k_mnts):
            a[:, k_mnts] -= (
                quad_integral(gamma, vec, a, k_mnts, p, m_mnts) * p[:, m_mnts]
            )
            # update gamma according to new a
            gamma = torch.linalg.inv(a) @ (a_init @ gamma_init)

        # setting new polynomial basis with new a
        q_ak_ak = quad_integral(gamma, vec, a, k_mnts, a, k_mnts)

        if q_ak_ak <= 0:
            return a_init, gamma_init, True
        else:
            p[:, k_mnts] = a[:, k_mnts] / sqrt(q_ak_ak)

    try:
        # new coefficient according to orthogonalized polynomials
        # update gamma according to new p
        gamma = torch.linalg.inv(p) @ (a_init @ gamma_init)
        return p, gamma, False
    except RuntimeError:
        return a_init, gamma_init, True


def l_func(coeffs: Tensor, vec: Vec, p: Tensor, mnts_cond: Tensor) -> Tensor:
    r"""Calculate the dual objective function with orthogonalized polynomial.
    Using the Lagrange multiplier, we can rewrite the dual objective function to unconstrained optimization problem.

    Here, we are calculating objective function

    .. math::
        \mathcal{L}(\vec{\gamma}) =\int
        \exp{\left(\sum_{k=1}^{K}\gamma_k p_k(\vec{x})\right)}d\vec{x} -
        \sum_{k=1}^{K}\gamma_k p_k(\vec{\mu})

    Args:
        coeffs (Tensor): Lagrange multipliers
        vec (Vec): vector space
        p (Tensor): orthogonalized polynomial
        mnts_cond (Tensor): alpha-scaled moment constrains

    Returns:
        float: Lagrange function
    """

    # Objective (lagrangian) function.
    return (vec.w * torch.exp(vec.get_poly(coeffs, p))).sum() - (
        coeffs * (mnts_cond @ p)
    ).sum()


def l_jac(coeffs: Tensor, vec: Vec, p: Tensor, mnts_cond: Tensor) -> Tensor:
    r"""Jacobian of the dual object function.

    .. math::

        \nabla \mathcal{L}(\vec{\gamma}) = \frac{\partial \mathcal{L}
        (\vec{\gamma})}
        {\partial \gamma_k} = \mathcal{Q}_{\vec{\gamma}}(p_k) - p_k(\vec{\mu})

    Args:
        coeffs (Tensor): Lagrangian multiplier
        vec (`Vec`): vector space object
        p (Tensor): polynomials
        mnts_cond (mnts): moments constrain

    Returns:
        Tensor: Jacobian vector
    """

    jac = torch.zeros(vec.p_order, dtype=vec.dtype.float)

    for k_mnts in range(vec.p_order):
        q_p_k = quad_integral(coeffs, vec, p, k_mnts)
        jac[k_mnts] = q_p_k - (mnts_cond @ p[:, k_mnts])

    return jac


def l_hess(coeffs: Tensor, vec: Vec, p: Tensor, _) -> Tensor:
    r"""Hessian matrix of the dual object function.
    Since we already performed orthogonalization, Hessian matrix can be
    simplified as follow:

    .. math::
        H_{km} = \frac{\partial^2 \mathcal{L}}
        {\partial \gamma_k \partial \gamma_m} =
        \mathcal{Q}_{\vec{\gamma}}(p_k p_m) = \delta_{km},

    where $\delta_{km}$ is the Kronecker delta.

    Args:
        coeffs (Tensor): Lagrange multiplier
        vec (Vec): vector space
        _: dummy. Put this here to have same structure with Jac and Lagrangian

    Returns:
        Tensor: Hessian matrix
    """

    hess = torch.zeros((vec.p_order, vec.p_order), dtype=vec.dtype.float)

    for k_mnts in range(vec.p_order):
        for m_mnts in range(vec.p_order):
            hess[k_mnts, m_mnts] = quad_integral(coeffs, vec, p, k_mnts, p, m_mnts)

    return hess
