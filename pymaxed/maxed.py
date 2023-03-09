#!/usr/bin/env python3
"""Maximum entropy distribution (MaxEd) module."""
import warnings
from dataclasses import dataclass
from math import log

import torch
from torch import Tensor

from pymaxed.lagrangian import l_func
from pymaxed.lagrangian import l_hess
from pymaxed.lagrangian import l_jac
from pymaxed.lagrangian import ortho_polynomial
from pymaxed.lagrangian import quad_integral
from pymaxed.minimize import FunctionTools
from pymaxed.minimize import minimize_bfgs
from pymaxed.vectors import Vec


@dataclass
class Maxed:
    """Maximum entropy distribution (MaxEd) class. Primary goal is to obtain the distribution from the moment constraints.

    Example:
        >>> target = [...]
        >>> mesh = Mesh(...)
        >>> vec = Vec(mesh, target, n_mnts, [...])
        >>> med = Maxed(vec)
        >>> med.solve()

    """

    vec: Vec
    """Vector spaces to reconstruct the distribution function."""
    lr: float = 1.0
    """Learning rate for the optimization process."""
    max_itr: int | None = None
    """Maximum number of iterations. If None, then the number of iterations is determined by the number of moments times 200."""
    gtol: float = 1e-5
    """Tolerance for the gradient norm."""
    xtol: float = 1e-9
    """Tolerance for the function/parameter changes."""
    disp: bool | int = False
    """Level of verbosity."""

    def solve(self) -> None:
        """Convert moments to the Maxed."""

        # Prepare initial conditions
        p_order = self.vec.p_order
        mnts_scaled = self.vec.mnts_scaled

        multiplier = self.vec.init

        p, gamma, err = ortho_polynomial(multiplier, self.vec, self.vec.a)

        if err:
            warnings.warn(
                "MaxEd: Orthogonalization failed. However, we will try to continue."
            )

        # Construct function tools
        functools = FunctionTools(
            l_jac, l_hess, ortho_polynomial, (self.vec, p, mnts_scaled)
        )

        # Optimization step
        res = minimize_bfgs(
            l_func,
            gamma,
            functools,
            self.lr,
            self.max_itr,
            self.gtol,
            self.xtol,
            self.disp,
        )

        if res["success"]:
            p = res["p"]
            coeffs = (p @ res["x"]).detach()
            coeffs[0] += log(self.vec.alpha)

            for k_mnts in range(p_order):
                coeffs[k_mnts] *= self.vec.alpha ** self.vec.mono[k_mnts, :].sum()

            # Density correction
            den = quad_integral(coeffs, self.vec, self.vec.a, 0)
            # Density correction for the distribution
            coeffs[0] = torch.log(torch.exp(coeffs[0]) / den)

            # Final coefficients
            self.coeffs = coeffs.detach()

            # The Maximum entropy distribution
            self.dist = self.dist_from_coeffs(self.coeffs)

            # Computed moments from the approximated distribution
            mnts_computed = torch.zeros_like(self.vec.mnts)
            # Calculate moments based on obtained MED
            for i_mnts in range(self.vec.p_order):
                mnts_computed[i_mnts] = quad_integral(
                    self.coeffs, self.vec, self.vec.a, i_mnts
                )
            self.mnts_computed = mnts_computed.detach()
            self.success = True
        else:
            self.coeffs = None
            self.dist = None
            self.mnts_computed = None
            self.success = False

    def dist_from_coeffs(self, coeffs: Tensor | None = None) -> Tensor:
        if coeffs is None:
            assert self.coeffs is not None, "Maxed: self.coeffs should exist."
            maxed = torch.exp(self.vec.get_poly(self.coeffs, self.vec.a, origin=True))
        else:
            maxed = torch.exp(self.vec.get_poly(coeffs, self.vec.a, origin=True))

        if torch.isnan(maxed).any() or torch.isinf(maxed).any():
            raise ValueError(
                "Maxed: invalid value detected during the construction of the PDF."
            )

        self.dist = maxed.detach()

        return self.dist
