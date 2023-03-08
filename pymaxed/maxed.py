#!/usr/bin/env python3
"""Maximum entropy distribution (MaxEd) module."""
import warnings
from dataclasses import dataclass

from torch import Tensor

from pymaxed.lagrangian import l_func
from pymaxed.lagrangian import ortho_polynomial
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

    def solve(self) -> Tensor:
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

        res = minimize_bfgs(
            l_func,
            gamma,
            (self.vec, p, mnts_scaled),
            self.lr,
            self.max_itr,
            self.gtol,
            self.xtol,
            self.disp,
        )

        ...
