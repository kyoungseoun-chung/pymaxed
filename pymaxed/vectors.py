#!/usr/bin/env python3
"""Vector space for MaxEnt module."""
from dataclasses import dataclass
from functools import cached_property
from math import pi

import torch
from pyapes.backend import DType
from pyapes.mesh import Mesh
from torch import Tensor

from pymaxed.tools import get_mono
from pymaxed.tools import gl_setup

COORDINATE = ["xyz", "rz"]
"""Supporting coordinate system. Cartesian (`xyz`) and axisymmetric (`rz`)."""


@dataclass
class Vec:
    """Construct vector space for the MaxEnt approximation.
    This `Vector` object contains two different sets of grid information.

    - Equidistance grid (original space) defined by `self.mesh` object.
    - Grid `self.vec` defined by the Gauss-Legendre polynomial.
        - This is the space actually used to solve the backward problem.

    Once the optimization process is done, you could have approximated PDFs either on `self.mesh` or `self.vec`.

    Args:
        mesh (Mesh): `Mesh` object (original space)
        target (Tensor | list[float]): target moment set.
    """

    mesh: Mesh
    """Original computational domain."""
    target: Tensor | list[float]
    """Target moment set."""
    mnts_order: int
    """Order of moments to be approximated."""
    n_gl_nodes: tuple[int, ...] | list[int]
    """Number of Gauss-Legendre polynomial nodes."""

    def __post_init__(self):
        assert self.coord in ["xyz", "rz"], f"Vector: {self.coord=} is not supported!"
        assert (
            len(self.n_gl_nodes) == self.dim
        ), f"Vector: {self.n_gl_nodes=} is not compatible with {self.dim=}!"

        # Make sure the target is a tensor and has a consistent dtype and device with mesh
        if isinstance(self.target, list):
            self._mnts = torch.tensor(
                self.target, dtype=self.dtype.float, device=self.device
            )
        else:
            self._mnts = self.target.to(self.device, self.dtype.float)

        # Setup computational vector space
        self.setup_space()

    @property
    def mnts(self) -> Tensor:
        """Target moment set."""
        return self._mnts

    @cached_property
    def mnts_scaled(self) -> Tensor:
        """Scaled target moment set."""
        scaled = torch.zeros(self.p_order, dtype=self.dtype.float, device=self.device)

        for i in range(self.p_order):
            scaled[i] = self.mnts[i] * self.alpha ** (self.mono[i, :].sum().item())

        return scaled

    @property
    def dim(self) -> int:
        """Dimension of the space. If `self.coord == "rz"`, it is 2."""

        return self.mesh.dim

    @property
    def coord(self) -> str:
        """Vector space coordinate system. Either `xyz` (Cartesian) or `rz`. `self.mesh.domain` object defines the system."""
        return self.mesh.coord_sys

    @property
    def p_order(self) -> int:
        """Order of the polynomial basis."""
        return self._p_order

    @property
    def mono(self) -> Tensor:
        """Sets of the monomial. See `pymaxed.tools.get_mono` for more details."""
        return self._mono

    @property
    def a(self) -> Tensor:
        """Identity matrix of the order of the polynomial basis."""
        return torch.eye(
            self.p_order, dtype=self.dtype.float, device=self.device
        ).detach()

    @property
    def dtype(self) -> DType:
        """Data type of `torch.tensor`."""
        return self.mesh.dtype

    @property
    def device(self) -> torch.device:
        """Torch device"""
        return self.mesh.device

    @property
    def alpha(self) -> float:
        """Scaling factor of the moments. This is used to normalize the moments, therefore, reduce the change of having numerical error during the optimization process."""
        return (
            1.0
            / (
                torch.prod(
                    torch.arange(
                        1,
                        2 * self.mnts_order,
                        2,
                        dtype=self.dtype.float,
                        device=self.device,
                    )
                )
                ** (1.0 / (2 * self.mnts_order))
            ).item()
        )

    @property
    def x(self) -> tuple[Tensor, ...]:
        """Scaled GL nodes"""
        return self._x

    @property
    def w(self) -> Tensor:
        """Scaled GL weights"""
        return self._w

    @property
    def v(self) -> tuple[Tensor, ...]:
        """Scaled original vector space."""
        return self._v

    @property
    def dv(self) -> tuple[Tensor, ...]:
        """De-scaled original vector space"""
        return self._dv

    @property
    def p(self) -> Tensor:
        """Polynomial basis."""
        return self._p

    @property
    def dp(self) -> Tensor:
        """De-scaled polynomial basis."""
        return self._dp

    @cached_property
    def init(self) -> torch.nn.Parameter:
        """Initial guess for the lagrange multiplier. Should be from the Gaussian distribution."""
        initial_guess = torch.zeros(
            self.p_order, dtype=self.dtype.float, device=self.device
        )

        if self.coord == "xyz":
            if self.dim == 1:
                initial_guess[2] = -0.5
            elif self.dim == 2:
                initial_guess[3] = -0.5
                initial_guess[5] = -0.5
            elif self.dim == 3:
                initial_guess[4] = -0.5
                initial_guess[6] = -0.5
                initial_guess[9] = -0.5
        elif self.coord == "rz":
            initial_guess[2] = -0.5
            initial_guess[5] = -0.5

        return torch.nn.Parameter(initial_guess)

    def get_poly(self, coeffs: Tensor, p_k: Tensor, origin: bool = False):
        """Construct full polynomials based on the coefficients, p_k, and the polynomial basis. If `origin` is `True`, the polynomial is constructed on the descaled-original space. Otherwise, it is constructed on the scaled GL nodes."""

        if origin:
            return _poly_point(coeffs, self.dp, p_k)
        else:
            return _poly_point(coeffs, self.p, p_k)

    def get_poly_target(
        self, p1: Tensor, t1: int, p2: Tensor | None = None, t2: int | None = None
    ) -> Tensor:
        """Get polynomial target.
        Args:
            p1 (Tensor): polynomial basis 1.
            t1 (int): target moment index 1.
            p2 (Tensor, optional): polynomial basis 2. Defaults to None.
            t2 (int, optional): target moment index 2. Defaults to None.

        Returns:
            Tensor: poly target at the desired grid.
        """
        return _poly_target_point(self.p, p1, t1, p2, t2)

    def setup_space(self) -> None:
        """Setup the vector space based on the Gauss-Legendre polynomial."""
        self._p_order, self._mono = get_mono(
            self.mnts_order, self.dim, self.dtype, self.device, self.coord
        )

        pw_pairs = [
            gl_setup(n_gl, lower.item(), upper.item(), self.dtype, self.device)
            for n_gl, lower, upper in zip(
                self.n_gl_nodes, self.mesh.lower, self.mesh.upper
            )
        ]

        # Scaled points and weights
        points = [p * self.alpha for p, _ in pw_pairs]
        weights = [w * self.alpha for _, w in pw_pairs]

        # Scaled original space
        self._v = tuple([v * self.alpha for v in self.mesh.grid])
        # De-scaled original space
        self._dv = self.mesh.grid

        # torch.meshgrid is by defaults in `ij` order
        # Scaled GL nodes
        self._x = torch.meshgrid(points, indexing="ij")

        # Scaled GL weights
        w = torch.cat(torch.meshgrid(weights, indexing="ij")).view(
            self.dim, *[w.shape[0] for w in weights]
        )
        self._w = torch.prod(w, dim=0)

        if self.coord == "rz":
            self._w *= 2.0 * pi * self._x[0]

        # Polynomial basis on the scaled GL nodes
        self._p = _poly_basis(self.mono, self.x, self.p_order)

        # Polynomial basis on the original space
        self._u = _poly_basis(self.mono, self.v, self.p_order)

        # Polynomial basis on the descaled original space
        self._dp = _poly_basis(self.mono, self.dv, self.p_order)


def _poly_basis(mono: Tensor, loc: tuple[Tensor, ...], p_order: int) -> Tensor:
    r"""Construct polynomial basis.

    .. math:

        \Phi(u) = [1, u, u^2, ...]

    Note:
        - Currently only supports upto 3D vector space.

    Args:
        mono (Tensor): monomial of the order of the moments
        loc (tuple[tensor, ...]): basis space
        p_order (int): order of polynomial power
    """

    dim = mono.shape[1]

    # shape = p_order x loc[0].shape
    poly_basis = torch.ones_like(loc[0]).repeat(p_order, *[1 for _ in range(dim)])

    perm = [i for i in range(dim, -1, -1)]

    for i, x in enumerate(loc):
        repeat_shape = x.T.shape if dim != 1 else x.shape
        x_r = x.repeat(p_order, *[1 for _ in range(dim)])
        mono_r = mono[:, i].unsqueeze(-1).T.repeat(*repeat_shape, 1).permute(perm)
        poly_basis *= x_r**mono_r

    return poly_basis


def _poly_point(coeffs: Tensor, basis: Tensor, p_k: Tensor) -> Tensor:
    """Get polynomial value at the grid.

    Args:
        coeffs (Tensor): Lagrangian multiplier
        basis (Tensor): polynomial basis
        p_k (Tensor): polynomial basis matrix. `p_k.shape == (p_order, p_order)`
            if p_k is eye(p_order), this means no polynomial
            orthogonalization applied.

    """
    # Leading dimension of the basis is the length of the total moments
    dim = basis.ndim - 1

    perm = [dim] + [i for i in range(0, dim)]

    return (((p_k @ coeffs).repeat(*basis[0].shape, 1)).permute(perm) * basis).sum(
        dim=0
    )


def _poly_target_point(
    basis: Tensor, p1: Tensor, t1: int, p2: Tensor | None = None, t2: int | None = None
) -> Tensor:
    """Get target poly on the grid.

    Args:
        basis (Tensor): polynomial basis
        p1 (Tensor): polynomial basis matrix 1
        t1 (int): target order of moment 1
        p2 (Optional, Tensor): polynomial matrix 2
        t2 (Optional, int): target order moment 2

    Returns:
        Tensor: target polynomials
    """

    dim = len(basis.shape) - 1
    perm = [dim] + list(range(0, dim))

    if p2 is None:
        pass
        return ((p1[:, t1].repeat(*basis[0].shape, 1)).permute(perm) * basis).sum(dim=0)
    else:
        if dim == 2:
            pass
        poly_1 = (p1[:, t1].repeat(*basis[0].shape, 1)).permute(perm) * basis
        poly_2_sum = ((p2[:, t2].repeat(*basis[0].shape, 1)).permute(perm) * basis).sum(
            dim=0
        )

        return (poly_1 * poly_2_sum).sum(dim=0)
