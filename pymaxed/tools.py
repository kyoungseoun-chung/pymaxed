#!/usr/bin/env python3
"""Contains tools for the PDF approximations.

Note:
    - The methods related to find monomials are copied and modified from
      https://people.sc.fsu.edu/~jburkardt/py_src/polynomial/
"""
import numpy as np
import torch
from pyapes.backend import DType
from torch import Tensor


def get_mono(
    n_mnts: int,
    n_dim: int,
    dtype: DType = DType("double"),
    device: torch.device = torch.device("cpu"),
    coord: str = "xyz",
) -> tuple[int, Tensor]:
    """Get the power of monomial.

    * If coord is xyz (cartesian), it follows Lexicographical order
    * if coord is rz (antisymmetric), it only has upto 4th order moment
    and 9 polynomial basis.

    Example:
        >>> # n_mnts = 3, n_dim = 2
        >>> # total order = (n_mnts + n_dim)!/n_mnts!n_dim!
        >>> order, monomial = get_mono(3, 2, dtype)
        order = 10, monomial = [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0], [0, 3], [1, 2], [2, 1], [3, 0]]

    See also:

    .. code-block:: text

        [x, y]  [x, y, z]
        [0, 1]  [0, 0, 0]
        [1, 0]  [0, 0, 1]
        [0, 2]  [0, 1, 0]
        [1, 1]  [1, 0, 0]
        [2, 0]  [0, 0, 2]
        [0, 3]  [0, 1, 1]
        [1, 2]  [0, 2, 0]
        [2, 1]  [1, 0, 1]
        [3, 0]  [1, 1, 0]
        ...


    Note:
        In the order of,
        0, z, zz, zzz, zzzz, rr, rrrr, zrr, zzrr

    Args:
        n_mnts (int): order of moments.
        n_dim (int): the spatial dimension.

    Returns:
        int: total order of monomials
        Tensor, int: monomial
    """

    if coord == "xyz":
        upper = torch.prod(
            torch.arange(1, n_mnts + n_dim + 1, dtype=dtype.float, device=device)
        )
        lower = torch.prod(
            torch.arange(1, n_mnts + 1, dtype=dtype.float, device=device)
        ) * torch.prod(torch.arange(1, n_dim + 1, dtype=dtype.float))
        k = int(upper / lower)
        mono = torch.zeros((k, n_dim), dtype=dtype.int, device=device)

        for r in range(1, k + 1):
            mono[r - 1, :] = mono_unrank_grlex(n_dim, r, dtype=dtype.int, device=device)
    else:
        k = 9
        mono = torch.tensor(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [2, 0],
                [4, 0],
                [2, 1],
                [2, 2],
            ],
            dtype=dtype.int,
            device=device,
        )

    return k, mono


def mono_unrank_grlex(
    m: int, rank: int, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Calculate monomial at specific rank.
    rank follows Lexicographical order.

    Args:
        m (int): the spatial dimension.
        rank (int): the rank of monomial

    Returns:
        Tensor: monomial
    """

    x = torch.zeros(m, dtype=dtype, device=device)

    # Special case M = 1.
    # modified by k.chung
    # if rank = 0, x[0] = 0, else = 1
    # to represent, 0, x, y, z
    if m == 1:
        # * modified to cope 1D power
        x[0] = rank - 1
        return x
    #  Determine the appropriate value of NM.
    #  Do this by adding up the number of compositions of sum 0, 1, 2,
    #  ..., without exceeding RANK.  Moreover, RANK - this sum essentially
    #  gives you the rank of the composition within the set of compositions
    #  of sum NM.  And that's the number you need in order to do the
    #  unranking.
    rank1 = 1
    nm = -1
    while True:
        nm = nm + 1
        r = i4_choose(nm + m - 1, nm)
        if rank < rank1 + r:
            break
        rank1 = rank1 + r
    rank2 = rank - rank1

    #  Convert to KSUBSET format.
    #  Apology: an unranking algorithm was available for KSUBSETS,
    #  but not immediately for compositions.  One day we will come back
    #  and simplify all this.

    ks = m - 1
    ns = nm + m - 1
    xs = torch.zeros(ks, dtype=dtype, device=device)

    j = 1

    for i in range(1, ks + 1):
        r = i4_choose(ns - j, ks - i)
        while r <= rank2 and 0 < r:
            rank2 = rank2 - r
            j = j + 1
            r = i4_choose(ns - j, ks - i)
        xs[i - 1] = j
        j = j + 1

    #  Convert from KSUBSET format to COMP format.
    x[0] = xs[0] - 1
    for i in range(2, m):
        x[i - 1] = xs[i - 1] - xs[i - 2] - 1
    x[m - 1] = ns - xs[ks - 1]

    return x


def i4_choose(n: int, k: int) -> float:
    mn = min(k, n - k)
    mx = max(k, n - k)

    if mn < 0:
        value = 0
    elif mn == 0:
        value = 1
    else:
        value = mx + 1
        for i in range(2, mn + 1):
            value = (value * (mx + i)) / i

    return value


def gl_setup(
    n_deg: int, a: float, b: float, dtype: DType, device: torch.device
) -> list[Tensor]:
    """Wrapper for np.polynomial.legendre.leggauss function.
    This is used to generate node and weight defined by the Gauss-Legendre polynomial.

    Args:
        n_deg (int): number of points evaluated
        a (float): lower bound
        b (float): upper bound
        dtype (DType): `DType` object.

    Note:
        if a == 0, only takes x and w where x > 0.
        This is implemented to account asymmetric conditions.

    Returns:
        Tensor, Tensor: nodes and weights
    """

    if a == 0:
        # since we need to take only half of x and w, increase
        # number of nodes twice
        points, weights = np.polynomial.legendre.leggauss(2 * n_deg)
        # scaled nodes and weights
        x = (b - a) / 2 * points + (a + b) / 2
        w = (b - a) / 2 * weights

        idx = x > 0
        x = x[idx]
        w = w[idx]

        n_node = x.shape[0]

        # just for in case...
        assert n_node != n_deg, "number of node is not met its input"

    else:
        points, weights = np.polynomial.legendre.leggauss(n_deg)

        # scaled nodes and weights
        x = (b - a) / 2 * points + (a + b) / 2
        w = (b - a) / 2 * weights

    return [
        torch.tensor(x, dtype=dtype.float, device=device),
        torch.tensor(w, dtype=dtype.float, device=device),
    ]


def moments_sampling(
    mnts_bounds: tuple,
    dtype: DType = DType("double"),
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    r"""Sample the moments from the prescribed moment space. Only works
    for 1D up-to 4th order moments and the 4th order moments ($p_4$).

    Note:
        - Moment space has to satisfy following condition:

            .. math::
                m_4  &\geqq m_3^2 + 1~\text{and} \\
                    m_3 &= 3,~\text{if}~m_4 = 0

        Reference:
            McDonald, J., & Torrilhon, M. (2013).
            Affordable robust moment closures for CFD based on
            the maximum-entropy hierarchy.
            Journal of Computational Physics, 251, 500-523.

        - Other than above restriction, moments are sampled
          uniform randomly within self.mnts_bound

    """

    mnts = torch.zeros(5, dtype=dtype.float, device=device)

    # moments 0th, 1st, and 2nd orders are fixed
    mnts[0] = 1
    mnts[1] = 0
    mnts[2] = 1

    mnts3 = (mnts_bounds[3][1] - mnts_bounds[3][0]) * np.random.rand() + mnts_bounds[3][
        0
    ]

    mnts[3] = mnts3

    mnts4 = 0

    while mnts4 < mnts3**2 + 1:
        mnts4 = (
            mnts_bounds[4][1] - mnts_bounds[4][0]
        ) * np.random.rand() + mnts_bounds[4][0]

        if mnts4 == 0:
            break

    if mnts4 != 0.0:
        mnts[4] = mnts4
    else:
        mnts[3] = 3
        mnts[4] = 0

    return mnts
