#!/usr/bin/env python3
"""Module contains some reference PDFs."""
from math import pi
from math import sqrt
from typing import TypedDict

import torch
from pyapes.backend import DType
from torch import Tensor


class RefPDFReturnType(TypedDict):
    """Return type of the reference PDFs."""

    mnts: list[float]
    vel: Tensor
    pdf: Tensor


def bi_normal_ref_1(
    mnts_order: int,
    m: float = 0.9,
    s: float = 0.3,
    bound: float = 10.0,
    n_vel: int = 500,
    dtype: DType = DType("double"),
) -> RefPDFReturnType:
    """Get 1D reference bi-normal distribution and calculate moments accordingly.
    Space is symmetric around 0 with the bounds of +- `bound`. The peaks of the PDF are equally separated by `m`.
    """

    vel = torch.linspace(-bound, bound, n_vel, dtype=dtype.float)

    mu_1 = m
    mu_2 = -mu_1
    sigma_1 = s
    sigma_2 = sqrt(2 - (sigma_1**2 + 2 * mu_1**2))

    pdf_bi_norm = 0.5 * (
        torch.exp(-0.5 * ((vel - mu_1) / sigma_1) ** 2) / (sqrt(2 * pi) * sigma_1)
        + torch.exp(-0.5 * ((vel - mu_2) / sigma_2) ** 2) / (sqrt(2 * pi) * sigma_2)
    )

    p_order = mnts_order + 1
    mnts = [
        torch.trapz((vel**i_mnts) * pdf_bi_norm, vel).item()
        for i_mnts in range(p_order)
    ]

    return {"mnts": mnts, "vel": vel, "pdf": pdf_bi_norm}


def bi_normal_ref_2(
    mnts_order: int,
    m: float = 1.2,
    s: float = 0.8,
    p: float = 0.25,
    bound: float = 10,
    n_vel: int = 500,
    dtype: DType = DType("double"),
) -> RefPDFReturnType:
    """Get 1D reference bi-normal distribution and calculate moments accordingly.
    Space is symmetric around 0 with the bounds of +- `bound`. The peaks of the PDF are equally separated by `m`.
    """

    vel = torch.linspace(-bound, bound, n_vel, dtype=dtype.float)

    pdf_bi_norm = (
        p * (torch.exp(-0.5 * ((vel - m) / s) ** 2) / (sqrt(2 * pi) * s))
    ) + ((1 - p) * (torch.exp(-0.5 * ((vel + m) / s) ** 2) / (sqrt(2 * pi) * s)))

    p_order = mnts_order + 1
    mnts = [
        torch.trapz((vel**i_mnts) * pdf_bi_norm, vel).item()
        for i_mnts in range(p_order)
    ]

    return {"mnts": mnts, "vel": vel, "pdf": pdf_bi_norm}
