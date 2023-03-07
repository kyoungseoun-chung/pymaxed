#!/usr/bin/env python3
import torch
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pymaxed.minimize import minimize_bfgs


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
