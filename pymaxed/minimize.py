#!/usr/bin/env python3
"""Minimize module. Most of the code copied from """
from typing import Callable
from typing import TypedDict

import torch
from torch import Tensor
from torch._vmap_internals import _vmap
from torch.optim.lbfgs import _strong_wolfe  # type: ignore


class ClosureReturnType(TypedDict):
    f: Tensor
    grad: Tensor
    hess: Tensor | None


class OptimReturnType(TypedDict):
    fun: Tensor
    x: Tensor
    grad: Tensor
    status: int
    success: bool
    message: str
    nit: int
    nfev: int
    hess_inv: Tensor


class ScalarFunction:
    """Scalar-valued objective function with autograd backend.
    This class provides a general-purpose objective wrapper which will
    compute first- and second-order derivatives via autograd as specified
    by the parameters of __init__.
    """

    def __init__(
        self,
        fun: Callable[[Tensor], Tensor],
        x_shape: torch.Size,
        hess=False,
        twice_diffable=True,
    ):
        self._fun = fun
        self._x_shape = x_shape
        self._hess = hess
        self._I = None
        self._twice_diffable = twice_diffable
        self.nfev = 0

    def fun(self, x: Tensor):
        """Evaluate the objective function."""
        if x.shape != self._x_shape:
            x = x.view(self._x_shape)
        f = self._fun(x)
        if f.numel() != 1:
            raise RuntimeError(
                "ScalarFunction: a provided function does not return scalar outputs."
            )
        self.nfev += 1

        return f

    def closure(self, x: Tensor) -> ClosureReturnType:
        """Evaluate the function, gradient, and hessian."""
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.fun(x)
            grad = torch.autograd.grad(f, x, create_graph=False)[0]

        hess: Tensor | None = None

        if self._hess:
            if self._I is None:
                self._I = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            hvp = lambda v: torch.autograd.grad(grad, x, v, retain_graph=True)[0]
            hess = _vmap(hvp)(self._I)

        return {"f": f.detach(), "grad": grad.detach(), "hess": hess}

    def dir_evaluate(self, x: Tensor, t: Tensor, d: Tensor) -> tuple[float, Tensor]:
        """Evaluate a direction and step size."""

        x = x + d.mul(t)
        x = x.detach().requires_grad_(True)

        with torch.enable_grad():
            f = self.fun(x)
        grad = torch.autograd.grad(f, x)[0]

        return f.item(), grad


class Hess:
    """Hessian evaluation"""

    def __init__(self, x: Tensor):
        self.n_updates = 0
        self.I = torch.eye(x.numel(), device=x.device, dtype=x.dtype)
        self.H = self.I.clone()

    def solve(self, grad: Tensor) -> Tensor:
        """Update search direction."""
        return self.H @ grad.neg()

    def update(self, s: Tensor, y: Tensor) -> None:
        """Update Hessian approximation."""
        rho_inv = y.dot(s)

        if rho_inv <= 1e-10:
            # curvature is negative; do not update
            return
        rho = rho_inv.reciprocal()

        if self.n_updates == 0:
            self.H.mul_(rho_inv / y.dot(y))

        R = torch.addr(self.I, s, y, alpha=-rho.item())
        torch.addr(
            torch.linalg.multi_dot((R, self.H, R.t())),
            s,
            s,
            alpha=rho.item(),
            out=self.H,
        )
        self.n_updates += 1


def minimize_bfgs(
    fun: Callable[[Tensor], Tensor],
    # Below two arguments will be needed later on
    # jac_func: Callable,
    # hess_func: Callable,
    x0: Tensor,
    lr: float = 1.0,
    max_iter: int | None = None,
    gtol: float = 1e-5,
    xtol: float = 1e-9,
    disp: bool | int = False,
) -> OptimReturnType:
    """Minimize a multivariate function with BFGS or L-BFGS.
    We choose from BFGS/L-BFGS with the `low_mem` argument.
    Parameters

    Args:
    fun (callable): scalar objective function to minimize
    x0 (Tensor): initialization point
    lr (float): step size for parameter updates. If using line search, this will be used as the initial step size for the search.
    max_iter (int, optional): maximum number of iterations to perform. Defaults to 200 * x0.numel()
    gtol (float): termination tolerance on 1st-order optimality (gradient norm).
    xtol (float): termination tolerance on function/parameter changes.
    disp (int or bool): Display (verbosity) level. Set to >0 to print status messages.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    lr = float(lr)
    disp = int(disp)

    if max_iter is None:
        max_iter = x0.numel() * 200

    # construct scalar objective function
    sf = ScalarFunction(fun, x0.shape)
    closure = sf.closure
    dir_evaluate = sf.dir_evaluate

    # compute initial f(x) and f'(x)
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)

    res = closure(x)
    f = res["f"]
    g = res["grad"]

    if disp > 1:
        print("initial fval: %0.4f" % f)

    # initial settings
    hess = Hess(x)
    d = g.neg()
    # t = min(1.0, g.norm(p=1).reciprocal()) * lr
    t = min(1.0, torch.abs(g).sum().reciprocal().item()) * lr
    n_iter = 0

    # BFGS iterations
    for n_iter in range(1, max_iter + 1):
        # ==================================
        #   compute Quasi-Newton direction
        # ==================================

        if n_iter > 1:
            d = hess.solve(g)

        # directional derivative
        gtd = g.dot(d)

        # check if directional derivative is below tolerance
        if gtd > -xtol:
            warnflag = 4
            msg = "Minimize: a non-descent direction was encountered."
            break

        # ======================
        #   update parameter
        # ======================
        #  Determine step size via strong-wolfe line search
        if gtd is None:
            gtd = g.mul(d).sum()

        # Use pytorch strong wolfe line search method
        f_new, g_new, t, _ = _strong_wolfe(
            sf.dir_evaluate, x.view(-1), t, d.view(-1), f.item(), g.view(-1), gtd
        )

        # convert back to torch scalar
        f_new: Tensor = torch.as_tensor(f_new, dtype=x.dtype, device=x.device)
        g_new: Tensor = g_new.view_as(x)
        x_new = x + d.mul(t)

        if disp > 1:
            print("iter %3d - fval: %0.4f" % (n_iter, f_new))

        # ================================
        #   update hessian approximation
        # ================================

        s = x_new.sub(x)
        y = g_new.sub(g)

        hess.update(s, y)

        # =========================================
        #   check conditions and update buffers
        # =========================================

        # convergence by insufficient progress
        if (torch.linalg.norm(s, ord=torch.inf) <= xtol) | ((f_new - f).abs() <= xtol):
            warnflag = 0
            msg = "Minimize: Optimization terminated successfully"
            break

        # update state
        f = f_new
        x.copy_(x_new)
        g.copy_(g_new)
        t = lr

        # convergence by 1st-order optimality
        if torch.linalg.norm(g, ord=torch.inf) <= gtol:
            warnflag = 0
            msg = "Minimize: optimization terminated successfully"
            break

        # precision loss; exit
        if ~f.isfinite():
            warnflag = 2
            msg = "Minimize: desired error not necessarily achieved due to precision loss."
            break

    else:
        # if we get to the end, the maximum num. iterations was reached
        warnflag = 1
        msg = "Minimize: maximum number of iterations has been exceeded."

    if disp:
        print(msg)
        print(f"\t- Current function value: {f}")
        print(f"\t- Iterations: {n_iter}")
        print(f"\t- Function evaluations: {sf.nfev}")

    return {
        "fun": f,
        "x": x.view_as(x0),
        "grad": g.view_as(x0),
        "status": warnflag,
        "success": (warnflag == 0),
        "message": msg,
        "nit": n_iter,
        "nfev": sf.nfev,
        "hess_inv": hess.H.view(2 * x0.shape),
    }
