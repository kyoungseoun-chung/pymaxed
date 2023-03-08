#!/usr/bin/env python3
from typing import Any
from typing import Callable
from typing import TypedDict

import torch
from attr import dataclass
from torch import Tensor
from torch.optim.lbfgs import _strong_wolfe  # type: ignore


class ClosureReturnType(TypedDict):
    """Closure return type.

    Returns:
        f (Tensor): objective function value
        grad (Tensor): gradient of the objective function
    """

    f: Tensor
    grad: Tensor


class OptimReturnType(TypedDict):
    """Optimization result return types."""

    fun: Tensor
    x: Tensor
    p: Tensor
    grad: Tensor
    status: int
    success: bool
    message: str
    nit: int
    nfev: int
    hess_inv: Tensor
    n_ortho_eval: int


@dataclass
class FunctionTools:
    """Collection of functions tools for ScalarFunction.

    This class should contiains:
        - jac: Jacobian of the objective function.
        - hess: Hessian of the objective function.
        - ortho_func: Orthogonalization function.
        - args: arguments for the above functions. Typically, `vec, p, mnts_scaled`
        - n_eval: number of function evaluations.

    Note:
        - This object is necessary for the re-orthogonalization process.
        - The `self.hess` is not the same as the `Hess` class. `Hess` class is used in the BFGS steps ans is an approximated evaluation of the Hessian while `self.hess` is the exact Hessian of the objective function that the user should provide.
    """

    jac: Callable[..., Tensor]
    hess: Callable[..., Tensor]
    ortho_func: Callable[..., tuple[Tensor, Tensor, float]]
    args: tuple[Any, ...]
    n_eval: int = 0


class ScalarFunction:
    """Scalar-valued objective function with autograd backend (If `jac` is not explicitly provided via `FunctionTools`. Otherwise use directly `FunctionTools.jac` method)."""

    def __init__(
        self,
        fun: Callable[[Tensor], Tensor],
        x_shape: torch.Size,
        functools: FunctionTools | None = None,
    ):
        """Construct ScalarFunction object.

        Args:
            fun (Callable[[Tensor], Tensor]): target objective function.
            x_shape (torch.Size): shape of the target variable.
            functools (FunctionTools, optional): function tools. Defaults to None.
        """
        self._fun = fun
        self.x_shape = x_shape

        self.I = None
        self.nfev = 0
        self.functools = functools

        if self.functools is None:
            self.jac = None
            self.hess = None
            self.args = None
        else:
            self.jac = self.functools.jac
            self.hess = self.functools.hess
            self.args = self.functools.args

    def fun(self, x: Tensor):
        """Evaluate the objective function."""
        if x.shape != self.x_shape:
            x = x.view(self.x_shape)

        if self.args is None:
            f = self._fun(x)
        else:
            f = self._fun(x, *self.args)

        if f.numel() != 1:
            raise RuntimeError(
                "ScalarFunction: a provided function does not return scalar outputs."
            )
        self.nfev += 1

        return f

    def hess_cond(self, x: Tensor) -> Tensor:
        """Condition number of the hessian. If the hessian is not invertible, return nan.
        This is used to check whether the orthogonalization is needed or not.
        """

        hess = self.hess_eval(x)

        try:
            return torch.linalg.cond(torch.linalg.inv(hess))
        except RuntimeError or torch.linalg.LinAlgError:
            return torch.tensor(torch.nan, dtype=x.dtype, device=x.device)

    def closure(self, x: Tensor) -> ClosureReturnType:
        """Evaluate the function, gradient, and hessian."""

        x = x.detach().requires_grad_(True)

        if self.jac is None:
            with torch.enable_grad():
                f = self.fun(x)
                grad = torch.autograd.grad(f, x, create_graph=False)[0]
        else:
            assert self.args is not None, "ScalarFunction: args must be provided."
            f = self.fun(x)
            grad = self.jac(x, *self.args)

        return {"f": f.detach(), "grad": grad.detach()}

    def hess_eval(self, x: Tensor) -> Tensor:
        """Evaluate hessian from the given callable function `self.hess`."""

        assert (
            self.hess is not None
        ), "ScalarFunction: self.hess must be provided to evaluate hessian."
        assert self.args is not None, "ScalarFunction: args must be provided."
        hess = self.hess(x, *self.args).detach()

        return hess

    def dir_evaluate(self, x: Tensor, t: Tensor, d: Tensor) -> tuple[float, Tensor]:
        """Evaluate a direction and step size."""

        x = x + d.mul(t)
        x = x.detach().requires_grad_(True)

        if self.jac is None:
            with torch.enable_grad():
                f = self.fun(x)
                grad = torch.autograd.grad(f, x)[0]
        else:
            f = self.fun(x)
            assert self.args is not None, "ScalarFunction: args must be provided."
            grad = self.jac(x, *self.args)

        return f.detach().item(), grad.detach()


class Hess:
    """(Approximated) Hessian evaluation."""

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
    fun: Callable[..., Tensor],
    x0: Tensor,
    functools: FunctionTools | None = None,
    lr: float = 1.0,
    max_iter: int | None = None,
    gtol: float = 1e-5,
    xtol: float = 1e-9,
    disp: bool | int = False,
    ortho: bool = False,
) -> OptimReturnType:
    """Minimize a multivariate function with BFGS.

    Note:
        - Large portion of the code is borrowed from https://github.com/rfeinman/pytorch-minimize and modified for our purposes.
        - My implementation of the previous modified scipy version can be found in `pystops_ml` package (but it is a private repository).

    Example:
        >>> def objective(x): ... # target objective function
        >>> B0 = torch.zeros(...) # target variable
        >>> res = minimize_bfgs(objective, B0)

    Args:
        fun (Callable): scalar objective function to minimize
        x0 (Tensor): initialization point
        lr (float): step size for parameter updates. If using line search, this will be used as the initial step size for the search.
        max_iter (int, optional): maximum number of iterations to perform. Defaults to 200 * x0.numel()
        gtol (float): termination tolerance on 1st-order optimality (gradient norm).
        xtol (float): termination tolerance on function/parameter changes.
        disp (int or bool): Display (verbosity) level. Set to >0 to print status messages.

    Returns:
        result (OptimReturnType): Result of the optimization routine.
    """
    lr = float(lr)
    disp = int(disp)

    if max_iter is None:
        max_iter = x0.numel() * 200

    # construct scalar objective function
    sf = ScalarFunction(fun, x0.shape, functools)

    # compute initial f(x) and f'(x)
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)

    res = sf.closure(x)
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
        # Quasi-Newton direction
        if n_iter > 1:
            d = hess.solve(g)

        # Searching direction
        gtd = g.dot(d)

        # Orthogonalization step using the modified Gram-schmidt algorithm
        if ortho:
            assert (
                functools is not None
            ), "Minimize: functools must be provided for orthogonalization."

            if sf.hess_cond.item() > 10:
                ortho_func = functools.ortho_func
                args = list(functools.args)
                p, x, ortho_err = ortho_func(x, d, args[0], args[1])

                if ortho_err:
                    warnflag = 5
                    msg = "Minimize: Orthogonalization failed."
                    break

                # Update orthogonalized polynomial
                args[1] = p
                functools.args = tuple(args)

                d = sf.closure(x)["grad"].neg()
                lhess = sf.hess_eval(x)

                try:
                    gtd = torch.linalg.inv(lhess).dot(d)
                except RuntimeError or torch.linalg.LinAlgError:
                    warnflag = 5
                    msg = "Minimize: Orthogonalization failed."
                    break

                functools.n_eval += 1

        # Update parameter
        # Use pytorch strong wolfe line search method
        f_new, g_new, t, _ = _strong_wolfe(
            sf.dir_evaluate, x.view(-1), t, d.view(-1), f.item(), g.view(-1), gtd
        )

        # convert back to torch scalar
        f_new = torch.as_tensor(f_new, dtype=x.dtype, device=x.device)
        g_new = g_new.view_as(x)
        x_new = x + d.mul(t)

        if disp > 1:
            print("iter %3d - fval: %0.4f" % (n_iter, f_new))

        s = x_new.sub(x)
        y = g_new.sub(g)

        # Update Hessian approximation
        hess.update(s, y)

        # Convergence check
        if (torch.linalg.norm(s, ord=torch.inf) <= xtol) | ((f_new - f).abs() <= xtol):
            warnflag = 0
            msg = "Minimize: Optimization terminated successfully."
            break

        # update state
        f = f_new
        x.copy_(x_new)
        g.copy_(g_new)
        t = lr

        # Convergence by 1st-order optimality
        if torch.linalg.norm(g, ord=torch.inf) <= gtol:
            warnflag = 0
            msg = "Minimize: optimization terminated successfully."
            break

        # Precision loss; exit
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
        "p": functools.args[1]
        if functools is not None
        else torch.zeros_like(x.view_as(x0)),
        "grad": g.view_as(x0),
        "status": warnflag,
        "success": (warnflag == 0),
        "message": msg,
        "nit": n_iter,
        "nfev": sf.nfev,
        "hess_inv": hess.H.view(2 * x0.shape),
        "n_ortho_eval": functools.n_eval if functools is not None else -1,
    }
