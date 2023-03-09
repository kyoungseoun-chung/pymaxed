# pymaxed: Python Module for Maximum Entropy Distribution

> Some core features are implemented in my other project [`pyapes`](https://github.com/kyoungseoun-chung/pyapes). Check the repository for more details. Due to its dependency to `pyapes`, some packages are indirectly inherited from there. (e.g. `torch` and `numpy`)

The library is heavily inspired by `minimize` module from [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) and [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize).

## Description

Package to compute approximated PDF using the Maximum Entropy Distribution (maxed).
Originally I wrote the code with `numba` library. Here, in this project, I refactored old code and optimize using `torch` so that we can utilize GPU computation.

## Dependencies

- `python >= 3.10` (but `3.11` is not supported yet due to `torch`)
- `pyapes`

## Installation

You can install the package using `pip`.

```bash
python -m pip install pymaxed
```

## Usage

To work with `pymaxed` package, you need `pyapes` package since it contains `geometry` and `mesh` related tools.

* Step 0: import relevant modules

  ```python
  from pymaxed.maxed import Maxed
  from pymaxed.vectors import Vec
  from pyapes.core.mesh import Mesh
  from pyapes.core.geometry import Box, Cylinder
  ```

* Step 1: set up the target set of moments. The number of moments will be the degree of freedom to be used in the reconstruction of the distribution.

  ```python
  target = [1, 0, 1, -0.27, 1.7178]
  ```

* Step 2: construct mesh and vector space.

  ```python
    mesh = Mesh(Box[-5:5], None, [100]) # If you want to work on axisymmetric domain, use Cylinder instead
    vec = Vec(mesh, target, 4, [100])
  ```
  
* Step 3: create `Maxed` object and solve the optimization problem.

  ```python
    maxed = Maxed(vec)
    maxed.solve()
    # After calling solve() method, you can access obtained coefficients via maxed.coeffs
    # And reconstructed PDF via maxed.dist
    # Reconstructed PDF (MaxEd) lies on vec.dv space.
  ```

For more details, check our [demo case](./demo/maxed.ipynb).
