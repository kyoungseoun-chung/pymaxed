# pymaxed: Python Module for Maximum Entropy Distribution

> Some core features are implemented in my other project [`pyapes`](https://github.com/kyoungseoun-chung/pyapes). Check the repository for more details. Due to its dependency to `pyapes`, some packages are indirectly inherited from there. (e.g. `torch` and `numpy`)

The library is heavily inspired by `minimize` module from [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) and [pytorch-minize](https://github.com/rfeinman/pytorch-minimize).

## Description

Package to compute approximated PDF using the Maximum Entropy Distribution (maxed).
Originally I wrote the code with `numba` library. Here, in this project, I refactored old code and optimize using `torch` so that we can utilize GPU computation.

## Installation

## Dependencies

- `python >= 3.10` (but `3.11` is not supported yet due to `torch`)
- `pyapes`
