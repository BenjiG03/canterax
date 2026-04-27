# Getting Started

## Installation

Clone the repository and install it into an environment that already has a supported Python and compiler stack for the JAX ecosystem.

```bash
git clone https://github.com/BenjiG03/canterax.git
cd canterax
pip install -e .
```

## Runtime dependencies

The package depends on:

- `jax`
- `jaxlib`
- `equinox`
- `diffrax`
- `optimistix`
- `lineax`
- `cantera`
- `pyyaml`
- `numpy`
- `matplotlib`

For exact package ranges currently tested in this repository, see the root `pyproject.toml`.

## First concepts

There are two user-facing entry points to know first:

- `canterax.Solution` for mechanism loading, thermodynamic state management, equilibrium, and property queries
- `canterax.ReactorNet` for integrating constant-pressure adiabatic reactor trajectories

Most workflows start with a `Solution`, then optionally pass its mechanism and initial state into a `ReactorNet`.

## Mechanism files

Canterax reads Cantera-compatible YAML mechanisms. In practice that means examples like:

- `gri30.yaml`
- `h2o2.yaml`
- custom mechanism YAML files on disk

If Cantera can locate the mechanism and the thermodynamic/kinetic models are within Canterax's current support, `Solution("mechanism.yaml")` is the normal entry point.

## What to expect from compatibility

The package is intentionally close to Cantera's ideal-gas `Solution` surface, but it is not a full replacement for every Cantera model.

Use Canterax when you need:

- a familiar ideal-gas interface
- JAX-native computation
- differentiable kinetics or reactor simulation

Keep using Cantera directly when you need:

- non-ideal phases
- broader transport coverage
- mature equilibrium modes outside `TP` and `HP`

## Next step

Continue to {doc}`quickstart` for a working `Solution` example.
