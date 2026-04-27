# Canterax

Canterax is a differentiable chemical kinetics library built on JAX with a Cantera-like `Solution` interface for ideal-gas workflows.

It is aimed at users who want familiar mechanism-loading and thermodynamic APIs, but also need JAX-native reactor models and differentiable kinetics calculations.

```{toctree}
:maxdepth: 2
:caption: Get Started

getting-started
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

solution-guide
equilibrium
reactors
validation
```

```{toctree}
:maxdepth: 2
:caption: Reference

api-reference
```

## What Canterax covers

- Load Cantera YAML mechanisms such as `gri30.yaml`
- Work with a Cantera-like ideal-gas `Solution` object
- Set state through familiar pairs like `TP`, `HP`, `UV`, `SP`, `SV`, `TD`, and `DP`
- Query scalar, standard-state, and partial molar thermodynamic properties
- Compute equilibrium in `TP` and `HP` modes
- Integrate constant-pressure reactor trajectories with JAX and Diffrax

## Current scope

Canterax currently focuses on ideal-gas chemistry and user-facing parity with the most common Cantera `ThermoPhase` patterns.

Notable gaps today:

- non-ideal thermodynamics
- diffusion-coefficient transport APIs
- pure-fluid and saturation properties
- equilibrium modes beyond `TP` and `HP`

## Why these docs are structured this way

The existing repository material is strong on internal architecture and validation. This Sphinx site is organized around user tasks instead:

- installing the package
- creating and interrogating a `Solution`
- changing thermodynamic state safely
- running equilibrium solves
- advancing reactor trajectories

## Project links

- Source repository: <https://github.com/BenjiG03/canterax>
- Issue tracker: <https://github.com/BenjiG03/canterax/issues>
- Package metadata: repository root `pyproject.toml`
