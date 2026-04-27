# Validation

This page summarizes what the repository currently validates against Cantera.

## What is covered

The automated tests cover:

- scalar thermodynamic properties
- basis-aware aliases
- standard-state and partial molar arrays
- implemented state setter/getter pairs
- `TP` and `HP` equilibrium
- mixture viscosity and thermal conductivity
- static kinetics parity
- reactor trajectory parity

## Interpreting the current status

The validation evidence in this repository is strongest for:

- ideal-gas `Solution` property parity
- equilibrium correctness for `TP` and `HP`
- constant-pressure reactor temperature trajectories

It is weaker for:

- breadth of transport modeling beyond viscosity and thermal conductivity
- equilibrium modes outside the currently implemented pair
- non-ideal chemistry

## Representative plots

```{image} ../images/thermo_validation.png
:alt: Thermodynamic validation plots
:width: 90%
```

```{image} ../images/equilibrium_validation.png
:alt: Equilibrium validation plots
:width: 90%
```

## Running the tests

From the repository root:

```bash
python -m pytest tests -q
```

## Performance note

Correctness parity is ahead of performance parity. In particular, `HP` equilibrium is currently much slower than Cantera because it wraps repeated `TP` solves inside an outer temperature iteration.

## Source material

The deeper benchmark tables and internal notes that informed this page still live in the repository wiki sources under `docs/wiki/`.
