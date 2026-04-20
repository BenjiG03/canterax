# Equilibrium

`canterax.equilibrate` and `Solution.equilibrate(...)` provide a Cantera-like equilibrium API for ideal-gas mixtures.

## Supported modes

- `TP`: hold temperature and pressure fixed, solve for equilibrium composition
- `HP`: hold enthalpy and pressure fixed, solve for equilibrium temperature and composition

## Public API

```python
gas.equilibrate(
    XY="TP",
    solver="auto",
    rtol=1e-9,
    max_steps=1000,
    max_iter=100,
    estimate_equil=0,
    log_level=0,
)
```

Supported arguments:

- `XY`: `"TP"` or `"HP"`
- `solver`: `"auto"` or `"element_potential"`
- `rtol`, `max_steps`, `max_iter`

Currently unsupported:

- other equilibrium modes
- nonzero `log_level`
- solver backends beyond the element-potential path used by Canterax

## Implementation

### `TP` mode

`TP` equilibrium uses the Gibbs / element-potential formulation already present in Canterax:

- elemental abundances are computed from the current composition
- species inconsistent with the present elements are filtered out
- a Levenberg-Marquardt solve enforces chemical potential balance, elemental balance, and total-mole consistency

### `HP` mode

`HP` equilibrium wraps the `TP` composition solver in an outer temperature solve:

- the current state defines the target enthalpy and pressure
- for each trial temperature, Canterax solves the equilibrium composition at `(T, P)`
- the outer iteration adjusts temperature until the equilibrium enthalpy matches the target enthalpy

## Validation

`TP` and `HP` equilibrium are both validated directly against Cantera in the pytest suite on:

- an air-like oxidizer state
- a reacting methane-air state

The tests compare final temperature, pressure, and composition against Cantera.
