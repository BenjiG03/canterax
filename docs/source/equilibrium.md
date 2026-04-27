# Equilibrium

Canterax currently supports ideal-gas equilibrium under two constraint pairs.

## Supported modes

- `TP`: fixed temperature and pressure
- `HP`: fixed enthalpy and pressure

The public entry point is the `Solution.equilibrate(...)` method.

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

## When to use each mode

Use `TP` when:

- the mixture temperature is prescribed
- you want equilibrium composition at a known operating state

Use `HP` when:

- you want an adiabatic equilibrium state at fixed pressure
- the conserved quantity is enthalpy rather than temperature

## Solver behavior

`TP` equilibrium uses the element-potential formulation inside Canterax.

`HP` equilibrium is built on top of the `TP` solver:

- it keeps the current enthalpy as the target
- it solves equilibrium composition at a trial temperature and pressure
- it iterates on temperature until the target enthalpy is matched

This means `HP` is validated for correctness, but it is currently much slower than `TP`.

## Example

```python
from canterax import Solution

gas = Solution("gri30.yaml")
gas.TPX = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"
gas.equilibrate("HP")

print(gas.T)
print(gas.X)
```

## Current limitations

- only `TP` and `HP` are implemented
- solver choices beyond the element-potential path are not currently exposed in a meaningful way
- `log_level` is present for compatibility but not a developed diagnostics surface

## Validation status

The repository test suite checks `TP` and `HP` equilibrium directly against Cantera across multiple mechanisms and compositions. See {doc}`validation` for the high-level validation summary.
