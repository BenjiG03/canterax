# Quickstart

This page shows the shortest path from mechanism loading to useful outputs.

## Create a gas object

```python
from canterax import Solution

gas = Solution("gri30.yaml")
gas.TPX = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"
```

## Query common properties

```python
print(gas.cp_mass)
print(gas.cv_mass)
print(gas.enthalpy_mole)
print(gas.viscosity)
print(gas.thermal_conductivity)
```

## Switch between mass and molar basis

Several aliases follow the active basis.

```python
gas = Solution("gri30.yaml")
gas.TPX = 1200.0, 101325.0, "H2:2, O2:1, N2:3.76"

gas.basis = "mass"
print(gas.h, gas.cp, gas.v)

gas.basis = "molar"
print(gas.h, gas.cp, gas.v)
```

## Reuse the same state with alternate setters

```python
gas = Solution("gri30.yaml")
gas.TPX = 1200.0, 2 * 101325.0, "CH4:1, O2:2, N2:7.52"

h, p, y = gas.HPY
gas.HPY = h, p, y

u, v, x = gas.UVX
gas.UVX = u, v, x

s, p = gas.SP
gas.SP = s, p
```

## Compute equilibrium

```python
gas = Solution("gri30.yaml")
gas.TPX = 2000.0, 101325.0, "CH4:1, O2:2, N2:7.52"
gas.equilibrate("TP")
```

```python
gas = Solution("gri30.yaml")
gas.TPX = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"
gas.equilibrate("HP")
```

## Run a reactor trajectory

```python
import cantera as ct
import jax.numpy as jnp

from canterax import ReactorNet, Solution

gas = Solution("gri30.yaml")
gas.TPX = 1200.0, ct.one_atm, "CH4:1.0, O2:2.0, N2:7.52"

net = ReactorNet(gas.mech)
result = net.advance(
    T0=gas.T,
    P=gas.P,
    Y0=jnp.array(gas.Y),
    t_end=1e-3,
)
```

The result from the default Diffrax path exposes time samples in `result.ts` and states in `result.ys`. The first state component is temperature; the remaining entries are species mass fractions.

## Where to go next

- {doc}`solution-guide` for the full `Solution` workflow
- {doc}`equilibrium` for supported equilibrium modes and tradeoffs
- {doc}`reactors` for solver behavior and output structure
