# API Reference

This page is a compact map of the public surface exposed from `canterax`.

## Top-level imports

```python
from canterax import ReactorNet, Solution
```

## `Solution`

Main responsibilities:

- load a mechanism
- store thermodynamic state
- expose Cantera-like ideal-gas properties
- compute equilibrium
- expose net production rates

Constructor:

```python
Solution(yaml_file: str)
```

Primary mutable fields:

- `T`
- `P`
- `X`
- `Y`
- `basis`

Implemented state pairs/triples:

- `TP`, `TPX`, `TPY`
- `HP`, `HPX`, `HPY`
- `UV`, `UVX`, `UVY`
- `SP`, `SPX`, `SPY`
- `SV`, `SVX`, `SVY`
- `TD`, `TDX`, `TDY`
- `DP`, `DPX`, `DPY`

Common scalar properties:

- `cp_mass`, `cp_mole`
- `cv_mass`, `cv_mole`
- `enthalpy_mass`, `enthalpy_mole`
- `int_energy_mass`, `int_energy_mole`
- `entropy_mass`, `entropy_mole`
- `gibbs_mass`, `gibbs_mole`
- `density_mass`, `density_mole`
- `volume_mass`, `volume_mole`
- `viscosity`
- `thermal_conductivity`
- `mean_molecular_weight`

Basis-aware aliases:

- `h`
- `u`
- `s`
- `g`
- `cp`
- `cv`
- `v`
- `density`

Selected methods:

- `species_name(k)`
- `species_index(name_or_index)`
- `element_name(m)`
- `element_index(name_or_index)`
- `n_atoms(species, element)`
- `mass_fraction_dict(threshold=0.0)`
- `mole_fraction_dict(threshold=0.0)`
- `equilibrate(...)`

## `ReactorNet`

Main responsibilities:

- hold a loaded mechanism
- integrate constant-pressure adiabatic reactors

Constructor:

```python
ReactorNet(mech)
```

Primary method:

```python
advance(
    T0,
    P,
    Y0,
    t_end,
    rtol=1e-7,
    atol=1e-10,
    solver=None,
    saveat=None,
    max_steps=100000,
    dt0=1e-8,
    stepsize_controller=None,
)
```

## Internal modules

These modules are present in the package, but most users should start with `Solution` and `ReactorNet`:

- `canterax.loader`
- `canterax.thermo`
- `canterax.kinetics`
- `canterax.equilibrate`
- `canterax.flow`
- `canterax.solvers.bdf`
