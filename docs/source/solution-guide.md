# Solution Guide

`canterax.Solution` is the main user-facing API. It wraps a loaded mechanism and exposes a Cantera-like ideal-gas state surface.

## Constructing a solution

```python
from canterax import Solution

gas = Solution("gri30.yaml")
```

Construction loads the mechanism immediately and initializes a default state:

- temperature `300 K`
- pressure `1 atm`
- composition set to pure `N2` when that species exists, otherwise the first species in the mechanism

## Setting composition

Both mole and mass fractions are accepted. Strings use Cantera-style composition syntax.

```python
gas.X = "CH4:1, O2:2, N2:7.52"
gas.Y = [0.055, 0.220, 0.725]
```

Canterax normalizes positive totals automatically. A non-positive total raises `ValueError`.

## Setting thermodynamic state

The most direct state pair is `TP`.

```python
gas.TP = 1200.0, 101325.0
gas.TPX = 1200.0, 101325.0, "CH4:1, O2:2, N2:7.52"
```

The following getter/setter pairs are implemented:

- `TP`, `TPX`, `TPY`
- `HP`, `HPX`, `HPY`
- `UV`, `UVX`, `UVY`
- `SP`, `SPX`, `SPY`
- `SV`, `SVX`, `SVY`
- `TD`, `TDX`, `TDY`
- `DP`, `DPX`, `DPY`

Passing `None` in a tuple keeps the current value for that slot.

```python
gas.TP = None, 2 * 101325.0
gas.TPX = 1500.0, None, None
```

## Basis-aware aliases

The aliases below follow `gas.basis`, which can be either `"mass"` or `"molar"`:

- `h`
- `u`
- `s`
- `g`
- `cp`
- `cv`
- `v`
- `density`

If you need an explicit unit basis regardless of current state, use the dedicated properties:

- `enthalpy_mass`, `enthalpy_mole`
- `entropy_mass`, `entropy_mole`
- `cp_mass`, `cp_mole`
- `density_mass`, `density_mole`
- `volume_mass`, `volume_mole`

## Scalar properties

Common properties include:

- `cp_mass`, `cp_mole`
- `cv_mass`, `cv_mole`
- `enthalpy_mass`, `enthalpy_mole`
- `int_energy_mass`, `int_energy_mole`
- `entropy_mass`, `entropy_mole`
- `gibbs_mass`, `gibbs_mole`
- `density_mass`, `density_mole`
- `volume_mass`, `volume_mole`
- `mean_molecular_weight`
- `viscosity`
- `thermal_conductivity`

## Species-level arrays

Canterax exposes the same kinds of standard-state and partial molar arrays users commonly inspect in Cantera:

- `standard_cp_R`
- `standard_enthalpies_RT`
- `standard_entropies_R`
- `standard_int_energies_RT`
- `standard_gibbs_RT`
- `partial_molar_cp`
- `partial_molar_enthalpies`
- `partial_molar_entropies`
- `partial_molar_int_energies`
- `chemical_potentials`
- `net_production_rates`

These are returned as NumPy arrays.

## Metadata and helper methods

Useful helpers include:

- `species_names`, `element_names`
- `n_species`, `n_reactions`, `n_elements`
- `molecular_weights`, `atomic_weights`
- `species_name(k)`, `species_index(name)`
- `element_name(m)`, `element_index(name)`
- `n_atoms(species, element)`
- `mass_fraction_dict(threshold=0.0)`
- `mole_fraction_dict(threshold=0.0)`

## Practical notes

- `Solution` is intentionally stateful and mutating, like Cantera's Python API.
- Composition setters accept strings, sequences, and arrays.
- Many internal computations use JAX arrays, but user-facing property getters typically return Python scalars or NumPy arrays.
- The current implementation targets ideal-gas mechanisms only.
