# Canterax

Differentiable, JAX-based chemical kinetics with a Cantera-like ideal-gas `Solution` API.
https://canterax.readthedocs.io/en/latest/

## Features

- Ideal-gas `ThermoPhase`-style `Solution` wrapper
- Cantera YAML mechanism loading
- NASA-7 thermodynamics and Arrhenius kinetics
- Cantera-like state setters including `TP`, `HP`, `UV`, `SP`, `SV`, `TD`, and `DP`
- Basis-aware aliases matching Cantera (`basis = "mass"` or `"molar"`)
- Mixture `viscosity` and `thermal_conductivity`
- `TP` and `HP` equilibrium modes
- Differentiable reactor integration with JAX and Diffrax

## Installation
From pypi:

```bash
pip install canterax
```

For local development:
```bash
git clone https://github.com/BenjiG03/canterax.git
cd canterax
pip install -e .
```

## Quickstart

```python
from canterax import Solution

gas = Solution("gri30.yaml")
gas.TPX = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"

print(gas.cp_mass)
print(gas.cv_mass)
print(gas.enthalpy_mole)
print(gas.viscosity)
print(gas.thermal_conductivity)
```

### Basis-aware aliases

```python
gas = Solution("gri30.yaml")
gas.TPX = 1200.0, 101325.0, "H2:2, O2:1, N2:3.76"

gas.basis = "mass"
print(gas.h, gas.cp, gas.v)

gas.basis = "molar"
print(gas.h, gas.cp, gas.v)
```

### State setters

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

### Equilibrium

```python
gas = Solution("gri30.yaml")
gas.TPX = 2000.0, 101325.0, "CH4:1, O2:2, N2:7.52"
gas.equilibrate("TP")

gas = Solution("gri30.yaml")
gas.TPX = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"
gas.equilibrate("HP")
```

### Species-level thermodynamic arrays

The `Solution` object exposes the standard and partial molar properties commonly used from Cantera:

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

## Supported `Solution` surface

Implemented ideal-gas thermodynamic/state API:

- State setters/getters:
  `TP`, `TPX`, `TPY`, `HP`, `HPX`, `HPY`, `UV`, `UVX`, `UVY`, `SP`, `SPX`, `SPY`, `SV`, `SVX`, `SVY`, `TD`, `TDX`, `TDY`, `DP`, `DPX`, `DPY`
- Scalar properties:
  `cp_mole`, `cp_mass`, `cv_mole`, `cv_mass`, `enthalpy_mole`, `enthalpy_mass`, `int_energy_mole`, `int_energy_mass`, `entropy_mole`, `entropy_mass`, `gibbs_mole`, `gibbs_mass`, `density_mole`, `density_mass`, `volume_mole`, `volume_mass`, `mean_molecular_weight`, `viscosity`, `thermal_conductivity`
- Basis-aware aliases:
  `h`, `u`, `s`, `g`, `cp`, `cv`, `v`, `density`
- Metadata/helpers:
  `thermo_model`, `phase_of_matter`, `reference_pressure`, `min_temp`, `max_temp`, `state_size`, `species_name`, `species_index`, `element_name`, `element_index`, `n_atoms`, `mass_fraction_dict`, `mole_fraction_dict`

Not Currently Implemented:

- non-ideal activity-based thermo
- pure-fluid / saturation properties
- plasma/electron properties
- diffusion-coefficient transport APIs

## Validation

The test suite validates the Canterax `Solution` object directly against Cantera:

- scalar thermodynamic properties
- species-level standard and partial molar arrays
- basis-aware aliases
- all implemented state setters
- `TP` and `HP` equilibrium
- mixture `viscosity` and `thermal_conductivity`

Run the suite with:

```bash
python -m pytest canterax/tests -q
```

## License

MIT License.
