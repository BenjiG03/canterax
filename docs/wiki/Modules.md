# Module Reference

## Overview

Canterax consists of 8 core modules:

| Module | Purpose |
| --- | --- |
| `constants.py` | Physical constants |
| `mech_data.py` | Frozen mechanism data structure |
| `loader.py` | YAML mechanism parsing and mechanism extraction |
| `thermo.py` | Ideal-gas NASA-7 thermodynamics and transport property helpers |
| `kinetics.py` | Reaction rate and production-rate calculations |
| `reactor.py` | ODE integration for reactor trajectories |
| `solution.py` | User-facing Cantera-like ideal-gas `Solution` API |
| `equilibrate.py` | `TP` / `HP` equilibrium |

## `mech_data.py`

`MechData` stores the arrays extracted from Cantera for use in JAX code, including:

- species and element names
- molecular and atomic weights
- element matrix
- NASA-7 coefficients and temperature limits
- reaction stoichiometry and Arrhenius parameters
- three-body / falloff data
- transport polynomial coefficients for viscosity and thermal conductivity

## `loader.py`

`load_mechanism(yaml_file)` loads a Cantera YAML file and extracts:

- species / element metadata
- NASA-7 polynomial coefficients
- stoichiometric and kinetic arrays
- transport model name
- fitted pure-species transport polynomials from Cantera

## `thermo.py`

Provides the pure functions that back the ThermoPhase surface:

- species standard-state properties from NASA-7 fits
- mixture enthalpy, internal energy, entropy, Gibbs free energy
- density and volume in mass and molar form
- partial molar arrays and chemical potentials
- mixture viscosity and thermal conductivity

## `solution.py`

`Solution` is the current Cantera-like ideal-gas wrapper.

### State surface

- Composition / scalar state:
  `T`, `P`, `X`, `Y`, `basis`
- State pairs / triples:
  `TP`, `TPX`, `TPY`, `HP`, `HPX`, `HPY`, `UV`, `UVX`, `UVY`, `SP`, `SPX`, `SPY`, `SV`, `SVX`, `SVY`, `TD`, `TDX`, `TDY`, `DP`, `DPX`, `DPY`

### Thermodynamic properties

- Explicit forms:
  `cp_mole`, `cp_mass`, `cv_mole`, `cv_mass`, `enthalpy_mole`, `enthalpy_mass`, `int_energy_mole`, `int_energy_mass`, `entropy_mole`, `entropy_mass`, `gibbs_mole`, `gibbs_mass`, `density_mole`, `density_mass`, `volume_mole`, `volume_mass`
- Basis-aware aliases:
  `h`, `u`, `s`, `g`, `cp`, `cv`, `v`, `density`
- Transport:
  `viscosity`, `thermal_conductivity`, `transport_model`

### Species-level arrays

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

### Metadata / helpers

- `thermo_model`, `phase_of_matter`, `is_compressible`, `is_pure`
- `reference_pressure`, `min_temp`, `max_temp`, `state_size`
- `mean_molecular_weight`, `molecular_weights`, `atomic_weights`
- `species_name`, `species_index`, `element_name`, `element_index`, `n_atoms`
- `mass_fraction_dict`, `mole_fraction_dict`
- `equilibrate(...)`

For the detailed property inventory, see [[Thermodynamics]].

## `equilibrate.py`

Implements the currently supported equilibrium modes:

- `TP`
- `HP`

`TP` uses the element-potential / Gibbs formulation. `HP` wraps the `TP` solve in an outer temperature iteration to enforce constant enthalpy at fixed pressure.

## `reactor.py`

Provides the stiff reactor ODE integration used for trajectory and sensitivity validation. The current validation suite benchmarks Kvaerno5 and BDF-based paths against Cantera.
