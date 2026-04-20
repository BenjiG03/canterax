# Thermodynamics Module

The `thermo` module and `Solution` wrapper implement the ideal-gas `ThermoPhase` surface currently supported by Canterax.

## Core Features

- Vectorized NASA-7 species thermodynamics
- Ideal-gas mixture properties in mass and molar form
- Standard-state and partial molar property arrays
- Basis-aware aliases matching Cantera semantics
- Cantera-style state setter/getter pairs
- Mixture `viscosity` and `thermal_conductivity`

## NASA-7 Formulation

Standard NASA-7 polynomials are used:

$$ \frac{C_p}{R} = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4 $$

$$ \frac{H}{RT} = a_0 + \frac{a_1 T}{2} + \frac{a_2 T^2}{3} + \frac{a_3 T^3}{4} + \frac{a_4 T^4}{5} + \frac{a_5}{T} $$

$$ \frac{S}{R} = a_0 \ln T + a_1 T + \frac{a_2 T^2}{2} + \frac{a_3 T^3}{3} + \frac{a_4 T^4}{4} + a_6 $$

Canterax combines the standard-state species properties with ideal mixing terms and the ideal-gas equation of state to compute mixture enthalpy, internal energy, entropy, Gibbs free energy, density, and volume.

## `thermo.py` Functions

Key functions currently used by the API layer:

| Function | Description |
| --- | --- |
| `get_cp_R` | Species `Cp/R` from NASA-7 fits |
| `get_h_RT` | Species `H/RT` from NASA-7 fits |
| `get_s_R` | Species `S/R` from NASA-7 fits |
| `compute_thermo_state` | Full ideal-gas state evaluation for mixture and species-level properties |
| `compute_mixture_props` | Backward-compatible mass-based thermo tuple used by kinetics/reactor code |
| `standard_cp_R` | Standard-state species `Cp/R` |
| `standard_enthalpies_RT` | Standard-state species `H/RT` |
| `standard_entropies_R` | Standard-state species `S/R` |
| `standard_int_energies_RT` | Standard-state species `U/RT` |
| `standard_gibbs_RT` | Standard-state species `G/RT` |
| `partial_molar_cp` | Partial molar heat capacities |
| `partial_molar_enthalpies` | Partial molar enthalpies |
| `partial_molar_entropies` | Partial molar entropies |
| `partial_molar_int_energies` | Partial molar internal energies |
| `chemical_potentials` | Species chemical potentials |
| `mixture_viscosity` | Mixture viscosity via transport polynomial fits + Wilke mixing |
| `mixture_thermal_conductivity` | Mixture thermal conductivity via transport polynomial fits |

## Available `Solution` Properties

The current `canterax.Solution` ideal-gas thermodynamic surface includes the following properties.

### State variables and composition

- `T`
- `P`
- `X`
- `Y`
- `basis`

### State setter/getter pairs

- `TP`, `TPX`, `TPY`
- `HP`, `HPX`, `HPY`
- `UV`, `UVX`, `UVY`
- `SP`, `SPX`, `SPY`
- `SV`, `SVX`, `SVY`
- `TD`, `TDX`, `TDY`
- `DP`, `DPX`, `DPY`

### Explicit scalar thermodynamic properties

- `cp_mole`, `cp_mass`
- `cv_mole`, `cv_mass`
- `enthalpy_mole`, `enthalpy_mass`
- `int_energy_mole`, `int_energy_mass`
- `entropy_mole`, `entropy_mass`
- `gibbs_mole`, `gibbs_mass`
- `density_mole`, `density_mass`
- `volume_mole`, `volume_mass`
- `mean_molecular_weight`

### Basis-aware aliases

These follow the current `basis` setting:

- `h`
- `u`
- `s`
- `g`
- `cp`
- `cv`
- `v`
- `density`

### Transport properties

- `viscosity`
- `thermal_conductivity`
- `transport_model`

### Species-level standard-state arrays

- `standard_cp_R`
- `standard_enthalpies_RT`
- `standard_entropies_R`
- `standard_int_energies_RT`
- `standard_gibbs_RT`

### Species-level partial molar arrays

- `partial_molar_cp`
- `partial_molar_enthalpies`
- `partial_molar_entropies`
- `partial_molar_int_energies`
- `chemical_potentials`

### Metadata and helper properties

- `thermo_model`
- `phase_of_matter`
- `is_compressible`
- `is_pure`
- `reference_pressure`
- `min_temp`
- `max_temp`
- `state_size`
- `n_species`
- `n_reactions`
- `n_elements`
- `species_names`
- `element_names`
- `molecular_weights`
- `atomic_weights`

### Helper methods

- `species_name(index)`
- `species_index(name_or_index)`
- `element_name(index)`
- `element_index(name_or_index)`
- `n_atoms(species, element)`
- `mass_fraction_dict(threshold=0.0)`
- `mole_fraction_dict(threshold=0.0)`
- `equilibrate(...)`

## Unsupported ThermoPhase Areas

Still out of scope for this ideal-gas layer:

- non-ideal activity / activity coefficient properties
- pure-fluid / saturation / quality properties
- plasma / electron-energy properties
- diffusion-coefficient transport APIs

## Source Code

- [src/canterax/thermo.py](../../src/canterax/thermo.py)
- [src/canterax/solution.py](../../src/canterax/solution.py)
