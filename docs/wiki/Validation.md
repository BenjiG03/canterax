# Validation

This page summarizes the current measured validation results for Canterax against Cantera.

## Scope

The validation coverage now includes:

- scalar ideal-gas ThermoPhase properties
- species-level standard-state and partial molar arrays
- basis-aware aliases
- Cantera-style state setters/getters
- `TP` and `HP` equilibrium
- mixture `viscosity` and `thermal_conductivity`
- static kinetics parity
- reactor trajectory parity
- sensitivity comparisons
- performance benchmarks

The dedicated parity tests also validate the expanded `Solution` ThermoPhase surface directly against `cantera.Solution`.

## Parity Test Matrix

The targeted parity tests now cover multiple mechanisms beyond the original single-case checks.

### Equilibrium parity tests

Direct pytest equilibrium comparisons (`TP` and `HP`) now run on:

- `gri30.yaml`
  - air-like oxidizer state
  - stoichiometric methane-air state
- `h2o2.yaml`
  - hydrogen / oxygen / argon state
- `gri30_highT.yaml`
  - high-temperature CO / O2 / N2 state
- `canterax/jp10.yaml`
  - JP-10 oxidation state

These tests compare final:

- temperature
- pressure
- mass fractions

### Reactor trajectory parity tests

Direct pytest trajectory comparisons now run on:

- `gri30.yaml` methane-air ignition trajectory
- `gri30.yaml` hydrogen-air trajectory
- `canterax/jp10.yaml` JP-10 oxidation trajectory

These tests compare sampled Canterax and Cantera temperature trajectories over the full integration window, not just the final state.

## Full Pytest Status

Local project suite:

```bash
python -m pytest canterax/tests -q
```

Latest local result in this workspace:

- `36 passed`
- `0 failed`
- `0 skipped`

The JP-10 loader test now resolves against the repo-local `canterax/jp10.yaml`, so the suite no longer skips that check in this workspace.

## Validation Statistics

The following numbers come from:

```bash
python canterax/tests/test_validation_suite.py
```

Validation conditions:

- GRI-30 methane: `CH4:1, O2:2, N2:7.52` at `1500 K`, `1 atm`
- GRI-30 ethane: `C2H6:1, O2:3.5, N2:13.16` at `1500 K`, `1 atm`
- GRI-30 propane: `C3H8:1, O2:5, N2:18.8` at `1500 K`, `1 atm`
- GRI-30 hydrogen: `H2:2, O2:1, N2:3.76` at `1200 K`, `1 atm`
- JP-10: `C10H16:1, O2:14, N2:52.64` at `1500 K`, `1 atm`

### Accuracy Summary

| Mechanism | Static Max Relative Error (`wdot`) | Final Trajectory Error (`|ΔT|`, K, Kvaerno) | Equilibrium Max `|ΔY|` |
| --- | ---: | ---: | ---: |
| GRI-30 (Methane) | `3.757973e-11` | `3.224225e-06` | `1.182454e-11` |
| GRI-30 (Ethane) | `3.540406e-11` | `1.960792e-05` | `1.230127e-12` |
| GRI-30 (Propane) | `8.247500e-12` | `2.012784e-05` | `8.903989e-14` |
| GRI-30 (Hydrogen) | `2.860492e-11` | `2.852782e-06` | `3.029466e-12` |
| JP-10 | `4.453018e-10` | `1.001875e-03` | `7.396861e-15` |

### Equilibrium Solver Statistics

#### `TP` mode

| Mechanism | Canterax Warm Time (ms) | Cantera Time (ms) | Canterax Steps |
| --- | ---: | ---: | ---: |
| GRI-30 (Methane) | `76.564` | `0.000` | `223` |
| GRI-30 (Ethane) | `82.067` | `0.000` | `264` |
| GRI-30 (Propane) | `73.127` | `0.000` | `284` |
| GRI-30 (Hydrogen) | `8.428` | `0.503` | `202` |
| JP-10 | `73.567` | `0.000` | `559` |

Cantera equilibrium timings above are as reported by the local run and round to `0.000 ms` for several cases because the measured elapsed time was below the displayed precision.

#### `HP` mode

The current `HP` statistics were measured by directly timing `sol.equilibrate("HP")` against Cantera for the same validation states:

| Mechanism | `|ΔT|` (K) | Max `|ΔY|` | Canterax Time (ms) | Cantera Time (ms) |
| --- | ---: | ---: | ---: | ---: |
| GRI-30 (Methane) | `2.674487e-07` | `3.570760e-11` | `5469.880` | `6.086` |
| GRI-30 (Ethane) | `1.142639e-07` | `1.738212e-11` | `2790.987` | `1.503` |
| GRI-30 (Propane) | `3.100640e-07` | `4.944693e-11` | `2969.525` | `2.160` |
| GRI-30 (Hydrogen) | `2.351771e-07` | `3.064141e-11` | `2090.210` | `0.949` |
| JP-10 | `3.171122e-07` | `5.862354e-11` | `6506.501` | `0.868` |

`HP` equilibrium is therefore validated for correctness with very small temperature and composition discrepancies, but it is currently much slower than Cantera because it performs repeated `TP` equilibrium solves inside the outer temperature iteration.

### Sensitivity Benchmark Summary

Kvaerno sensitivity comparisons were successfully run for all five mechanisms. BDF sensitivity benchmarking is still skipped because reverse-mode differentiation through the current BDF path fails on JAX `lax` control flow.

| Mechanism | Canterax Warm Time (ms) | Cantera Time (ms) | Status |
| --- | ---: | ---: | --- |
| GRI-30 (Methane) | `31.333` | `373.185` | PASS |
| GRI-30 (Ethane) | `53.663` | `572.960` | PASS |
| GRI-30 (Propane) | `71.823` | `1015.299` | PASS |
| GRI-30 (Hydrogen) | `26.839` | `214.710` | PASS |
| JP-10 | `73.866` | `253.602` | PASS |

Representative top-sensitivity comparisons from the latest run:

- Methane: strongest Cantera / Canterax match on `R117` at `-2.92e-07` vs `-2.92e-07`
- Ethane: strongest mismatch in reported top five on `R157` with ratio `0.52`
- Propane: strongest mismatch in reported top five on `R311` with ratio `0.54`
- Hydrogen: top five reported sensitivities all matched at ratio `1.00`
- JP-10: top five reported sensitivities all matched at ratio `1.00`

### Reactor Performance Comparison

Measured on the local machine with the suite’s 1 ms reactor benchmark:

| Mechanism | Solver | Canterax Warm Time (ms) | Cantera Time (ms) | Canterax Steps | Canterax ms/Step |
| --- | --- | ---: | ---: | ---: | ---: |
| GRI-30 (Methane) | Kvaerno5 | `56.305` | `5.519` | `39` | `1.444` |
| GRI-30 (Methane) | BDF | `2076.072` | `5.519` | `17395` | `0.119` |
| GRI-30 (Ethane) | Kvaerno5 | `580.965` | `27.041` | `370` | `1.570` |
| GRI-30 (Ethane) | BDF | `602.887` | `27.041` | `5335` | `0.113` |
| GRI-30 (Propane) | Kvaerno5 | `696.516` | `28.875` | `439` | `1.587` |
| GRI-30 (Propane) | BDF | `1359.442` | `28.875` | `12708` | `0.107` |
| GRI-30 (Hydrogen) | Kvaerno5 | `212.784` | `25.174` | `146` | `1.458` |
| GRI-30 (Hydrogen) | BDF | `171.650` | `25.174` | `1497` | `0.115` |
| JP-10 | Kvaerno5 | `155.883` | `21.418` | `338` | `0.461` |
| JP-10 | BDF | `265.503` | `21.418` | `6636` | `0.040` |

## ThermoPhase Parity Coverage

The expanded `Solution` API is validated directly against Cantera for:

- `cp_mole`, `cp_mass`, `cv_mole`, `cv_mass`
- `enthalpy_mole`, `enthalpy_mass`
- `int_energy_mole`, `int_energy_mass`
- `entropy_mole`, `entropy_mass`
- `gibbs_mole`, `gibbs_mass`
- `density_mole`, `density_mass`
- `volume_mole`, `volume_mass`
- `mean_molecular_weight`
- `viscosity`
- `thermal_conductivity`
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

Validated state setter coverage:

- `HPY`
- `UVX`
- `SPY`
- `SVX`
- `TDY`
- `DPX`

Basis coverage:

- `basis = "mass"`
- `basis = "molar"`

Aliases validated in both bases:

- `h`
- `u`
- `s`
- `g`
- `cp`
- `cv`
- `v`
- `density`

## JP-10 Mechanism Note

The repo-local JP-10 mechanism used by the tests is:

- `canterax/jp10.yaml`

For provenance, a publicly available JP-10 mechanism source is also available in online mechanism collections such as the JP-10 Gao 2015 entry in [CollectionOfMechanisms](https://github.com/jiweiqi/CollectionOfMechanisms/tree/master/JetFuel/JP10/Gao2015_RMG).

The local Cantera run emits NASA polynomial continuity warnings for species `C5H11CO` at `Tmid = 1000 K`; those warnings come from the mechanism data itself and do not currently fail the validation suite.
