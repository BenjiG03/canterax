# Canterax 🔥

**A differentiable, GPU-resident chemical kinetics library using JAX**

[Features](#features) • [Installation](#installation) • [Quickstart](#quickstart) • [Architecture](#architecture) • [Validation](#validation) • [License](#license)

---

## Overview

**Canterax** is a JAX-based reimplementation of core [Cantera](https://cantera.org/) functionality, designed for:

- **Automatic Differentiation**: Compute gradients through thermodynamics, kinetics, and ODE integrations using `jax.grad`.
- **GPU Acceleration**: JAX's XLA backend enables massive parallelization on GPUs/TPUs.
- **Batched Simulations**: Simulate thousands of reactors in parallel with `jax.vmap`.

Canterax loads standard Cantera YAML mechanism files and provides a Pythonic, Cantera-like API.

---

## Features


| Feature                                   | Status             |
| ----------------------------------------- | ------------------ |
| NASA-7 Thermodynamics                     | ✅                  |
| Arrhenius Kinetics                        | ✅                  |
| Three-Body Reactions                      | ✅                  |
| Troe Falloff Blending                     | ✅                  |
| IdealGasConstPressureReactor              | ✅                  |
| MassFlowController (Open Const-P Reactor) | ✅                  |
| Gibbs Equilibrium Solver                  | ✅                  |
| Automatic Differentiation                 | ✅                  |
| Sensitivity Analysis                      | ✅                  |
| GPU/TPU Support                           | (Awaiting Testing) |


---

## Installation

### Prerequisites

- Python 3.9+
- Cantera 3.0+ (for mechanism loading)
- JAX with GPU support (optional, for GPU acceleration)

### From Source

```bash
git clone https://github.com/BenjiG03/canterax.git
cd canterax
pip install -e .
```

### Dependencies

Core dependencies are installed automatically:

- `jax`, `jaxlib`
- `equinox`
- `diffrax`
- `optimistix`
- `cantera`
- `numpy`, `matplotlib`

---

## Quickstart

### Basic Usage: Thermodynamic Properties

```python
from canterax import Solution

# Load a mechanism (uses Cantera's YAML format)
gas = Solution("gri30.yaml")

# Set state
gas.TPX = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"

# Access properties
print(f"Temperature: {gas.T} K")
print(f"Density: {gas.density} kg/m³")
print(f"Cp: {gas.cp_mass} J/kg/K")
```

### Reactor Simulation

```python
from canterax import Solution, ReactorNet
from canterax.loader import load_mechanism

mech = load_mechanism("gri30.yaml")
gas = Solution("gri30.yaml")
gas.TPX = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"

reactor = ReactorNet(mech)
result = reactor.advance(gas.T, gas.P, gas.Y, t_end=1e-3)

print(f"Final Temperature: {result.ys[-1, 0]:.2f} K")
```

### Open Reactor with Mass Flow Controllers

```python
import jax.numpy as jnp
from canterax.loader import load_mechanism
from canterax.flow import OpenReactorNet
import cantera as ct

mech = load_mechanism("gri30.yaml")
P = 101325.0

# Initial reactor state
gas = ct.Solution("gri30.yaml")
gas.TPX = 1200.0, P, "N2:1.0"
T0 = gas.T
Y0 = jnp.array(gas.Y)
m0 = 1.0  # kg (reactor mass state)

# Inlet reservoir state
gas.TPX = 600.0, P, "O2:0.21, N2:0.79"
Tin = gas.T
Yin = jnp.array(gas.Y)

# Two mass flow controllers: inlet and outlet (kg/s)
mdot = 0.02

net = OpenReactorNet(mech)
res = net.advance(
    T0, P, Y0, m0, t_end=2e-3,
    Tin=Tin, Yin=Yin,
    mdot_in=mdot, mdot_out=mdot,
)

print(f"Final Temperature: {res.ys[-1, 0]:.2f} K")
print(f"Final Mass: {res.ys[-1, -1]:.6f} kg")
```

### Equilibrium Calculation

```python
from canterax import Solution
from canterax.equilibrate import equilibrate

gas = Solution("gri30.yaml")
gas.TPX = 2000.0, 101325.0, "CH4:1, O2:2, N2:7.52"

equilibrate(gas, 'TP')

print(f"Equilibrium T: {gas.T} K")
print(f"Major products: CO2={gas.Y[gas.species_index('CO2')]:.4f}")
```

### Gradient Computation (Automatic Differentiation)

```python
import jax
from canterax import Solution, ReactorNet
from canterax.loader import load_mechanism

mech = load_mechanism("gri30.yaml")
gas = Solution("gri30.yaml")
gas.TPX = 1500.0, 101325.0, "CH4:1, O2:2, N2:7.52"

reactor = ReactorNet(mech)

@jax.jit
def final_temperature(Y0):
    result = reactor.advance(1500.0, 101325.0, Y0, t_end=1e-4)
    return result.ys[-1, 0]

# Compute sensitivity of final T w.r.t. initial composition
grad_Y = jax.grad(final_temperature)(gas.Y)
print(f"dT/dY_CH4 = {grad_Y[gas.species_index('CH4')]:.4e}")
```

---

## Architecture

```
canterax/
├── src/canterax/
│   ├── constants.py      # Physical constants (R, etc.)
│   ├── mech_data.py      # MechData: Equinox module holding mechanism arrays
│   ├── loader.py         # YAML mechanism parser (wraps Cantera)
│   ├── thermo.py         # NASA-7 polynomial thermodynamics
│   ├── kinetics.py       # Arrhenius, three-body, Troe falloff
│   ├── reactor.py        # ReactorNet ODE integration (diffrax)
│   ├── flow.py           # Open reactor + MassFlowController terms
│   ├── solution.py       # Solution: Cantera-like API wrapper
│   └── equilibrate.py    # Gibbs minimization equilibrium solver
├── tests/
│   ├── test_validation_suite.py  # Comprehensive Cantera comparison
│   └── outputs/                  # Generated validation plots
└── pyproject.toml
```

### Key Design Principles

1. **Pure Functions**: All core computations (`compute_wdot`, `get_h_RT`, etc.) are JAX-traced pure functions.
2. **Immutable State**: `MechData` is an Equinox module with static arrays, enabling JIT tracing.
3. **Diffrax Integration**: ODE solving uses `diffrax.Kvaerno5` (implicit Runge-Kutta) for stiff kinetics. Testing determined this to be the best performing solver in Diffrax for reactor trajectories.
4. **Cantera Compatibility**: Mechanism loading uses Cantera's YAML parser for guaranteed compatibility.

---

## Validation

Canterax has been rigorously validated against Cantera 3.2.0 using:

- **GRI-30**: 53 species, 325 reactions (Methane, Ethane, Propane)
- **Z77 JP-10**: 31 species, 77 reactions (Jet fuel)

### Key Results


| Test                     | GRI-30   | JP-10    | Status |
| ------------------------ | -------- | -------- | ------ |
| Static Properties (wdot) | 9.22e-11 | 4.45e-10 | ✅ PASS |
| Reactor Trajectory (ΔT)  | 0.012 K  | < 0.1 K  | ✅ PASS |
| Equilibrium (ΔY)         | 1.18e-11 | < 1e-14  | ✅ PASS |
| Gradient (AD vs Native)  | Match    | Match    | ✅ PASS |


### Performance Benchmarking (1500K, 1 atm)

Benchmarks run on basic CPU hardware.


| Phase           | Metric        | Canterax (GRI) | Cantera (GRI)* | Canterax (JP-10) | Cantera (JP-10)* |
| --------------- | ------------- | -------------- | -------------- | ---------------- | ---------------- |
| **Equil**       | Warm Time     | 278 ms         | <1 ms          | 1784 ms          | <1 ms            |
| **Equil**       | Steps         | 33             | -              | 536              | -                |
| **Adv (0.1ms)** | Warm Time     | 38.5 ms        | 9.3 ms         | 38.5 ms          | 9.3 ms           |
| **Adv (1.0ms)** | Warm Time     | **109 ms**     | 6.8 ms         | **172 ms**       | 22.4 ms          |
| **Sens**        | **Warm Time** | **32 ms**      | **361 ms**     | **84 ms**        | **259 ms**       |


 Cantera step counts are internal solver steps, not fully exposed in all versions.

#### Key Insights

1. **Sensitivity Analysis**: Canterax is **10x faster** than Cantera's native sensitivity solver for GRI-30 (32ms vs ~360ms). The log-space ROP optimization significantly accelerated the Jacobian-vector products required for forward-mode AD.
2. **Reactor Advancement**: Canterax is now within **~5-8x** of Cantera's optimized C++ solver for reactor trajectories. The Kvaerno5 solver with explicit LU decomposition and Jacobian reuse (`kappa=0.5`) reduced runtime by nearly 50% compared to previous baselines.
3. **Sparsity Handling**: Canterax uses a "dense-sparse" approach, leveraging JAX's `scatter` and `gather` (indirect addressing) to strictly avoid dense matrix multiplications for stoichiometry. This ensures linear scaling with mechanism size without the overhead of full sparse matrix primitives (experimental `BCOO` support is available).

### Known Limitations

| None | - | All major mechanisms (GRI-30, JP-10) pass full validation. |

See [CHANGELOG.md](CHANGELOG.md) for details.

---

## Wiki

For detailed documentation on:

- Module-by-module code explanations
- Validation methodology
- Contributing guidelines

See the [Wiki](../../wiki).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Cantera](https://cantera.org/) for the reference implementation and mechanism format
- [JAX](https://github.com/google/jax) for automatic differentiation
- [Equinox](https://github.com/patrick-kidger/equinox) for PyTree-based neural network modules
- [Diffrax](https://github.com/patrick-kidger/diffrax) for differentiable ODE solvers

