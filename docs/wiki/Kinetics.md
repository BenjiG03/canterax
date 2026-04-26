# Kinetics Module

The `kinetics` module handles reaction rate computations, including Arrhenius rates, three-body interactions, and pressure-dependent falloff corrections.

## Core Features

- **Standard Arrhenius**: $k = A T^b \exp(-E_a/RT)$
- **Three-Body Reactions**: Efficiently calculates enhanced collision rates using third-body efficiencies.
- **Falloff Reactions**:
    - Supports Lindemann and Troe formalisms.
    - Robust blending factor $F_{cent}$ calculation.
- **Fixed-Width Sparse Stoichiometry**: The default runtime path uses padded species-index and stoichiometric-coefficient arrays rather than dense stoichiometric matrix multiplies.
- **Optional BCOO Sparse Path**: Experimental JAX `BCOO` sparse matrices are also built and can be enabled for comparison with the default path.
- **Compatibility Dense Storage**: Dense stoichiometric matrices are still stored in the mechanism data structure for compatibility and reference.
- **Log-Space ROP**: Rate of Progress (ROP) calculated in log-space for AD stability and efficiency.

## Implementation Details

### Stoichiometry Representation

Canterax currently keeps three stoichiometric representations in `MechData`:

- dense arrays:
  - `reactant_stoich`
  - `product_stoich`
  - `net_stoich`
- fixed-width sparse arrays:
  - `reactants_idx`, `reactants_nu`
  - `products_idx`, `products_nu`
- experimental JAX sparse arrays:
  - `reactant_stoich_sparse`
  - `product_stoich_sparse`
  - `net_stoich_sparse`

The default kinetics path is the fixed-width sparse form. Each reaction stores only the participating species indices and their stoichiometric coefficients, padded to a mechanism-wide maximum width. This avoids dense reaction-by-species matrix operations in the main source-term path while still keeping shapes static for JAX compilation.

### Default `wdot` Path

`compute_wdot(..., use_experimental_sparse=False)` uses the fixed-width sparse representation in two places:

- reaction rate-of-progress terms gather concentrations with `reactants_idx` / `products_idx`
- species source terms are accumulated with a scatter-add over the padded stoichiometric entries

This is the main runtime path used by the library today.

### Experimental Sparse Path

`compute_wdot(..., use_experimental_sparse=True)` switches the final source-term accumulation to the JAX `BCOO` sparse representation:

$$ \dot{\omega} = ROP \cdot \nu_{net} $$

where `nu_net` is stored as `net_stoich_sparse`.

The same flag also switches third-body efficiency handling to `efficiencies_sparse @ conc`.

### Dense Matrices

Dense stoichiometric matrices are still constructed during mechanism loading and stored on `MechData`, but they are not the default execution path for kinetics. They are primarily retained for compatibility, debugging, and parity with earlier implementations.

### Optimized Rate of Progress (ROP)
To improve Automatic Differentiation (AD) performance - especially for reverse-mode gradients - the Rate of Progress is calculated in log-space:

$$ ROP = k \cdot \exp \left( \sum_{i} \nu_i \ln([\text{Conc}]_i) \right) $$

This avoids complex chains of power/product rules during backpropagation.
### Source Code
- [src/canterax/kinetics.py](../../src/canterax/kinetics.py)
