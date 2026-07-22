import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.solution import Solution

def test_equilibrium():
    yaml_path = "gri30.yaml"
    sol_can = ct.Solution(yaml_path)
    sol_jan = Solution(yaml_path)
    
    # Condition 1: Air at 1000K
    T, P = 1000.0, 101325.0
    X0 = 'O2:0.21, N2:0.79'
    
    print(f"Testing equilibrium at {T}K, 1 atm...")
    
    # Cantera
    sol_can.TPX = T, P, X0
    sol_can.equilibrate('TP')
    Y_can = sol_can.Y
    
    # Canterax
    sol_jan.set_TPX(T, P, X0)
    sol_jan.equilibrate('TP')
    Y_jan = sol_jan.Y
    
    # Compare
    err = np.max(np.abs(Y_jan - Y_can))
    print(f"  Max mass fraction error: {err:.2e}")
    
    # Condition 2: Stoichiometric Methane-Air at 2000K
    T, P = 2000.0, 101325.0
    X0 = 'CH4:1.0, O2:2.0, N2:7.52'
    
    print(f"Testing equilibrium at {T}K, 1 atm (Combustion Products)...")
    
    sol_can.TPX = T, P, X0
    sol_can.equilibrate('TP')
    Y_can_comb = sol_can.Y
    
    sol_jan.set_TPX(T, P, X0)
    sol_jan.equilibrate('TP')
    Y_jan_comb = sol_jan.Y
    
    err_comb = np.max(np.abs(Y_jan_comb - Y_can_comb))
    print(f"  Max mass fraction error (combustion): {err_comb:.2e}")
    
    # Plotting
    os.makedirs("tests/outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(sol_jan.n_species), Y_can_comb, alpha=0.5, label='Cantera')
    plt.bar(np.arange(sol_jan.n_species), Y_jan_comb, alpha=0.5, label='Canterax')
    plt.title(f"Equilibrium Composition at {T}K")
    plt.xlabel("Species Index")
    plt.ylabel("Mass Fraction")
    plt.yscale('log')
    plt.ylim(1e-10, 1)
    plt.legend()
    plt.savefig("tests/outputs/equilibrium_validation.png")
    
    assert err < 1e-4 # Less strict for equilibrium penalty method
    assert err_comb < 1e-3


def test_equilibrate_hp_fixed_shape_matches_solution_equilibrate():
    """equilibrate_hp_fixed_shape (one fused jax.jit HP solve, see its
    docstring) must match Solution.equilibrate("HP") to solver tolerance --
    it exists purely to avoid the ~200 separately-dispatched eager kernels
    the general Cantera-compatible path costs at setup time (W1 in jax_dmrj),
    not to change the physics. Requires every mechanism ELEMENT to have
    nonzero abundance in the unburned mixture (see the function's docstring),
    so both mixtures below carry a trace of every gri30 element (O, H, C, N,
    Ar) even where that species is not the intended fuel/oxidizer -- a
    mixture missing an element (e.g. no argon) is exactly the unsupported,
    genuinely-sparse case the docstring calls out."""
    from canterax.equilibrate import equilibrate_hp_fixed_shape
    from canterax.thermo import compute_thermo_state

    yaml_path = "gri30.yaml"
    for X0, T0, P in [
        ("CH4:1.0, O2:2.0, N2:7.52, AR:0.01", 1200.0, 101325.0),
        ("H2:2.0, O2:1.0, N2:3.76, CH4:0.01, AR:0.01", 800.0, 2.0e5),
    ]:
        sol_ct = ct.Solution(yaml_path)
        sol_ct.TPX = T0, P, X0
        sol_ct.equilibrate("HP")
        T_expected, Y_expected = float(sol_ct.T), np.asarray(sol_ct.Y)

        sol_jan = Solution(yaml_path)
        sol_jan.set_TPX(T0, P, X0)
        Y0 = jnp.asarray(sol_jan.Y)
        target_h = float(compute_thermo_state(T0, P, Y0, sol_jan.mech)["h_mass"])

        T_fused, Y_fused = equilibrate_hp_fixed_shape(sol_jan.mech, T0, P, Y0, target_h)

        assert abs(float(T_fused) - T_expected) < 1e-2
        assert np.max(np.abs(np.asarray(Y_fused) - Y_expected)) < 1e-6


if __name__ == "__main__":
    try:
        test_equilibrium()
        print("Equilibrium validation passed!")
    except Exception as e:
        print(f"Equilibrium validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
