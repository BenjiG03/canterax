"""Mechanism loading utilities backed by Cantera YAML parsing."""

from functools import partial

import yaml
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import cantera as ct
from jax.experimental import sparse
from .mech_data import MechData


@partial(jax.jit, static_argnames=("nse",))
def _build_bcoo(dense, nse):
    """Fused BCOO conversion: one compiled program, not one per internal op.

    ``nse`` is a static Python int (the caller counts nonzeros on the
    plain-numpy array before it ever becomes a jax array) so the sparse
    output has a fixed, traceable shape -- see the load_mechanism call site.
    """
    return sparse.BCOO.fromdense(dense, nse=nse)


def load_mechanism(yaml_file: str) -> MechData:
    """Load a Cantera YAML mechanism and return a MechData object.
    
    This function uses Cantera to parse the YAML and then extracts the 
    relevant data into JAX arrays, performing unit conversions to 
    mol-based units. Compatible with Cantera 3.0+.
    """
    sol = ct.Solution(yaml_file)
    n_species = sol.n_species
    n_reactions = sol.n_reactions
    
    # 1. Species Data
    species_names = tuple(sol.species_names)
    # MW: kg/kmol
    mol_weights = np.asarray(sol.molecular_weights)

    # Element Data
    element_names = tuple(sol.element_names)
    n_elements = len(element_names)
    atomic_weights = np.asarray(sol.atomic_weights)
    element_matrix = np.zeros((n_elements, n_species))
    for i, spec in enumerate(sol.species()):
        for el, count in spec.composition.items():
            element_matrix[sol.element_index(el), i] = count
    
    # NASA-7 Coefficients
    nasa_low = np.zeros((n_species, 7))
    nasa_high = np.zeros((n_species, 7))
    nasa_T_mid = np.zeros(n_species)
    nasa_T_low = np.zeros(n_species)
    nasa_T_high = np.zeros(n_species)
    
    for i, spec in enumerate(sol.species()):
        poly = spec.thermo
        if not isinstance(poly, ct.NasaPoly2):
            raise ValueError(f"Species {spec.name} does not use NASA-7 (NasaPoly2) format.")
        
        coeffs = poly.coeffs
        nasa_T_mid[i] = coeffs[0]
        nasa_high[i] = coeffs[1:8]
        nasa_low[i] = coeffs[8:15]
        
        nasa_T_low[i] = poly.min_temp
        nasa_T_high[i] = poly.max_temp

    viscosity_poly = np.zeros((n_species, 5))
    conductivity_poly = np.zeros((n_species, 5))
    for i in range(n_species):
        viscosity_poly[i] = sol.get_viscosity_polynomial(i)
        conductivity_poly[i] = sol.get_thermal_conductivity_polynomial(i)

    # 2. Reaction Data
    reactant_stoich = np.zeros((n_reactions, n_species))
    product_stoich = np.zeros((n_reactions, n_species))
    
    A = np.zeros(n_reactions)
    b = np.zeros(n_reactions)
    Ea = np.zeros(n_reactions) # kJ/mol
    
    is_three_body = np.zeros(n_reactions, dtype=bool)
    efficiencies = np.ones((n_reactions, n_species))
    is_reversible = np.zeros(n_reactions, dtype=bool)
    
    is_falloff = np.zeros(n_reactions, dtype=bool)
    A_low = np.zeros(n_reactions)
    b_low = np.zeros(n_reactions)
    Ea_low = np.zeros(n_reactions) # kJ/mol
    
    troe_params = np.zeros((n_reactions, 4))
    has_troe = np.zeros(n_reactions, dtype=bool)
    
    for i, rxn in enumerate(sol.reactions()):
        is_reversible[i] = rxn.reversible
        # Stoichiometry
        for sp, coeff in rxn.reactants.items():
            reactant_stoich[i, sol.species_index(sp)] = coeff
        for sp, coeff in rxn.products.items():
            product_stoich[i, sol.species_index(sp)] = coeff
            
        # Reaction types and parameters
        r_type = rxn.reaction_type
        rate = rxn.rate
        
        if 'three-body' in r_type:
            is_three_body[i] = True
            if rxn.third_body:
                # Get the default efficiency for this reaction (0.0 for "explicit" third bodies)
                default_eff = rxn.third_body.default_efficiency
                efficiencies[i, :] = default_eff  # Set all to default first
                
                eff_map = rxn.third_body.efficiencies
                for sp_name, eff in eff_map.items():
                    efficiencies[i, sol.species_index(sp_name)] = eff
            
            A[i] = rate.pre_exponential_factor
            b[i] = rate.temperature_exponent
            Ea[i] = rate.activation_energy
            
        elif 'falloff' in r_type:
            is_falloff[i] = True
            is_three_body[i] = True # Falloff is implicitly a 3-body collision
            
            if rxn.third_body:
                # Get the default efficiency for this reaction
                default_eff = rxn.third_body.default_efficiency
                efficiencies[i, :] = default_eff  # Set all to default first
                
                eff_map = rxn.third_body.efficiencies
                for sp_name, eff in eff_map.items():
                    efficiencies[i, sol.species_index(sp_name)] = eff
            
            # High-pressure limit
            A[i] = rate.high_rate.pre_exponential_factor
            b[i] = rate.high_rate.temperature_exponent
            Ea[i] = rate.high_rate.activation_energy
            
            # Low-pressure limit
            A_low[i] = rate.low_rate.pre_exponential_factor
            b_low[i] = rate.low_rate.temperature_exponent
            Ea_low[i] = rate.low_rate.activation_energy
            
            if hasattr(rate, 'falloff_coeffs') and len(rate.falloff_coeffs) > 0:
                has_troe[i] = True
                tp = rate.falloff_coeffs
                # Troe can be 3 or 4 parameters
                troe_params[i, :len(tp)] = tp
                if len(tp) == 3:
                    troe_params[i, 3] = 1e30 # Large T** effectively disables it
            else:
                # Lindemann limit represented with a degenerate Troe parameter set.
                has_troe[i] = True
                troe_params[i] = [1.0, 1e30, 1e30, 1e30]
                
        else: # Elementary reaction
            A[i] = rate.pre_exponential_factor
            b[i] = rate.temperature_exponent
            Ea[i] = rate.activation_energy
            
    net_stoich = product_stoich - reactant_stoich

    # 3. Process Sparse Data
    # Determine max dimensions
    n_reac_list = [len(r.reactants) for r in sol.reactions()]
    n_prod_list = [len(r.products) for r in sol.reactions()]
    max_reactants = max(n_reac_list) if n_reac_list else 0
    max_products = max(n_prod_list) if n_prod_list else 0
    
    # Max count of explicit third-body efficiencies.
    n_eff_list = []
    for rxn in sol.reactions():
        if rxn.third_body:
            n_eff_list.append(len(rxn.third_body.efficiencies))
        else:
            n_eff_list.append(0)
    max_efficiencies = max(n_eff_list) if n_eff_list else 0

    reactants_idx = np.full((n_reactions, max_reactants), n_species, dtype=np.int32)
    reactants_nu = np.zeros((n_reactions, max_reactants))
    products_idx = np.full((n_reactions, max_products), n_species, dtype=np.int32)
    products_nu = np.zeros((n_reactions, max_products))
    
    efficiencies_idx = np.full((n_reactions, max_efficiencies), n_species, dtype=np.int32)
    efficiencies_val = np.zeros((n_reactions, max_efficiencies))
    default_efficiency = np.zeros(n_reactions)

    for i, rxn in enumerate(sol.reactions()):
        for j, (sp, nu) in enumerate(rxn.reactants.items()):
            reactants_idx[i, j] = sol.species_index(sp)
            reactants_nu[i, j] = nu
        for j, (sp, nu) in enumerate(rxn.products.items()):
            products_idx[i, j] = sol.species_index(sp)
            products_nu[i, j] = nu
        
        if rxn.third_body:
            default_efficiency[i] = rxn.third_body.default_efficiency
            for j, (sp, eff) in enumerate(rxn.third_body.efficiencies.items()):
                efficiencies_idx[i, j] = sol.species_index(sp)
                efficiencies_val[i, j] = eff
        else:
            default_efficiency[i] = 1.0

    # 4. Experimental BCOO Sparse Matrices
    # Only net_stoich and efficiencies feed the traced kinetics path
    # (kinetics.py's compute_wdot); reactant_stoich/product_stoich have no
    # sparse consumer anywhere in canterax, so building their BCOO forms was
    # dead computation -- and, along with the two below, one of the setup-time
    # eager-dispatch storms W1 (jax_dmrj/PERF_AUDIT.md) traced back to
    # load_mechanism. Two things made BCOO.fromdense eager: its default
    # nse=None infers the nonzero count via jax ops (an untraceable dynamic
    # shape), and it was called bare, outside any jit. Passing a static nse
    # (counted on the plain-numpy dense array, before any jax array exists)
    # fixes the shape, and _build_bcoo below fuses the conversion into one
    # compiled program instead of each internal primitive dispatching
    # separately.
    net_stoich_sparse = _build_bcoo(jnp.array(net_stoich), int(np.count_nonzero(net_stoich)))
    efficiencies_sparse = _build_bcoo(jnp.array(efficiencies), int(np.count_nonzero(efficiencies)))

    # Fields below are handed to MechData as plain NumPy, not jnp.array(...).
    # Every jnp.array(...) conversion here used to be its own separately-
    # dispatched "stage" compile at load time (~20 tiny eager kernels per
    # mechanism load, part of the W1 eager-setup-dispatch storm -- see
    # PERF_AUDIT.md); MechData's fields flow into jax.jit-decorated functions
    # (compute_thermo_state, compute_wdot, equilibrate_hp_fixed_shape, ...)
    # either directly as jit arguments or as explicit array arguments pulled
    # out beforehand, and JAX converts NumPy leaves to device arrays as part
    # of that call, at no extra eager cost. mol_weights/atomic_weights are
    # already jnp (needed eagerly above, for the sol.molecular_weights /
    # sol.atomic_weights conversion); the sparse BCOO fields must stay jax
    # arrays (built via _build_bcoo above).
    return MechData(
        n_species=n_species,
        species_names=species_names,
        mol_weights=mol_weights,
        n_elements=n_elements,
        element_names=element_names,
        element_matrix=element_matrix,
        atomic_weights=atomic_weights,
        nasa_low=nasa_low,
        nasa_high=nasa_high,
        nasa_T_mid=nasa_T_mid,
        nasa_T_low=nasa_T_low,
        nasa_T_high=nasa_T_high,
        reference_pressure=float(sol.reference_pressure),
        min_temp=float(sol.min_temp),
        max_temp=float(sol.max_temp),
        thermo_model=sol.thermo_model,
        phase_of_matter=sol.phase_of_matter,
        n_reactions=n_reactions,
        max_reactants=max_reactants,
        max_products=max_products,
        reactants_idx=reactants_idx,
        reactants_nu=reactants_nu,
        products_idx=products_idx,
        products_nu=products_nu,
        reactant_stoich=reactant_stoich,
        product_stoich=product_stoich,
        net_stoich=net_stoich,
        A=A,
        b=b,
        Ea=Ea,
        is_three_body=is_three_body,
        max_efficiencies=max_efficiencies,
        efficiencies_idx=efficiencies_idx,
        efficiencies_val=efficiencies_val,
        default_efficiency=default_efficiency,
        efficiencies=efficiencies,
        is_reversible=is_reversible,
        is_falloff=is_falloff,
        A_low=A_low,
        b_low=b_low,
        Ea_low=Ea_low,
        troe_params=troe_params,
        has_troe=has_troe,
        net_stoich_sparse=net_stoich_sparse,
        efficiencies_sparse=efficiencies_sparse,
        transport_model=sol.transport_model,
        viscosity_poly=viscosity_poly,
        conductivity_poly=conductivity_poly
    )
