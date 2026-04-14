import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import cantera as ct
import equinox as eqx

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.flow import OpenReactorNet, open_constp_reactor_rhs


def _disable_reactions_cantera(gas: ct.Solution):
    # Cantera API supports per-reaction multipliers.
    gas.set_multiplier(0.0)


def _disable_reactions_canterax(mech):
    mech = eqx.tree_at(lambda m: m.A, mech, jnp.zeros_like(mech.A))
    if hasattr(mech, "A_low") and mech.A_low is not None:
        mech = eqx.tree_at(lambda m: m.A_low, mech, jnp.zeros_like(mech.A_low))
    return mech


def test_mass_flow_controller_parity_no_reactions():
    """Parity vs Cantera for an open const-P reactor with inlet/outlet MFCs.

    Chemistry is disabled so the test isolates flow + mixing + enthalpy terms.
    """
    yaml_path = "gri30.yaml"
    mech = _disable_reactions_canterax(load_mechanism(yaml_path))

    P = 101325.0
    T_reac0 = 1200.0
    X_reac0 = "N2:1.0"

    T_in = 600.0
    X_in = "O2:0.21, N2:0.79"

    mdot = 0.02  # kg/s
    t_end = 2e-3

    # --- Cantera reference ---
    gas_reac = ct.Solution(yaml_path)
    _disable_reactions_cantera(gas_reac)
    gas_reac.TPX = T_reac0, P, X_reac0
    reactor = ct.IdealGasConstPressureReactor(gas_reac)

    gas_in = ct.Solution(yaml_path)
    _disable_reactions_cantera(gas_in)
    gas_in.TPX = T_in, P, X_in
    inlet = ct.Reservoir(gas_in)

    gas_out = ct.Solution(yaml_path)
    _disable_reactions_cantera(gas_out)
    gas_out.TPX = T_reac0, P, X_reac0
    outlet = ct.Reservoir(gas_out)

    mfc_in = ct.MassFlowController(inlet, reactor, mdot=mdot)
    mfc_out = ct.MassFlowController(reactor, outlet, mdot=mdot)

    net_ct = ct.ReactorNet([reactor])
    net_ct.rtol = 1e-10
    net_ct.atol = 1e-16
    net_ct.advance(t_end)

    T_ct = reactor.T
    Y_ct = reactor.thermo.Y
    m_ct = reactor.mass

    # --- Canterax ---
    gas_tmp = ct.Solution(yaml_path)
    gas_tmp.TPX = T_reac0, P, X_reac0
    Y0 = jnp.array(gas_tmp.Y)
    m0 = float(m_ct)  # start from Cantera mass for a fair comparison

    gas_tmp.TPX = T_in, P, X_in
    Yin = jnp.array(gas_tmp.Y)

    net_jt = OpenReactorNet(mech)
    res = net_jt.advance(
        T_reac0,
        P,
        Y0,
        m0,
        t_end,
        Tin=T_in,
        Yin=Yin,
        mdot_in=mdot,
        mdot_out=mdot,
        rtol=1e-10,
        atol=1e-16,
    )
    res = jax.block_until_ready(res)

    T_jt = float(res.ys[-1, 0])
    Y_jt = np.array(res.ys[-1, 1:-1])
    m_jt = float(res.ys[-1, -1])

    # --- Compare ---
    assert abs(T_jt - T_ct) < 5e-3  # K
    assert np.max(np.abs(Y_jt - Y_ct)) < 2e-10
    assert abs(m_jt - m_ct) / max(abs(m_ct), 1e-30) < 5e-10


def test_mass_flow_controller_parity_with_reactions_short():
    """Parity vs Cantera for open const-P reactor with reactions (short horizon)."""
    yaml_path = "gri30.yaml"
    mech = load_mechanism(yaml_path)

    P = 101325.0
    T_reac0 = 1400.0
    X_reac0 = "CH4:1.0, O2:2.0, N2:7.52"

    T_in = 1200.0
    X_in = "CH4:0.5, O2:2.0, N2:7.52"

    mdot = 0.01  # kg/s
    t_end = 2e-5

    # --- Cantera reference ---
    gas_reac = ct.Solution(yaml_path)
    gas_reac.TPX = T_reac0, P, X_reac0
    reactor = ct.IdealGasConstPressureReactor(gas_reac)

    gas_in = ct.Solution(yaml_path)
    gas_in.TPX = T_in, P, X_in
    inlet = ct.Reservoir(gas_in)

    gas_out = ct.Solution(yaml_path)
    gas_out.TPX = T_reac0, P, X_reac0
    outlet = ct.Reservoir(gas_out)

    _ = ct.MassFlowController(inlet, reactor, mdot=mdot)
    _ = ct.MassFlowController(reactor, outlet, mdot=mdot)

    net_ct = ct.ReactorNet([reactor])
    net_ct.rtol = 1e-9
    net_ct.atol = 1e-14
    net_ct.advance(t_end)

    T_ct = reactor.T
    Y_ct = reactor.thermo.Y
    m_ct = reactor.mass

    # --- Canterax ---
    gas_tmp = ct.Solution(yaml_path)
    gas_tmp.TPX = T_reac0, P, X_reac0
    Y0 = jnp.array(gas_tmp.Y)
    m0 = float(m_ct)

    gas_tmp.TPX = T_in, P, X_in
    Yin = jnp.array(gas_tmp.Y)

    net_jt = OpenReactorNet(mech)
    res = net_jt.advance(
        T_reac0,
        P,
        Y0,
        m0,
        t_end,
        Tin=T_in,
        Yin=Yin,
        mdot_in=mdot,
        mdot_out=mdot,
        rtol=1e-9,
        atol=1e-14,
    )
    res = jax.block_until_ready(res)

    T_jt = float(res.ys[-1, 0])
    Y_jt = np.array(res.ys[-1, 1:-1])

    assert abs(T_jt - T_ct) < 2e-2  # K
    assert np.max(np.abs(Y_jt - Y_ct)) < 5e-8


def test_mass_flow_controller_rhs_jittable_single_trace():
    """Ensure the open RHS traces and compiles as one XLA computation."""
    yaml_path = "gri30.yaml"
    mech = _disable_reactions_canterax(load_mechanism(yaml_path))

    P = 101325.0
    gas = ct.Solution(yaml_path)
    gas.TPX = 1000.0, P, "N2:1.0"
    Y = jnp.array(gas.Y)

    state = jnp.concatenate([jnp.array([1000.0]), Y, jnp.array([1.0])])
    args = (P, mech, jnp.array([0.01, 0.0]), 900.0, Y, jnp.array([0.01, 0.0]))

    f = jax.jit(open_constp_reactor_rhs)
    out = f(0.0, state, args)
    jax.block_until_ready(out)

    lowered = f.lower(0.0, state, args)
    hlo_obj = lowered.compiler_ir(dialect="hlo")
    hlo = hlo_obj.as_hlo_text() if hasattr(hlo_obj, "as_hlo_text") else str(hlo_obj)

    # Check that lowering produced an HLO module.
    assert "HloModule" in hlo


def test_mass_flow_controller_gradients_vs_finite_difference():
    """Compare jax.grad against finite differences for mdot sensitivity.

    We differentiate the RHS (not the full ODE solve) to avoid reverse-mode
    limitations through implicit solver control flow.
    """
    yaml_path = "gri30.yaml"
    mech = _disable_reactions_canterax(load_mechanism(yaml_path))
    P = 101325.0

    T_reac0 = 1100.0
    X_reac0 = "N2:1.0"
    T_in = 800.0
    X_in = "O2:0.21, N2:0.79"

    gas = ct.Solution(yaml_path)
    gas.TPX = T_reac0, P, X_reac0
    Y0 = jnp.array(gas.Y)
    gas.TPX = T_in, P, X_in
    Yin = jnp.array(gas.Y)

    m0 = 1.0
    t_end = 5e-4

    state0 = jnp.concatenate([jnp.array([T_reac0]), Y0, jnp.array([m0])])
    mdot_out = 0.02

    def dTdt_from_mdot(mdot_in):
        args = (
            P,
            mech,
            jnp.array([mdot_in, 0.0]),
            T_in,
            Yin,
            jnp.array([mdot_out, 0.0]),
        )
        return open_constp_reactor_rhs(0.0, state0, args)[0]

    g_ad = float(jax.grad(dTdt_from_mdot)(0.02))

    eps = 1e-6
    f_p = float(dTdt_from_mdot(0.02 + eps))
    f_m = float(dTdt_from_mdot(0.02 - eps))
    g_fd = (f_p - f_m) / (2 * eps)

    denom = max(abs(g_fd), 1e-12)
    assert abs(g_ad - g_fd) / denom < 5e-4

