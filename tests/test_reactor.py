import os
import sys

import cantera as ct
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.loader import load_mechanism
from canterax.reactor import ReactorNet


TRAJECTORY_CASES = [
    {
        "label": "gri30_methane",
        "mech": "gri30.yaml",
        "T0": 1200.0,
        "P": ct.one_atm,
        "X0": "CH4:1.0, O2:2.0, N2:7.52",
        "t_end": 1e-3,
        "sample_dt": 1e-6,
        "max_dT": 1.0,
    },
    {
        "label": "gri30_hydrogen",
        "mech": "gri30.yaml",
        "T0": 1200.0,
        "P": ct.one_atm,
        "X0": "H2:2.0, O2:1.0, N2:3.76",
        "t_end": 1e-3,
        "sample_dt": 2e-5,
        "max_dT": 1.0,
    },
    {
        "label": "jp10_oxidation",
        "mech": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jp10.yaml")),
        "T0": 1500.0,
        "P": ct.one_atm,
        "X0": "C10H16:1.0, O2:14.0, N2:52.64",
        "t_end": 1e-3,
        "sample_dt": 2e-5,
        "max_dT": 2.0,
    },
]


@pytest.mark.parametrize("case", TRAJECTORY_CASES, ids=[case["label"] for case in TRAJECTORY_CASES])
def test_reactor_trajectory_parity(case):
    mech = load_mechanism(case["mech"])
    sol_ct = ct.Solution(case["mech"])

    T0 = case["T0"]
    P = case["P"]
    X0 = case["X0"]
    t_end = case["t_end"]
    sample_dt = case["sample_dt"]

    sol_ct.TPX = T0, P, X0
    reactor = ct.IdealGasConstPressureReactor(sol_ct)
    sim = ct.ReactorNet([reactor])
    t_ct = np.arange(0.0, t_end, sample_dt)
    T_ct = []
    for target in t_ct:
        sim.advance(float(target))
        T_ct.append(reactor.T)
    T_ct = np.array(T_ct)

    sol_ct.TPX = T0, P, X0
    Y0 = jnp.array(sol_ct.Y)
    net = ReactorNet(mech)
    saveat = diffrax.SaveAt(ts=jnp.array(t_ct))
    res = net.advance(T0, P, Y0, t_end, saveat=saveat)

    t_jx = np.array(res.ts)
    T_jx = np.array(res.ys[:, 0])
    max_dT = np.max(np.abs(T_jx - T_ct))

    os.makedirs("tests/outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(t_ct * 1e3, T_ct, "b-", label="Cantera", linewidth=2)
    plt.plot(t_jx * 1e3, T_jx, "r--", label="Canterax", linewidth=2)
    plt.title(f"Reactor Temperature Trajectory ({case['label']})")
    plt.xlabel("Time [ms]")
    plt.ylabel("Temperature [K]")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"tests/outputs/{case['label']}_reactor_trajectory.png")
    plt.close()

    assert max_dT < case["max_dT"]
