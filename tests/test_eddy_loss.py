from __future__ import annotations

import numpy as np

from motor_eddy import make_four_pole_sample
from motor_eddy import CoilGeometry, Rotor, SimulationConfig
from motor_eddy.core import eddy_loss_round_wire, fft_harmonics_over_rotor_angle, run_eddy_loss_simulation
from motor_eddy.harmonic2d import CylindricalHarmonicSpec, run_cylindrical_harmonic_model


def replace_coil(coil: CoilGeometry, **kwargs) -> CoilGeometry:
    data = coil.__dict__.copy()
    data.update(kwargs)
    return CoilGeometry(**data)


def replace_rotor(rotor: Rotor, **kwargs) -> Rotor:
    data = rotor.__dict__.copy()
    data.update(kwargs)
    return Rotor(**data)


def fast_case():
    coil, rotor, config = make_four_pole_sample()
    config = SimulationConfig(
        n_rotor_samples=32,
        n_wire_quad_per_segment=3,
        n_surface_quad=2,
        max_harmonic=6,
        use_end_turns=True,
    )
    return coil, rotor, config


def test_speed_scaling_omega_squared():
    coil, rotor, config = fast_case()
    vals = []
    for omega in (100.0, 250.0, 500.0):
        P = run_eddy_loss_simulation(coil, replace_rotor(rotor, omega=omega), config)["P_total_W"]
        vals.append(P / omega**2)
    assert np.allclose(vals, vals[0], rtol=1e-12, atol=0.0)


def test_wire_diameter_scaling_d_fourth():
    coil, rotor, config = fast_case()
    vals = []
    for d in (0.3e-3, 0.5e-3, 0.8e-3):
        c = replace_coil(coil, wire_diameter=d)
        vals.append(run_eddy_loss_simulation(c, rotor, config)["P_total_W"] / d**4)
    assert np.allclose(vals, vals[0], rtol=1e-12, atol=0.0)


def test_resistivity_inverse_scaling():
    coil, rotor, config = fast_case()
    vals = []
    for rho in (1.2e-8, 1.724e-8, 3.0e-8):
        c = replace_coil(coil, wire_resistivity=rho)
        vals.append(run_eddy_loss_simulation(c, rotor, config)["P_total_W"] * rho)
    assert np.allclose(vals, vals[0], rtol=1e-12, atol=0.0)


def test_zero_speed_zero_loss():
    coil, rotor, config = fast_case()
    P = run_eddy_loss_simulation(coil, replace_rotor(rotor, omega=0.0), config)["P_total_W"]
    assert P == 0.0


def test_end_turns_change_loss_when_included():
    coil, rotor, config = fast_case()
    with_end = run_eddy_loss_simulation(coil, rotor, config)["P_total_W"]
    no_end = run_eddy_loss_simulation(
        coil,
        rotor,
        SimulationConfig(**{**config.__dict__, "use_end_turns": False}),
    )["P_total_W"]
    assert with_end > no_end


def test_higher_harmonics_drop_with_larger_gap_in_2d_model():
    coil, rotor, config = fast_case()
    spectrum = CylindricalHarmonicSpec(0.020, (0.4, 0.16, 0.08, 0.04, 0.02))
    near = run_cylindrical_harmonic_model(coil, rotor, config, spectrum)["P_by_harmonic_W"]
    far_coil = replace_coil(coil, radius=0.040)
    far = run_cylindrical_harmonic_model(far_coil, rotor, config, spectrum)["P_by_harmonic_W"]
    near_ratio = near.loc[near.index[-1], "P_W"] / near.loc[near.index[0], "P_W"]
    far_ratio = far.loc[far.index[-1], "P_W"] / far.loc[far.index[0], "P_W"]
    assert far_ratio < near_ratio


def test_uniform_sinusoid_benchmark():
    pole_pairs = 2
    omega = 123.0
    rotor = Rotor(magnets=[], pole_pairs=pole_pairs, omega=omega)
    coil = CoilGeometry(0.03, 0.0, 0.01, 0.2, 1, 0.7e-3, 1.724e-8)
    config = SimulationConfig(64, 1, 1, 1)
    B0 = 0.05
    total_length = 0.2
    psi = np.linspace(0.0, 2.0 * np.pi, config.n_rotor_samples, endpoint=False)
    B = np.zeros((len(psi), 1, 3))
    B[:, 0, 0] = B0 * np.cos(pole_pairs * psi)
    Bpk2, harmonics = fft_harmonics_over_rotor_angle(B, rotor, config)
    P, _, _ = eddy_loss_round_wire(Bpk2, np.array([total_length]), coil, rotor, harmonics)
    expected = np.pi * coil.wire_diameter**4 * total_length / (128.0 * coil.wire_resistivity) * (pole_pairs * omega) ** 2 * B0**2
    assert np.isclose(P, expected, rtol=1e-13)
