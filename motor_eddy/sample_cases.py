from __future__ import annotations

import numpy as np

from .models import CoilGeometry, MagnetBlock, Rotor, SimulationConfig


def make_four_pole_sample(omega: float = 2 * np.pi * 100.0) -> tuple[CoilGeometry, Rotor, SimulationConfig]:
    """Return a compact four-pole sample case for tests and notebooks."""
    magnets = []
    radius = 0.020
    for i in range(4):
        magnets.append(
            MagnetBlock(
                theta0=i * np.pi / 2,
                radius=radius,
                radial_thickness=0.004,
                tangential_width=0.016,
                axial_length=0.020,
                polarity=1 if i % 2 == 0 else -1,
                Br=1.2,
                mu_r=1.05,
            )
        )
    rotor = Rotor(magnets=magnets, pole_pairs=2, omega=omega)
    coil = CoilGeometry(
        radius=0.028,
        theta0=0.0,
        circumferential_width=0.018,
        axial_length=0.026,
        turns=12,
        wire_diameter=0.0005,
        wire_resistivity=1.724e-8,
    )
    config = SimulationConfig(
        n_rotor_samples=64,
        n_wire_quad_per_segment=8,
        n_surface_quad=4,
        max_harmonic=12,
        use_end_turns=True,
        amplitude_convention="peak",
    )
    return coil, rotor, config
