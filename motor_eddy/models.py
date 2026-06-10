from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    Br: float          # remanence [T]
    mu_r: float        # relative permeability [-]
    rho: float         # conductor resistivity [ohm m]


@dataclass(frozen=True)
class MagnetBlock:
    theta0: float      # initial center angle [rad]
    radius: float      # center radius [m]
    radial_thickness: float
    tangential_width: float
    axial_length: float
    polarity: int      # +1 for N, -1 for S
    Br: float
    mu_r: float = 1.05


@dataclass(frozen=True)
class Rotor:
    magnets: list[MagnetBlock]
    pole_pairs: int
    omega: float       # mechanical angular speed [rad/s]


@dataclass(frozen=True)
class CoilGeometry:
    radius: float
    theta0: float
    circumferential_width: float
    axial_length: float
    turns: int
    wire_diameter: float
    wire_resistivity: float


@dataclass(frozen=True)
class SimulationConfig:
    n_rotor_samples: int
    n_wire_quad_per_segment: int
    n_surface_quad: int
    max_harmonic: int | None
    use_end_turns: bool = True
    amplitude_convention: str = "peak"
