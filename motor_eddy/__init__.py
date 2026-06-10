"""Coreless/yokeless PM motor open-circuit winding eddy-current loss models."""
from .models import Material, MagnetBlock, Rotor, CoilGeometry, SimulationConfig
from .core import (
    build_rectangular_coil_centerline,
    magnet_surface_quadrature,
    magnetic_field_from_magnets,
    compute_Bperp_time_series,
    fft_harmonics_over_rotor_angle,
    eddy_loss_round_wire,
    run_eddy_loss_simulation,
)
from .harmonic2d import CylindricalHarmonicSpec, run_cylindrical_harmonic_model

__all__ = [
    "Material",
    "MagnetBlock",
    "Rotor",
    "CoilGeometry",
    "SimulationConfig",
    "build_rectangular_coil_centerline",
    "magnet_surface_quadrature",
    "magnetic_field_from_magnets",
    "compute_Bperp_time_series",
    "fft_harmonics_over_rotor_angle",
    "eddy_loss_round_wire",
    "run_eddy_loss_simulation",
    "CylindricalHarmonicSpec",
    "run_cylindrical_harmonic_model",
]
