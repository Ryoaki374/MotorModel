from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .core import eddy_loss_round_wire
from .models import CoilGeometry, Rotor, SimulationConfig


@dataclass(frozen=True)
class CylindricalHarmonicSpec:
    """Input spectrum for the fast 2D cylindrical harmonic scaling model.

    Bn0_peak_T gives radial/tangential field peak amplitudes at magnet radius
    for q=1,2,... odd PM harmonics n=(2q-1)*p.  gap_model="power" uses
    Bn(Rc)=Bn0*(Rm/Rc)**(n+1); gap_model="exp" uses Bn=Bn0*exp(-n*g/Rm).
    """

    magnet_radius: float
    Bn0_peak_T: tuple[float, ...]
    gap_model: str = "power"


def run_cylindrical_harmonic_model(
    coil: CoilGeometry,
    rotor: Rotor,
    config: SimulationConfig,
    spectrum: CylindricalHarmonicSpec,
) -> dict[str, Any]:
    """Fast Model A: 2D cylindrical PM harmonic eddy-loss estimate.

    This model ignores axial end effects and treats the PM field as a periodic
    cylindrical harmonic spectrum.  It is intended for scaling checks rather
    than absolute accuracy.
    """
    max_h = config.max_harmonic or len(spectrum.Bn0_peak_T)
    h = np.arange(1, min(max_h, len(spectrum.Bn0_peak_T)) + 1, dtype=int)
    spatial_n = (2 * h - 1) * rotor.pole_pairs
    gap = coil.radius - spectrum.magnet_radius
    B0 = np.asarray(spectrum.Bn0_peak_T[: len(h)], dtype=float)
    if spectrum.gap_model == "power":
        decay = (spectrum.magnet_radius / coil.radius) ** (spatial_n + 1)
    elif spectrum.gap_model == "exp":
        decay = np.exp(-spatial_n * gap / spectrum.magnet_radius)
    else:
        raise ValueError('gap_model must be "power" or "exp"')
    Bpk = B0 * decay

    # Active-side-only 2D estimate: both axial sides see transverse radial/tangential field.
    active_length = 2.0 * coil.axial_length
    # Approximate the line integral by one point per harmonic with total dl.
    Bpk2 = (Bpk**2)[:, None]
    dl = np.array([active_length])
    # h in eddy_loss_round_wire is electrical harmonic; PM odd spatial sequence maps to h=1,3,5...
    electrical_h = (2 * h - 1).astype(int)
    P_total, P_by_h, _ = eddy_loss_round_wire(Bpk2, dl, coil, rotor, electrical_h)
    return {
        "P_total_W": P_total,
        "P_by_harmonic_W": pd.DataFrame({
            "electrical_harmonic": electrical_h,
            "spatial_order": spatial_n,
            "Bpk_T_at_coil": Bpk,
            "P_W": P_by_h,
        }),
        "diagnostics": {
            "model": "2D cylindrical harmonic",
            "gap_m": gap,
            "gap_model": spectrum.gap_model,
            "assumptions": [
                "axial end effects ignored",
                "end turns ignored",
                "absolute field amplitudes depend on supplied Bn0_peak_T",
            ],
        },
    }
