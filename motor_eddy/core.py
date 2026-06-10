from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

from .models import CoilGeometry, MagnetBlock, Rotor, SimulationConfig

MU0 = 4.0e-7 * np.pi
_EPS = 1.0e-30


def _er(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)


def _etheta(theta: float) -> np.ndarray:
    return np.array([-np.sin(theta), np.cos(theta), 0.0], dtype=float)


def _rot_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


@lru_cache(maxsize=64)
def _legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    xi, wi = leggauss(n)
    return 0.5 * (xi + 1.0), 0.5 * wi


def build_rectangular_coil_centerline(
    coil: CoilGeometry, config: SimulationConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return quadrature points, tangent vectors, dl weights, and segment labels
    for one turn of a rectangular coil on cylindrical surface r=coil.radius.

    The active sides are axial line segments.  The optional end turns are arcs
    on the same cylindrical radius.  All dimensions are SI units.
    """
    n = config.n_wire_quad_per_segment
    if n <= 0:
        raise ValueError("n_wire_quad_per_segment must be positive")
    u, w = _legendre_01(n)
    dtheta = coil.circumferential_width / coil.radius
    th1 = coil.theta0 - 0.5 * dtheta
    th2 = coil.theta0 + 0.5 * dtheta
    pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def add_axial(theta: float, z_start: float, z_end: float, label: str) -> None:
        z = z_start + (z_end - z_start) * u
        pts = np.column_stack(
            [coil.radius * np.cos(theta) * np.ones_like(z),
             coil.radius * np.sin(theta) * np.ones_like(z), z]
        )
        tangent_sign = np.sign(z_end - z_start) or 1.0
        tangents = np.tile(np.array([0.0, 0.0, tangent_sign]), (n, 1))
        dl = abs(z_end - z_start) * w
        labels = np.array([label] * n, dtype=object)
        pieces.append((pts, tangents, dl, labels))

    def add_arc(theta_start: float, theta_end: float, z: float, label: str) -> None:
        theta = theta_start + (theta_end - theta_start) * u
        pts = np.column_stack(
            [coil.radius * np.cos(theta), coil.radius * np.sin(theta), z * np.ones_like(theta)]
        )
        tangent_sign = np.sign(theta_end - theta_start) or 1.0
        tangents = tangent_sign * np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        dl = coil.radius * abs(theta_end - theta_start) * w
        labels = np.array([label] * n, dtype=object)
        pieces.append((pts, tangents, dl, labels))

    add_axial(th1, -0.5 * coil.axial_length, 0.5 * coil.axial_length, "active_side_1")
    add_axial(th2, 0.5 * coil.axial_length, -0.5 * coil.axial_length, "active_side_2")
    if config.use_end_turns:
        add_arc(th1, th2, 0.5 * coil.axial_length, "end_turn_top")
        add_arc(th2, th1, -0.5 * coil.axial_length, "end_turn_bottom")

    points = np.vstack([p[0] for p in pieces])
    tangents = np.vstack([p[1] for p in pieces])
    dl = np.concatenate([p[2] for p in pieces])
    labels = np.concatenate([p[3] for p in pieces])
    return points, tangents, dl, labels


def magnet_surface_quadrature(
    magnet: MagnetBlock, rotor_angle: float, n_quad: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return surface quadrature points x', outward normals n', area weights dS,
    and magnetization vector M for the rotated magnet.

    The finite magnet is approximated as a local rectangular prism with axes in
    the radial, tangential, and axial directions.  Magnetic surface charge is
    sigma_m = M dot n on all six prism faces.  No FEM mesh or PDE solver is used.
    """
    if n_quad <= 0:
        raise ValueError("n_quad must be positive")
    theta = magnet.theta0 + rotor_angle
    er = _er(theta)
    et = _etheta(theta)
    ez = np.array([0.0, 0.0, 1.0])
    center = magnet.radius * er
    dims = np.array([magnet.radial_thickness, magnet.tangential_width, magnet.axial_length], dtype=float)
    basis = np.vstack([er, et, ez])
    M = magnet.polarity * (magnet.Br / MU0) * er

    u, w = _legendre_01(n_quad)
    xi = (u - 0.5)
    wi = w
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    areas: list[np.ndarray] = []

    # face normal axis a; in-face axes b,c
    for a, b, c in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        for sign in (-1.0, 1.0):
            uu, vv = np.meshgrid(xi, xi, indexing="ij")
            ww = np.outer(wi, wi).ravel()
            local = np.zeros((n_quad * n_quad, 3), dtype=float)
            local[:, a] = sign * 0.5 * dims[a]
            local[:, b] = uu.ravel() * dims[b]
            local[:, c] = vv.ravel() * dims[c]
            pts = center + local @ basis
            normal = sign * basis[a]
            dS = dims[b] * dims[c] * ww
            points.append(pts)
            normals.append(np.tile(normal, (pts.shape[0], 1)))
            areas.append(dS)

    return np.vstack(points), np.vstack(normals), np.concatenate(areas), M


def magnetic_field_from_magnets(
    points: np.ndarray, rotor: Rotor, rotor_angle: float, config: SimulationConfig
) -> np.ndarray:
    """
    Compute B at requested points using magnetic surface-charge quadrature.
    This is a direct boundary integral over permanent-magnet faces; no FEM,
    spatial mesh, PDE solver, or measured back-EMF waveform is used.
    """
    obs = np.asarray(points, dtype=float)
    B = np.zeros_like(obs)
    for magnet in rotor.magnets:
        src, normals, dS, M = magnet_surface_quadrature(magnet, rotor_angle, config.n_surface_quad)
        sigma = normals @ M
        charge_weight = sigma * dS
        r = obs[:, None, :] - src[None, :, :]
        r2 = np.sum(r * r, axis=2)
        inv_r3 = 1.0 / np.maximum(r2, _EPS) ** 1.5
        B += (MU0 / (4.0 * np.pi)) * np.einsum("ps,psk,s->pk", inv_r3, r, charge_weight)
    return B


def compute_Bperp_time_series(
    coil: CoilGeometry, rotor: Rotor, config: SimulationConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each rotor angle sample, compute B_perp at all coil quadrature points.
    Returns Bperp[angle_index, point_index, 3], dl, labels, points, tangents.
    """
    points, tangents, dl, labels = build_rectangular_coil_centerline(coil, config)
    angles = np.linspace(0.0, 2.0 * np.pi, config.n_rotor_samples, endpoint=False)
    Bperp = np.empty((config.n_rotor_samples, points.shape[0], 3), dtype=float)
    for k, angle in enumerate(angles):
        B = magnetic_field_from_magnets(points, rotor, angle, config)
        Bpar = np.sum(B * tangents, axis=1)[:, None] * tangents
        Bperp[k] = B - Bpar
    return Bperp, dl, labels, points, tangents


def _selected_harmonics(rotor: Rotor, config: SimulationConfig) -> np.ndarray:
    nyquist_order = config.n_rotor_samples // 2
    max_h = nyquist_order // rotor.pole_pairs
    if config.max_harmonic is not None:
        max_h = min(max_h, config.max_harmonic)
    return np.arange(1, max_h + 1, dtype=int)


def fft_harmonics_over_rotor_angle(
    Bperp_series: np.ndarray, rotor: Rotor, config: SimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert rotor-angle samples to harmonic coefficients.

    Real Fourier normalization is
        x(psi) = a0 + sum_h a_h cos(h p psi) + b_h sin(h p psi).
    For amplitude_convention="peak", Bpk2[h, point] is the vector peak
    amplitude squared, sum_c (a_h,c**2 + b_h,c**2).  These peak amplitudes are
    compatible with the pi*d^4/(128*rho) round-wire coefficient used here.
    """
    if config.amplitude_convention != "peak":
        raise ValueError('Only amplitude_convention="peak" is currently implemented')
    n = Bperp_series.shape[0]
    coeff = np.fft.rfft(Bperp_series, axis=0) / n
    harmonics = _selected_harmonics(rotor, config)
    Bpk2 = np.zeros((len(harmonics), Bperp_series.shape[1]), dtype=float)
    for row, h in enumerate(harmonics):
        order = h * rotor.pole_pairs
        if order >= coeff.shape[0]:
            continue
        c = coeff[order]
        # For positive non-Nyquist bins, a=2Re(C), b=-2Im(C), so peak^2=4|C|^2.
        factor = 1.0 if (n % 2 == 0 and order == n // 2) else 4.0
        Bpk2[row] = factor * np.sum(np.abs(c) ** 2, axis=1)
    return Bpk2, harmonics


def eddy_loss_round_wire(
    Bpk2: np.ndarray, dl: np.ndarray, coil: CoilGeometry, rotor: Rotor, harmonics: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute open-circuit local eddy-current loss in a round wire.

    The implemented peak-amplitude formula is
        P = Nc * sum_i sum_h pi*d^4/(128*rho) * (h*p*Omega)^2
            * |B_perp,h(x_i)|_pk^2 * dl_i.
    The 1/128 coefficient assumes sinusoidal peak magnetic-flux-density
    amplitudes, not RMS phasors.  With RMS B, the coefficient would be doubled.
    """
    coefficient = np.pi * coil.wire_diameter**4 / (128.0 * coil.wire_resistivity)
    omega_h = harmonics.astype(float) * rotor.pole_pairs * rotor.omega
    per_h_point = coil.turns * coefficient * (omega_h[:, None] ** 2) * Bpk2 * dl[None, :]
    per_h = np.sum(per_h_point, axis=1)
    return float(np.sum(per_h)), per_h, per_h_point


def _minimum_point_to_magnet_surface_distance(points: np.ndarray, rotor: Rotor) -> float:
    min_dist = np.inf
    for m in rotor.magnets:
        theta = m.theta0
        R = _rot_z(theta)
        er = _er(theta)
        et = _etheta(theta)
        ez = np.array([0.0, 0.0, 1.0])
        basis = np.vstack([er, et, ez])
        center = m.radius * er
        local = (points - center) @ basis.T
        half = np.array([m.radial_thickness, m.tangential_width, m.axial_length]) * 0.5
        outside = np.maximum(np.abs(local) - half, 0.0)
        min_dist = min(min_dist, float(np.min(np.linalg.norm(outside, axis=1))))
    return min_dist


def _diagnostics(
    coil: CoilGeometry,
    rotor: Rotor,
    config: SimulationConfig,
    points: np.ndarray,
    labels: np.ndarray,
    Bperp_series: np.ndarray,
    harmonics: np.ndarray,
) -> dict[str, Any]:
    warnings: list[str] = []
    hmax = int(harmonics[-1]) if len(harmonics) else 0
    requested_hmax = int(config.max_harmonic) if config.max_harmonic is not None else hmax
    omega_max = hmax * rotor.pole_pairs * abs(rotor.omega)
    skin_depth = np.inf if omega_max == 0.0 else float(np.sqrt(2.0 * coil.wire_resistivity / (MU0 * omega_max)))
    d_over_delta = 0.0 if np.isinf(skin_depth) else float(coil.wire_diameter / skin_depth)
    if d_over_delta > 0.3:
        warnings.append(f"d/delta_min={d_over_delta:.3g} > 0.3; thin-wire low-frequency approximation may be inaccurate.")
    if requested_hmax and config.n_rotor_samples < 4 * requested_hmax + 1:
        warnings.append(
            f"n_rotor_samples={config.n_rotor_samples} is below recommended "
            f"4H+1={4*requested_hmax+1} for requested H={requested_hmax}."
        )
    if config.max_harmonic is not None and hmax < config.max_harmonic:
        warnings.append(
            f"max_harmonic={config.max_harmonic} exceeds FFT limit for "
            f"n_rotor_samples={config.n_rotor_samples} and p={rotor.pole_pairs}; using H={hmax}."
        )
    min_dist = _minimum_point_to_magnet_surface_distance(points, rotor) if len(rotor.magnets) else np.inf
    min_dim = min([coil.wire_diameter] + [min(m.radial_thickness, m.tangential_width, m.axial_length) for m in rotor.magnets])
    if min_dist < 0.25 * min_dim:
        warnings.append("Evaluation points are close to a magnet surface; refine surface quadrature and check singular behavior.")
    end_mask = np.char.startswith(labels.astype(str), "end_turn")
    if np.any(end_mask):
        end_rms = float(np.sqrt(np.mean(np.sum(Bperp_series[:, end_mask, :] ** 2, axis=2))))
        all_rms = float(np.sqrt(np.mean(np.sum(Bperp_series**2, axis=2))))
        if all_rms > 0 and end_rms > 0.5 * all_rms:
            warnings.append("End turns see strong field; 2D active-length-only models may underpredict loss.")
    if coil.turns > 1:
        warnings.append("turns > 1 is modeled as P_total = N_c P_one_turn; winding-pack offsets are ignored.")
    return {
        "skin_depth_min_m": skin_depth,
        "d_over_delta_max": d_over_delta,
        "n_rotor_samples": config.n_rotor_samples,
        "n_wire_points": int(points.shape[0]),
        "n_surface_quad": config.n_surface_quad,
        "min_point_to_magnet_surface_m": min_dist,
        "warnings": warnings,
    }


def run_eddy_loss_simulation(coil: CoilGeometry, rotor: Rotor, config: SimulationConfig) -> dict[str, Any]:
    """
    Full 3D magnetic-charge forward-model pipeline:
        build coil -> compute B_perp time series -> FFT -> compute P.

    Returns total loss, harmonic and segment breakdowns, B_perp RMS diagnostics,
    and warnings.  This function uses only magnet geometry, magnetization,
    rotation speed, coil geometry, wire diameter, and resistivity.
    """
    Bperp, dl, labels, points, tangents = compute_Bperp_time_series(coil, rotor, config)
    Bpk2, harmonics = fft_harmonics_over_rotor_angle(Bperp, rotor, config)
    P_total, P_by_h, P_h_point = eddy_loss_round_wire(Bpk2, dl, coil, rotor, harmonics)

    P_by_harmonic = pd.DataFrame({
        "harmonic": harmonics,
        "mechanical_order": harmonics * rotor.pole_pairs,
        "frequency_Hz": harmonics * rotor.pole_pairs * rotor.omega / (2.0 * np.pi),
        "P_W": P_by_h,
    })
    seg_rows = []
    rms_rows = []
    for label in pd.unique(labels):
        mask = labels == label
        seg_rows.append({"segment": label, "P_W": float(np.sum(P_h_point[:, mask]))})
        rms_rows.append({
            "segment": label,
            "Bperp_rms_T": float(np.sqrt(np.mean(np.sum(Bperp[:, mask, :] ** 2, axis=2))))
        })
    P_by_segment = pd.DataFrame(seg_rows)
    Bperp_rms_by_segment = pd.DataFrame(rms_rows)
    diagnostics = _diagnostics(coil, rotor, config, points, labels, Bperp, harmonics)
    return {
        "P_total_W": P_total,
        "P_by_harmonic_W": P_by_harmonic,
        "P_by_segment_W": P_by_segment,
        "Bperp_rms_by_segment_T": Bperp_rms_by_segment,
        "diagnostics": diagnostics,
        "Bperp_series_T": Bperp,
        "points_m": points,
        "tangents": tangents,
        "dl_m": dl,
        "segment_labels": labels,
        "harmonics": harmonics,
        "Bpk2_T2": Bpk2,
    }
