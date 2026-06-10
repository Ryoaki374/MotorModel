from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.sample_case import make_four_pole_sample
from motor_eddy import CoilGeometry, Rotor, run_eddy_loss_simulation


def replace_coil(coil: CoilGeometry, **kwargs) -> CoilGeometry:
    data = coil.__dict__.copy()
    data.update(kwargs)
    return CoilGeometry(**data)


def replace_rotor(rotor: Rotor, **kwargs) -> Rotor:
    data = rotor.__dict__.copy()
    data.update(kwargs)
    return Rotor(**data)


def main() -> None:
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    coil, rotor, config = make_four_pole_sample()
    result = run_eddy_loss_simulation(coil, rotor, config)
    print(f"P_total_W = {result['P_total_W']:.6g}")
    print(result["P_by_harmonic_W"].head())
    print(result["P_by_segment_W"])
    for w in result["diagnostics"]["warnings"]:
        print("WARNING:", w)

    omegas = np.array([100, 200, 400, 800], dtype=float) * 2 * np.pi
    P_omega = [run_eddy_loss_simulation(coil, replace_rotor(rotor, omega=o), config)["P_total_W"] for o in omegas]
    plt.figure()
    plt.loglog(omegas, P_omega, "o-")
    plt.xlabel("Mechanical speed Ω [rad/s]")
    plt.ylabel("Eddy loss P [W]")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(out / "P_vs_omega.png", dpi=160)

    ds = np.array([0.3, 0.4, 0.5, 0.7], dtype=float) * 1e-3
    P_d = [run_eddy_loss_simulation(replace_coil(coil, wire_diameter=d), rotor, config)["P_total_W"] for d in ds]
    plt.figure()
    plt.loglog(ds, P_d, "o-")
    plt.xlabel("Wire diameter d [m]")
    plt.ylabel("Eddy loss P [W]")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(out / "P_vs_d.png", dpi=160)

    plt.figure()
    hb = result["P_by_harmonic_W"]
    plt.bar(hb["harmonic"], hb["P_W"])
    plt.xlabel("Electrical harmonic h")
    plt.ylabel("Loss contribution [W]")
    plt.tight_layout()
    plt.savefig(out / "harmonic_breakdown.png", dpi=160)

    plt.figure(figsize=(7, 3))
    sb = result["P_by_segment_W"]
    plt.bar(sb["segment"], sb["P_W"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Loss contribution [W]")
    plt.tight_layout()
    plt.savefig(out / "segment_breakdown.png", dpi=160)


if __name__ == "__main__":
    main()
