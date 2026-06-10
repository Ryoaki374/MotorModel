# Coreless PM Motor Open-Circuit Winding Eddy Loss

This repository implements a bottom-up Python forward model for local eddy-current loss in round-wire, open-terminal coils of a coreless/yokeless radial-flux permanent-magnet motor.  It does **not** use FEM, mesh-based PDE solvers, or measured back-EMF waveforms.

## Implemented models

### Model B: 3D magnetic-charge integration

Permanent magnets are finite rectangular prisms in local radial/tangential/axial coordinates.  Their external field is computed by direct magnetic surface-charge quadrature:

\[
\mathbf B(\mathbf x)=\frac{\mu_0}{4\pi}\sum_m\int_{\partial V_m}
\sigma_m(\mathbf x')\frac{\mathbf x-\mathbf x'}{|\mathbf x-\mathbf x'|^3}\,dS',
\qquad \sigma_m=\mathbf M\cdot\mathbf n.
\]

The coil is a rectangular loop on a cylindrical surface.  The code samples the rotor angle, projects the magnetic field perpendicular to the local wire tangent, extracts electrical harmonics with an FFT, and evaluates

\[
P_\mathrm{eddy}=N_c\frac{\pi d^4}{128\rho}
\sum_i\sum_h (hp\Omega)^2 |\mathbf B_{\perp,h}(\mathbf x_i)|_\mathrm{pk}^2\Delta l_i.
\]

The coefficient uses `amplitude_convention="peak"`.

### Model A: 2D cylindrical harmonic estimate

A fast scaling model represents the PM field as cylindrical spatial harmonics \(n=(2q-1)p\) with either power-law or exponential gap decay.  This is useful for sensitivity checks, especially \(P\propto\Omega^2\), \(P\propto d^4\), \(P\propto\rho^{-1}\), and rapid high-harmonic decay with air gap.

## Repository layout

- `motor_eddy/models.py` - dataclasses for materials, magnets, rotor, coil, and simulation settings.
- `motor_eddy/core.py` - 3D magnetic-charge model, FFT harmonic extraction, loss calculation, diagnostics.
- `motor_eddy/harmonic2d.py` - 2D cylindrical harmonic comparison model.
- `examples/sample_case.py` - four-pole sample motor input.
- `scripts/run_sample.py` - generates sample numerical output and plots.
- `tests/test_eddy_loss.py` - validation tests requested in the specification.
- `model_notes.md` - assumptions, equations, warnings, and unresolved issues.

## Quick start

```bash
python -m pytest
python scripts/run_sample.py
```

Sample plots are written to `outputs/`:

- `P_vs_omega.png`
- `P_vs_d.png`
- `harmonic_breakdown.png`
- `segment_breakdown.png`

## Minimal API example

```python
from examples.sample_case import make_four_pole_sample
from motor_eddy import run_eddy_loss_simulation

coil, rotor, config = make_four_pole_sample()
result = run_eddy_loss_simulation(coil, rotor, config)
print(result["P_total_W"])
print(result["P_by_harmonic_W"])
print(result["P_by_segment_W"])
print(result["diagnostics"]["warnings"])
```

## Main limitations

- The round-wire loss expression assumes thin wire, low frequency, and negligible eddy-current reaction.
- Multi-turn coils are currently scaled as `P_total = N_c P_one_turn`; winding-pack offsets are not yet resolved turn-by-turn.
- The finite magnet is a local rectangular-prism approximation to a curved magnet.
- Surface quadrature is direct and robust but not accelerated; large sweeps may require caching, batching, or fast multipole/tree methods.
