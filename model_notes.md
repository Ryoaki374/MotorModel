# Model notes and assumptions

## Goal

Estimate open-circuit local eddy-current loss in a single round-wire winding of a coreless/yokeless radial-flux PM motor directly from geometry, magnetization, speed, wire diameter, and resistivity.  The implementation intentionally avoids FEM, mesh-based PDE solvers, and measured back-EMF input.

## Relation to reviewed analytical PM-machine papers

The reviewed notes in `main.tex` summarize a vector-potential harmonic workflow in which geometry and excitation define \(A_z(r,\theta)\), and eddy current follows from a time derivative of vector potential plus a zero-net-current correction.  The present implementation keeps the same forward-model spirit but removes slots, teeth, iron, armature reaction, and load current.  Instead, it computes magnet field directly in free space and applies the low-frequency round-wire loss formula.

## Coordinates

Global Cartesian coordinates are used:

\[
x=r\cos\theta,\qquad y=r\sin\theta,\qquad z=z.
\]

The rotor rotates about the \(z\)-axis by mechanical angle \(\psi\).

## 3D magnetic-charge PM field model

Each magnet is a finite local rectangular prism with dimensions

- radial thickness \(t_m\),
- tangential width \(w_m\),
- axial length \(L_m\).

At magnet center angle \(\theta_m+\psi\), the local basis is

\[
\hat{\mathbf e}_r=(\cos\theta,\sin\theta,0),\quad
\hat{\mathbf e}_\theta=(-\sin\theta,\cos\theta,0),\quad
\hat{\mathbf e}_z=(0,0,1).
\]

Magnetization is

\[
\mathbf M=s_m\frac{B_r}{\mu_0}\hat{\mathbf e}_r,
\qquad s_m\in\{+1,-1\}.
\]

The field outside magnets is evaluated by surface magnetic charge:

\[
\phi_m(\mathbf x)=\frac{1}{4\pi}\int_{\partial V_m}\frac{\sigma_m(\mathbf x')}{|\mathbf x-\mathbf x'|}\,dS',
\qquad
\sigma_m=\mathbf M\cdot\mathbf n.
\]

Using \(\mathbf H=-\nabla\phi_m\) with the sign convention implemented in the code gives the direct field integral

\[
\mathbf B(\mathbf x)=\frac{\mu_0}{4\pi}\sum_m\int_{\partial V_m}
\sigma_m(\mathbf x')\frac{\mathbf x-\mathbf x'}{|\mathbf x-\mathbf x'|^3}\,dS'.
\]

The six prism faces are integrated with tensor-product Gauss-Legendre quadrature.  No volume mesh is generated.

## Coil centerline model

One turn is represented by four centerline segments on cylinder radius \(R_c\):

1. active side 1 at \(\theta_0-\Delta\theta_c/2\),
2. active side 2 at \(\theta_0+\Delta\theta_c/2\),
3. top end turn at \(z=+L_c/2\),
4. bottom end turn at \(z=-L_c/2\),

where \(\Delta\theta_c=w_c/R_c\).  Segment line integrals use Gauss-Legendre quadrature.  `use_end_turns=False` omits the two end turns for active-length-only comparisons.

## Eddy-current loss in round wire

The open terminal condition eliminates net terminal current but not local cross-sectional eddy current.  The model assumes

\[
d\ll\delta,
\qquad
\delta=\sqrt{\frac{2\rho}{\mu_0\omega_{\max}}},
\]

and neglects eddy-current reaction on the applied PM field.

Only field perpendicular to the wire tangent contributes:

\[
\mathbf B_\perp=\mathbf B-(\mathbf B\cdot\hat{\mathbf t})\hat{\mathbf t}.
\]

Rotor-angle samples are Fourier transformed using

\[
\mathbf B_\perp(\psi)=\mathbf a_0+
\sum_{h=1}^{H}\left[\mathbf a_h\cos(hp\psi)+\mathbf b_h\sin(hp\psi)\right].
\]

For `amplitude_convention="peak"`,

\[
|\mathbf B_{\perp,h}|_\mathrm{pk}^2=
\sum_{c=x,y,z}(a_{h,c}^2+b_{h,c}^2).
\]

The discrete one-turn-plus-turn-multiplier loss is

\[
P_\mathrm{eddy}=N_c\frac{\pi d^4}{128\rho}
\sum_{i}\sum_{h}(hp\Omega)^2
|\mathbf B_{\perp,h}(\mathbf x_i)|_\mathrm{pk}^2\Delta l_i.
\]

The coefficient \(1/128\) is for peak sinusoidal magnetic flux density.  If RMS phasors are used, the coefficient must be adjusted.

## 2D cylindrical harmonic model

The comparison model ignores axial variation and end turns.  PM harmonics use

\[
n=(2q-1)p,
\]

with either

\[
B_n(R_c)=B_{n,0}\left(\frac{R_m}{R_c}\right)^{n+1}
\]

or

\[
B_n(g)=B_{n,0}\exp(-ng/R_m).
\]

This model is meant for scaling and gap sensitivity, not final absolute loss prediction.

## Diagnostics and warnings

The simulation reports warnings for:

1. \(d/\delta_\min>0.3\), where the thin-wire approximation may be poor.
2. insufficient rotor samples compared with the requested harmonic count.
3. evaluation points too close to a magnet surface.
4. strong PM field on end turns, indicating that 2D active-length-only models may underpredict loss.
5. `turns > 1`, because turn packing is represented by a simple multiplier.

## Unresolved implementation issues

- Add explicit per-turn offsets in radius, angle, and axial position for real winding packs.
- Add curved magnet geometry or segmented curved-surface quadrature instead of a local rectangular prism.
- Add magnet relative-permeability corrections beyond the current \(M\approx B_r/\mu_0\) assumption.
- Add adaptive quadrature near magnet edges and faces.
- Add field/loss caching for sweeps over speed, wire diameter, and resistivity.
- Validate against controlled analytical magnet benchmarks and independent measurements when available.
