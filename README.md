# Renormalization Robotics: Experiments

Experiments for *Effective Parameters, Real Behavior: Renormalization for Robotics — From Infinite Electron Mass to Sim-to-Real Gap* ([arXiv:2607.24079](https://arxiv.org/abs/2607.24079)).

The paper argues that simulator parameters should be treated as effective, resolution-dependent quantities $\theta^\star_{\mathrm{sim}}(a)$ rather than copies of measured values.
Its analytic example is PD control at a finite simulation frequency, where a small effective delay $\delta t \propto 1/f$ shifts the parameters that make the simulator reproduce the real motion:

$$
K_D^{\mathrm{sim}} = K_D^{\mathrm{real}} + \delta t\,K_P^{\mathrm{real}},
\qquad
J^{\mathrm{sim}} = J^{\mathrm{measured}} + \delta t\,K_D^{\mathrm{real}}.
$$

The experiments below test this directly: identify the best simulator parameters at each simulation frequency and watch them flow.

## Experiment 1: PD-controlled pendulum at finite simulation frequency

### System

| Quantity | Value |
|---|---|
| Pendulum | uniform rod, hinge at one end, gravity on |
| Mass $m$ | 1 lb = 0.4536 kg |
| Length $l$ | 1 ft = 0.3048 m |
| Inertia about hinge $J = ml^2/3$ | 0.01405 kg m² |
| Max gravity torque $mgl/2$ | 0.678 N m |
| $K_P$, $K_D$ | 80 N m/rad, 2 N m s/rad (ANYmal-C in [legged_gym](https://github.com/leggedrobotics/legged_gym)) |

Derived scales: $\omega_n = \sqrt{K_P/J} = 75.5$ rad/s (12.0 Hz), damping ratio $\zeta = 0.94$, $K_D/J = 142\ \mathrm{s^{-1}}$.
The expansion parameter is $hK_D/J$ with $h = 1/f$; it equals 0.71 at the legged_gym default $f = 200$ Hz.

### Controller and command

Torque $\tau = K_P(\theta_{\mathrm{ref}} - \theta) - K_D\dot\theta$, as in legged_gym (zero target velocity).
The reference is $\theta_{\mathrm{ref}}(t) = A\sin(2\pi f_{\mathrm{ref}} t)$ with large amplitude $A = \pi/2$ and $f_{\mathrm{ref}} = 2$ Hz.
A policy issues commands at the control frequency $f_{\mathrm{ctrl}}$, so $\theta_{\mathrm{ref}}$ is sampled at $f_{\mathrm{ctrl}}$ and held between commands.
The command sequence is known exactly and is fed identically to the "real" system and to the simulator.

### Ground truth ("real")

On a real robot the actuator PD loop runs at kHz rates, so the real system is modeled as continuous-time PD under the held command:

$$
J\ddot\theta = -\tfrac{1}{2}mgl\sin\theta + K_P(\theta_{\mathrm{ref},k} - \theta) - K_D\dot\theta,
\qquad t \in [t_k, t_{k+1}).
$$

It is solved interval by interval with DOP853 at `rtol = atol = 1e-12`, which is exact for our purposes.

### Simulator

The physics frequency is $f = f_{\mathrm{ctrl}} \times \text{decimation}$ and spans 10 Hz to 4000 Hz.
The same $f$ is reached through several $(f_{\mathrm{ctrl}}, \text{decimation})$ pairs; if the effective parameters depend only on the physics resolution, points with equal $f$ should collapse.
The PD torque is recomputed at every physics step from the current state, as in legged_gym.

MuJoCo arms (the same pendulum, different discretization schemes):

| Arm | Integrator | PD implementation | Corresponds to |
|---|---|---|---|
| A | `Euler` (semi-implicit) | torque computed in Python each physics step | legged_gym-style explicit PD |
| B | `RK4` | torque computed in Python each physics step, held through RK stages | the paper's delay model with $\delta t = h/2$ |
| C | `implicitfast` | native `position` actuator ($k_p$, $k_v$) | implicit actuators, e.g. Isaac Lab `ImplicitActuator` |

First-order predictions from a modified-equation analysis of a damped oscillator (to be tested, not assumed):

| Arm | $J^\star$ | $K_D^\star$ | $K_P^\star$ |
|---|---|---|---|
| A | $J + \tfrac{h}{2}K_D$ | $K_D$ | $K_P$ |
| B | $J + \tfrac{h}{2}K_D$ | $K_D + \tfrac{h}{2}K_P$ | $K_P$ |
| C | $J - \tfrac{h}{2}K_D$ | $K_D$ | $K_P$ |

Arm B is the paper's formula with $\delta t = h/2$.
In arm A the $K_P \to K_D$ leakage is predicted to vanish at first order, and in arm C the inertia shift is predicted to change sign: the bare parameters are scheme-dependent, like counterterms under different regularization schemes.

### System identification

For each arm and each $(f_{\mathrm{ctrl}}, \text{decimation})$, find $(J, K_P, K_D)$ minimizing the RMSE between simulated and real $\theta$ on the simulator's time grid.
Gravity ($mgl/2$ fixed) breaks the overall scale symmetry $(J, K_P, K_D) \to \lambda(J, K_P, K_D)$, so all three are identifiable in principle; the conditioning of the scale direction is reported alongside the results.

- Optimizer: `scipy.optimize.least_squares` on log-parameters, with the trajectory residual as the residual vector.
- Continuation from high $f$ to low $f$: the solution at one frequency initializes the next lower one, plus random restarts to check for local minima.
- Unstable rollouts (explicit PD with measured parameters becomes unstable below roughly 70 Hz) receive a large constant residual.

### Figures

1. **Flow**: identified $J^\star/J$, $K_D^\star/K_D$, $K_P^\star/K_P$ versus $f$, with first-order predictions.
2. **Gap**: trajectory RMSE versus $f$ for measured, analytically renormalized (no fitting), and identified parameters.
3. **Validation**: parameters identified at $f_{\mathrm{ref}} = 2$ Hz, tested on other reference frequencies and amplitudes.
4. **Trajectories at 50 Hz**: real versus simulated $\theta(t)$ with measured and identified parameters.

## Roadmap

Each step is one commit, reviewed before the next begins.

- [x] Plan (this README)
- [ ] MuJoCo: Figure 1 (flow) + pendulum video
- [ ] MuJoCo: Figure 2 (gap)
- [ ] MuJoCo: Figure 3 (validation)
- [ ] MuJoCo: Figure 4 (trajectories at 50 Hz)
- [ ] Isaac: Figures 1–4
