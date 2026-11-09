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

The physics frequency is $f = f_{\mathrm{ctrl}} \times \text{decimation}$, with both factors integers.
The physics time step $h = 1/f$ targets a uniform grid $h = 0.25, 0.5, \dots, 5$ ms ($f$ = 200–4000 Hz).
Each target is reached from five nominal control frequencies (10, 20, 50, 100, 200 Hz): the decimation is the integer nearest to $f/f_{\mathrm{ctrl}}$, and $f_{\mathrm{ctrl}}$ is then the integer (Hz) nearest to $f/\text{decimation}$, which keeps $h$ within 2.3% of the grid.
For example, $h = 5$ ms is $200 \times 1$, $50 \times 4$, or $10 \times 20$, and $h = 3.5$ ms is $286 \times 1$, $95 \times 3$, or $10 \times 29$; across the grid the nominal 50, 100, 200 Hz groups use 47–56, 89–125, 154–286 Hz.
If the effective parameters depend only on the physics resolution, points with the same $h$ should collapse across control frequencies.
The PD torque is recomputed at every physics step from the current state, as in legged_gym.

MuJoCo arms (the same pendulum, different discretization schemes):

| Arm | Integrator | PD implementation | Corresponds to |
|---|---|---|---|
| A | `Euler` (semi-implicit) | torque computed in Python each physics step | legged_gym-style explicit PD |
| B | `RK4` | torque computed in Python each physics step, held through RK stages | the paper's delay model with $\delta t = h/2$ |
| C | `implicitfast` | native `position` actuator ($k_p$, $k_v$) | implicit actuators, e.g. Isaac Lab `ImplicitActuator` |

First-order predictions (to be tested, not assumed):

| Arm | fitted | $J^\star$ | $K_D^\star$ | $K_P^\star$ | command delay $d^\star$ |
|---|---|---|---|---|---|
| A | $J, K_P, K_D, d$ | $J + \tfrac{h}{2}K_D$ | $K_D$ | $K_P$ | $h/2$ |
| B | $J, K_P, K_D$ | $J + \tfrac{h}{2}K_D$ | $K_D + \tfrac{h}{2}K_P$ | $K_P$ | — |
| C | $J, K_P, K_D, d$ | $J - \tfrac{h}{2}K_D$ | $K_D$ | $K_P$ | $h/2$ |
| A (3 parameters) | $J, K_P, K_D$ | $J + hK_D$ | $K_D + \tfrac{h}{2}K_P$ | $K_P$ | — |
| C (3 parameters) | $J, K_P, K_D$ | $J$ | $K_D + \tfrac{h}{2}K_P$ | $K_P$ | — |

- **Arm B** holds the torque over each physics step, which delays both feedback terms by $h/2$; this is the paper's formula with $\delta t = h/2$.
- **Arm A** is semi-implicit Euler, whose velocity is staggered half a step behind the position: velocity feedback lags by $h/2$ (so $J^\star = J + \tfrac{h}{2}K_D$, with no $K_P \to K_D$ leakage), and a held command takes effect $h/2$ early.
  The command advance is a time shift, which $(J, K_P, K_D)$ cannot represent; a command delay $d^\star = h/2$ cancels it.
  Without $d$, the fit absorbs the advance only approximately (last two rows), exactly for smooth commands and poorly for coarse staircase commands at low $f_{\mathrm{ctrl}}$.
- **Arm C** treats damping implicitly, which is exactly arm A with $J \to J + hK_D$.

The bare inertia is therefore scheme-dependent ($J + \tfrac{h}{2}K_D$, $J + \tfrac{h}{2}K_D$, $J - \tfrac{h}{2}K_D$) while the matched behavior is the same, like counterterms under different regularization schemes.

### System identification

For each arm and each $(f_{\mathrm{ctrl}}, \text{decimation})$, find $(J, K_P, K_D)$ minimizing the RMSE between simulated and real $\theta$ on the simulator's time grid.
Gravity ($mgl/2$ fixed) breaks the overall scale symmetry $(J, K_P, K_D) \to \lambda(J, K_P, K_D)$, so all three are identifiable in principle; the conditioning of the scale direction is reported alongside the results.

- Optimizer: `scipy.optimize.least_squares` on log-parameters, with the trajectory residual as the residual vector.
- Continuation from small to large $h$: the solution at one step size initializes the next, plus random restarts to check for local minima.
- Unstable rollouts (explicit PD with measured parameters becomes unstable below roughly 70 Hz) receive a large constant residual.
- Arms A and C also identify a command delay $d$: the held command is shifted by $d$, and each physics step uses its average over the step, so $d = 0$ reproduces the held command exactly.

### Figures

1. **Flow**: parameter shifts $\Delta J$, $\Delta K_D$, $\Delta K_P$ (and the command delay where fitted) versus $h = 1/f$, with first-order theory and linear fits, plus the post-fit RMSE.
2. **Gap**: trajectory RMSE versus $f$ for measured, analytically renormalized (no fitting), and identified parameters.
3. **Validation**: parameters identified at $f_{\mathrm{ref}} = 2$ Hz, tested on other reference frequencies and amplitudes.
4. **Trajectories at 50 Hz**: real versus simulated $\theta(t)$ with measured and identified parameters.

## Results (MuJoCo)

### Figure 1: flow

![Figure 1](exp1_pendulum/results/fig1_flow_mujoco.png)

Parameter rows: shifts $\Delta J$, $\Delta K_D$, $\Delta K_P$ and the command delay $d^\star$ (arms A, C) against $h$, pooled over all control frequencies, with the first-order theory (solid) and a fit (dashed): linear, or quadratic $c_0 + c_1 h + c_2 h^2$ where the theory slope is 0.
Last row: post-fit RMSE against $h^2$.

- **Every flow is linear in $h$ and starts with the predicted slope.** Linear fits over the whole range $h \leq 5$ ms, and over $h \leq 2$ ms where $hK_D/J \leq 0.28$:

  | | theory | fit, $h \leq 5$ ms | fit, $h \leq 2$ ms |
  |---|---|---|---|
  | B: $\Delta J$ ($10^{-3}$ kg m²/ms) | 1.00 | 1.04 ($R^2$ 0.9996) | 1.02 ($R^2$ 1.0000) |
  | B: $\Delta K_D$ ($10^{-2}$ N m s/rad/ms) | 4.00 | 4.13 ($R^2$ 0.996) | 4.03 ($R^2$ 0.9995) |
  | A: $\Delta J$ | 1.00 | 1.17 ($R^2$ 0.999) | 1.07 ($R^2$ 0.9998) |
  | A: $d^\star$ (ms/ms) | 0.50 | 0.44 ($R^2$ 0.999) | 0.47 ($R^2$ 0.9998) |
  | C: $\Delta J$ | −1.00 | −0.85 ($R^2$ 0.998) | −0.93 ($R^2$ 0.9998) |

  The slopes approach theory as the range shrinks; the remaining deviation is the curvature of the second-order terms.
- **Where the theory predicts no first-order flow, the quadratic fit finds none.** Fitting $c_0 + c_1 h + c_2 h^2$:

  | | $c_0$ | $c_1$ (per ms) | $c_2$ (per ms²) | $R^2$ |
  |---|---|---|---|---|
  | A, C: $\Delta K_D$ ($10^{-2}$ N m s/rad) | −0.0004 | 0.0007 | 0.0964 | 0.9999 |
  | A, C: $\Delta K_P$ (N m/rad) | 0.0001 | −0.0001 | 0.0006 | 0.574 |
  | B: $\Delta K_P$ (N m/rad) | 0.0152 | −0.0275 | 0.0111 | 0.183 |

  $\Delta K_D$ in A and C is a pure $h^2$ term (1.2% of $K_D$ at 5 ms). $\Delta K_P$ stays below 0.03% (A, C) and 0.6% (B) of $K_P$; its low $R^2$ reflects control frequencies drifting in opposite directions at large $h$, not a common trend.
- **All first-order effects are absorbed.** The post-fit RMSE grows as $h^2$, a straight line against $h^2$ in every column. In A and C the five control frequencies fall on one line; in B the slope is larger for coarser commands, and the nominal 100 and 200 Hz groups wobble because their actual $f_{\mathrm{ctrl}}$ changes along $h$ (89–125 and 154–286 Hz). With the command delay, A and C collapse across control frequencies, and at $h = 0.25$ ms the RMSE drops from $3\times10^{-5}$–$8\times10^{-4}$ rad (three parameters) to $2\times10^{-8}$ rad.
- **The extra parameter appears only where the theory says it should.** Fitting $d$ in arm B as a control gives $d^\star/h = -0.04, -0.011, -0.003$ at $h$ = 5, 1, 0.25 ms, vanishing as $h \to 0$.
- **Bare parameters are scheme-dependent.** A and C differ only by $J \to J + hK_D$, with identical $K_P^\star$, $K_D^\star$, $d^\star$, and RMSE.
- **Three parameters are not enough for A and C**: fitting only $(J, K_P, K_D)$, coarse commands pull $K_P^\star$ and $K_D^\star$ off the line ($\Delta K_D$ pooled $R^2 = 0.02$), and the RMSE is linear in $h$ instead of $h^2$. This is step 4 of the paper's procedure: the parameter set had to be expanded.
- Below 200 Hz ($h > 5$ ms) the first-order picture breaks down; this regime is left to Figure 2.

The pendulum itself: [`exp1_pendulum/results/pendulum.mp4`](exp1_pendulum/results/pendulum.mp4) (ground truth tracking the 50 Hz held command, half speed).

## Reproduce

```bash
uv sync
cd exp1_pendulum
uv run python sanity_check.py  # MuJoCo model vs ground-truth ODE
uv run python run_flow.py      # identification sweep, a few minutes on 15 cores
uv run python run_flow.py --delay  # same, also identifying the command delay
uv run python plot_flow.py     # Figure 1
uv run python video.py         # pendulum video (needs EGL)
```

## Roadmap

Each step is one commit, reviewed before the next begins.

- [x] Plan (this README)
- [x] MuJoCo: Figure 1 (flow) + pendulum video
- [ ] MuJoCo: Figure 2 (gap)
- [ ] MuJoCo: Figure 3 (validation)
- [ ] MuJoCo: Figure 4 (trajectories at 50 Hz)
- [ ] Isaac: Figures 1–4
