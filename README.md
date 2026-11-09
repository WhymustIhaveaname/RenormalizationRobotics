# Renormalization Robotics: Experiments

Experiments for *Effective Parameters, Real Behavior: Renormalization for Robotics — From Infinite Electron Mass to Sim-to-Real Gap* ([arXiv:2607.24079](https://arxiv.org/abs/2607.24079)).

The paper argues that simulator parameters should be treated as effective, resolution-dependent quantities $\theta^\star_{\mathrm{sim}}(a)$ rather than copies of measured values.
Its analytic example is PD control at a finite simulation frequency, where a small effective delay $\delta t \propto 1/f$ shifts the parameters that make the simulator reproduce the real motion:

$$
K_D^{\mathrm{sim}} = K_D^{\mathrm{real}} + \delta t\,K_P^{\mathrm{real}},
\qquad
J^{\mathrm{sim}} = J^{\mathrm{measured}} + \delta t\,K_D^{\mathrm{real}}.
$$

## Experiment 1: PD-controlled pendulum at finite simulation frequency

![Result](exp1_pendulum/results/fig_pendulum_mujoco.png)

**(a) The measured parameters are the wrong ones to use.** At the legged_gym default of 200 Hz physics, a simulator given the rod's measured inertia and the robot's actual gains swings up to 0.05 rad away from the real pendulum on the 2 Hz command. The paper's first-order prediction, which needs no real-world data at all, stays on it. Pooled over all five commands the RMSE goes from 2.64e−2 rad to 7.30e−4 rad, with one parameter set used for every command.

**(b) The same holds across simulation frequencies, and the measured parameters eventually fail outright.**

- The renormalized parameters give a **36× smaller** error at 200 Hz, with no fitting. Measured parameters only match it at $f$ = 7.2 kHz, so the same accuracy costs 36× fewer physics steps.
- Fitting on top of the formula gains only another 1.50×, and the fitted parameters differ from the predicted ones by about 1% ($J^\star/J$ = 1.369 against 1.356 at 200 Hz). That residual is the second-order term in $h$, not an artifact of fitting: identifying on one command or on all five gives the same parameters to four digits.
- At 50 Hz the measured parameters diverge, while $J^\star = 2.42J$ keeps the same simulation stable and accurate to 1.10e−2 rad. Inflating the inertia both absorbs the omitted sub-step dynamics and raises the stability limit.

The panel shows 50–1000 Hz, the range a robot simulator actually runs at; the sweep itself goes to 5 kHz.

**The parameters that reproduce the real motion flow linearly with the time step, as predicted.** Fitting $(J, K_P, K_D)$ at each resolution and taking the slope over $h \leq 2.5$ ms:

| | identified slope | predicted |
|---|---|---|
| $J^\star/J$ | 0.0725 per ms | 0.0712 ($K_D/2J$) |
| $K_D^\star/K_D$ | 0.0202 per ms | 0.0200 ($K_P/2K_D$) |

$K_P^\star$ stays within 0.07% of $K_P$ for $f \geq 200$ Hz, as predicted.
At 200 Hz the inertia that works is 1.37 times the measured one; at 50 Hz, 2.90 times.
The identified parameters leave these straight lines once $hK_D/J$ exceeds 1, which is where the first-order expansion stops being valid.

### System

| Quantity | Value |
|---|---|
| Pendulum | uniform rod, hinge at one end, gravity on |
| Mass $m$ | 1 lb = 0.4536 kg |
| Length $l$ | 1 ft = 0.3048 m |
| Inertia about hinge $J = ml^2/3$ | 0.01405 kg m² |
| Max gravity torque $mgl/2$ | 0.678 N m |
| $K_P$, $K_D$ | 80 N m/rad, 2 N m s/rad (ANYmal-C in [legged_gym](https://github.com/leggedrobotics/legged_gym)) |

Derived scales: $\omega_n = \sqrt{K_P/J} = 75.5$ rad/s (12.0 Hz), damping ratio $\zeta = 0.94$.
The expansion parameter is $hK_D/J$, which equals 0.71 at the legged_gym default $f = 200$ Hz and reaches 1 at $f = 142$ Hz.

### Controller and command

Torque $\tau = K_P(\theta_{\mathrm{ref}} - \theta) - K_D\dot\theta$, as in legged_gym (zero target velocity).
The reference is $\theta_{\mathrm{ref}}(t) = A\sin(2\pi f_{\mathrm{drive}} t)$ at five drive frequencies, $f_{\mathrm{drive}}$ = 0.5, 1, 2, 4, 8 Hz, each for 4 s.
$A = \pi/2$ up to 2 Hz and falls as $1/f_{\mathrm{drive}}$ above it, so the peak command velocity stays at 19.7 rad/s, as a real joint would.
The amplitude is large, so gravity acts nonlinearly throughout.
A single parameter set must serve all five commands, so identification cannot specialize to one of them.
The control frequency is fixed at 50 Hz and only the decimation varies, so every resolution sees the same held command and the same real trajectory; $h = 1/(50 \times \text{decimation})$ runs from 20 ms down to 0.2 ms.

### Ground truth ("real")

On a real robot the actuator PD loop runs at kHz rates, so the real system is continuous-time PD under the held command:

$$
J\ddot\theta = -\tfrac{1}{2}mgl\sin\theta + K_P(\theta_{\mathrm{ref},k} - \theta) - K_D\dot\theta,
\qquad t \in [t_k, t_{k+1}).
$$

It is solved interval by interval with DOP853 at `rtol = atol = 1e-12`.

### Simulator

MuJoCo with the `RK4` integrator; the PD torque is computed in Python from the current state at the start of each physics step and held for the whole step.
Holding the torque delays both feedback terms by half a step on average, which is exactly the paper's model with $\delta t = h/2$, so the predictions are

$$
J^\star = J + \tfrac{h}{2}K_D, \qquad K_D^\star = K_D + \tfrac{h}{2}K_P, \qquad K_P^\star = K_P.
$$

(`common.py` also implements MuJoCo's `Euler` and `implicitfast` schemes, which shift the bare parameters differently; they are not reported here.)

### System identification

For each resolution, find the one $(J, K_P, K_D)$ minimizing the RMSE between the simulated and real $\theta$, pooled over all five commands, on the simulator's time grid.
Gravity breaks the overall scale symmetry $(J, K_P, K_D) \to \lambda(J, K_P, K_D)$, so all three are identifiable.

- `scipy.optimize.least_squares` on log-parameters, with the trajectory residual as the residual vector.
- Continuation from fine to coarse steps, plus restarts from the measured and predicted parameters.
- Diverged rollouts receive a large constant residual.

### Checks

- MuJoCo's hinge inertia and gravity torque match the analytic values to 10 digits, and a native PD actuator integrated with RK4 at 20 kHz reproduces the ground truth to $1.3\times10^{-12}$ rad (`sanity_check.py`).
- The effective parameters depend on the physics resolution alone, not on how the command was generated: identified on the 2 Hz command across five control frequencies from 10 to 200 Hz, they agree within 0.03% at $h$ = 1 ms and 0.2% at $h$ = 2.5 ms (`run_flow.py`).
- The step size at which measured parameters match the renormalized accuracy is measured, not extrapolated: the sweep runs measured parameters down to $h$ = 0.04 ms (25 kHz).

The pendulum itself: [`exp1_pendulum/results/pendulum.mp4`](exp1_pendulum/results/pendulum.mp4) (ground truth tracking the held command at each of the five drive frequencies in turn, half speed).

## Reproduce

```bash
uv sync
cd exp1_pendulum
uv run python sanity_check.py     # MuJoCo model vs ground-truth ODE
uv run python run_gap.py          # error of each parameter set across resolutions
uv run python plot_pendulum.py    # the figure
uv run python run_flow.py         # control-frequency check
uv run python video.py            # pendulum video (needs EGL)
```

## Roadmap

Each step is one commit, reviewed before the next begins.

- [x] Plan
- [x] MuJoCo: flow, gap, and the pendulum video
- [ ] MuJoCo: validation (parameters fitted at one condition, tested at others)
- [ ] Isaac: the same experiment
