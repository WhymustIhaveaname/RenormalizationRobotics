# Renormalization Robotics: Experiments

Experiments for *Effective Parameters, Real Behavior: Renormalization for Robotics — From Infinite Electron Mass to Sim-to-Real Gap* ([arXiv:2607.24079](https://arxiv.org/abs/2607.24079)).

The paper argues that simulator parameters should be treated as effective, resolution-dependent quantities $\theta^\star_{\mathrm{sim}}(a)$ rather than copies of measured values.
Its analytic example is PD control at a finite simulation frequency, where a small effective delay $\delta t \propto 1/f$ shifts the parameters that make the simulator reproduce the real motion:

$$
K_D^{\mathrm{sim}} = K_D^{\mathrm{real}} + \delta t \cdot K_P^{\mathrm{real}},
\qquad
J^{\mathrm{sim}} = J^{\mathrm{measured}} + \delta t \cdot K_D^{\mathrm{real}}.
$$

## Experiment 1: PD-controlled pendulum at finite simulation frequency

![Result](exp1_pendulum/results/fig_pendulum_mujoco.png)

**(a) The measured parameters are the wrong ones to use.** At the legged_gym default of 200 Hz physics, a simulator given the rod's measured inertia and the robot's actual gains swings up to 0.05 rad away from the real pendulum on the 2 Hz command. The paper's first-order prediction, which needs no real-world data at all, stays on it. Pooled over all five commands the RMSE goes from 2.64e−2 rad to 7.30e−4 rad, with one parameter set used for every command.

**(b) The same holds across simulation frequencies, and the measured parameters eventually fail outright.**

- The renormalized parameters give a **36× smaller** error at 200 Hz, with no fitting. Measured parameters only match it at $f$ = 7.2 kHz, so the same accuracy costs 36× fewer physics steps.
- Fitting on top of the formula gains only another 1.50×, and the fitted parameters differ from the predicted ones by about 1% ($J^\star/J$ = 1.369 against 1.356 at 200 Hz). That residual is the second-order term in $h$, not an artifact of fitting: identifying on one command or on all five gives the same parameters to four digits.
- At 50 Hz the measured parameters diverge, while $J^\star = 2.42J$ keeps the same simulation stable and accurate to 1.10e−2 rad. Inflating the inertia both absorbs the omitted sub-step dynamics and raises the stability limit.

The panel shows 50–1000 Hz, the range a robot simulator actually runs at; the sweep itself goes to 5 kHz.

### Flow of the effective parameters

![Flow](exp1_pendulum/results/fig1_flow_mujoco.png)

Identified parameters against the physics time step, for the three MuJoCo schemes, on a nearly uniform grid $h$ = 0.25 to 5 ms reached with integer control frequency and decimation. Shifts are shown as fractions of the measured value, so the scale is the same in any engine. Arms A and C identify a command delay as well; without it they cannot represent the half-step command advance of semi-implicit Euler.

Each panel that the theory predicts a flow for is fitted as $c_1 h + c_2 h^2$ through the origin, so the first-order slope $c_1$ can be read against the theory on its own. **Fitting a straight line instead is what makes the theory look wrong**: the raw linear slope of $\Delta J/J$ in arm A is 0.0834, 17% above prediction, purely because a curve is being measured with a ruler.

| | $c_1$ identified | $c_1$ predicted | error |
|---|---|---|---|
| A: $\Delta J/J$ | 0.0708 per ms | 0.0712 ($K_D/2J$) | −0.5% |
| B: $\Delta J/J$ | 0.0709 per ms | 0.0712 | −0.4% |
| C: $\Delta J/J$ | −0.0708 per ms | −0.0712 | −0.5% |
| B: $\Delta K_D/K_D$ | 0.0198 per ms | 0.0200 ($K_P/2K_D$) | −1% |
| A, C: command delay | 0.50 per ms | 0.50 ($h/2$) | −0.03% |
| A, C: $\Delta K_D/K_D$ | 1.8e−6 per ms | 0 | — |

The bare inertia is scheme-dependent — $J + hK_D/2$ for both explicit-PD schemes (Euler and RK4) but $J - hK_D/2$ for implicitfast — while the matched behaviour is the same, like counterterms under different regularization schemes.

**Second order.** Backward error analysis of the step map gives the continuous system each integrator is really solving. For semi-implicit Euler with an explicit PD torque (A) and for implicit damping (C), which is the same map with $J \to J + hK_D$:

$$J_e = J \mp \frac{h K_D}{2} - h^2 \frac{J K_P + K_D^2}{12 J}, \qquad K_{P,e} = K_P, \qquad K_{D,e} = K_D - h^2 \frac{K_P K_D}{12 J}$$

so the identified damping must curve upward as $\Delta K_D = + h^2 K_P K_D / 12 J$. The $K_D$ row of A and C draws that parabola: predicted $c_2$ = 4.75e−4 per ms², identified 4.82e−4, a 1.6% error with no fitted scale. Feeding these effective parameters to an exact integrator drops the error from $O(h^2)$ to $O(h^3)$, confirming the derivation.

What is left over at $O(h^2)$ after projecting onto the three tunable operators is $-c\left[\cos\theta \left(K_P(\theta - r) + c\sin\theta\right) + J\dot\theta^2\sin\theta\right] / 12 J^2$, proportional to the gravity torque $c = mgl/2$. **A linear plant stays exactly renormalizable at second order**; only the gravitational nonlinearity obstructs it.

**$K_P$ does not flow, and its row is where the obstruction lands.** The derivation gives $K_{P,e} = K_P$ exactly to second order, so the $K_P$ row is drawn unfitted, on the same scale its own scheme's $\Delta J/J$ uses, where it is a flat line. The scatter visible once that row is magnified is not a flow but the least-squares projection of the unrepresentable remainder, and it moves with the command rather than with $h$. Holding $h$ = 5 ms and changing only the command:

| | $\Delta J/J \div h^2$ | $\Delta K_D/K_D \div h^2$ | $\Delta K_P/K_P \div h^2$ |
|---|---|---|---|
| $f_\mathrm{ctrl} \times$ dec = 10×20 → 200×1 | 0.0166 throughout | 4.80e−4 → 4.66e−4 | +3.0e−6 → **−9.6e−6** |
| drive 0.5 → 8 Hz | 0.0166 throughout | 4.78e−4 → 4.79e−4 | −2.2e−6 → **+6.4e−7** |
| amplitude 0.05π → 1.0π | 0.0166 throughout | 4.78e−4 → 5.16e−4 | −1.5e−7 → **+4.9e−5** |
| fitting window 1 s → 4 s | 0.0166 throughout | 4.66e−4 → 4.81e−4 | −9.6e−6 → **+1.04e−5** |

$J$ and $K_D$ are counterterms: they are inside the span, so they do not care what the command is. $\Delta K_P$ changes sign four different ways, grows with amplitude as the gravitational obstruction should, and is three orders of magnitude smaller than $\Delta J/J$ anyway.

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
J\ddot\theta = -\frac{1}{2} m g l \sin\theta + K_P(\theta_{\mathrm{ref},k} - \theta) - K_D\dot\theta,
\qquad t \in [t_k, t_{k+1}).
$$

It is solved interval by interval with DOP853 at `rtol = atol = 1e-12`.

### Simulator

MuJoCo with the `RK4` integrator; the PD torque is computed in Python from the current state at the start of each physics step and held for the whole step.
Holding the torque delays both feedback terms by half a step on average, which is exactly the paper's model with $\delta t = h/2$, so the predictions are

$$
J^\star = J + \frac{h}{2} K_D, \qquad K_D^\star = K_D + \frac{h}{2} K_P, \qquad K_P^\star = K_P.
$$

`common.py` also implements MuJoCo's `Euler` (arm A) and `implicitfast` (arm C) schemes, which shift the bare parameters differently; the flow figure covers all three, the behaviour figure uses RK4.

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

## Experiment 2: driven double pendulum, a chaotic system

The same driven link now carries a second, unactuated link. That link is hinged 40% along its own length, so its mass centre sits 3.05 cm from its hinge and gravity barely restores it — the arrangement of a chaotic-pendulum desk toy. It tumbles 14.5 times in 3 s, and a $10^{-9}$ rad perturbation of the initial state grows to $6\times10^{-5}$ rad over the same window (Lyapunov exponent 3.1 /s).

Here the paper's prediction is the multi-body one from its appendix, $\mathbf M^{\mathrm{sim}} = \mathbf M^{\mathrm{real}} + \delta t \cdot \mathbf K_D$: a constant added to the joint-space inertia, which in MuJoCo is exactly a rotor inertia (`armature`) on the driven joint. The physical joint has none; at 200 Hz the simulator needs 5.0e−3 kg m², which is 7.3% of the joint's own inertia.

### Flow of the effective parameters

![Flow](exp2_double_pendulum/results/fig_flow_mujoco.png)

Fitting (armature, $K_P$, $K_D$), plus a command delay for the two first-order schemes, on a nearly uniform grid $h$ = 1 to 20 ms, each step reached with an integer control frequency and decimation. The three columns are the same MuJoCo schemes as in experiment 1.

| | rotor-inertia slope | predicted | post-fit RMSE vs $h^2$ |
|---|---|---|---|
| B: RK4 + explicit PD | 1.02 per ms ($R^2$ = 0.998) | 1.00 ($K_D/2$) | $R^2$ = 0.993 |
| A: Euler + explicit PD | 2.69 per ms ($R^2$ = 0.984) | 1.00 | $R^2$ = 0.895 |
| C: implicitfast + native PD | 0.30 per ms ($R^2$ = 0.874) | −1.00 | $R^2$ = 0.889 |

**Only RK4 follows the prediction here**, and it follows it closely: slope 1.02 against 1.00, $\Delta K_D$ 4.32 against 4.00, and a post-fit error proportional to $h^2$. For Euler and implicitfast the identified rotor inertia misses the predicted line, the identified command delay comes out negative instead of $+h/2$, and the post-fit error no longer scales as $h^2$ — at $h$ = 1 ms it is 6.1e−3 rad against RK4's 6.9e−5.

That contrast is the main thing the second link adds. With one joint, everything the simulator omits acts through that joint, and all three schemes could absorb it. With a second, unactuated link, a first-order integrator commits an $O(h)$ error in the mechanism itself, part of it on a joint that carries no PD parameter. Adding a rotor inertia on the free joint as well only takes Euler from 4.4e−3 to 2.7e−3 rad, still 7× worse than what RK4 reaches with the formula alone and no fitting. This is step 4 of the paper's procedure reached from the other side: when the effective parameters stop preserving the behaviour, it is the simulator, not the parameter set, that has to be expanded.

### Behaviour

![Result](exp2_double_pendulum/results/fig_double_pendulum_mujoco.png)

**(a) Chaos turns a parameter error into a lost trajectory.** At 200 Hz the measured-parameter simulation follows the real free link for about 1.7 s and then leaves it for good, ending 40 rad — six turns — away. The renormalized one stays within a few radians through 2.5 s.

**(b) 10× smaller error over the first second, with no fitting**, and the same at every resolution.

**Chaos also raises the price of stopping at first order.** In the single pendulum the analytic formula captured about 90% of the gap between measured and best-fit parameters; here it captures 44%, and fitting buys a further 20×. The residual is the second-order term in $h$, which the Lyapunov exponent amplifies. The formula is still 10× better than the measured parameters and still needs no real-world data, but a chaotic observable is where one would expect to need the next order.

The three pendulums side by side: [`exp2_double_pendulum/results/double_pendulum.mp4`](exp2_double_pendulum/results/double_pendulum.mp4) — real, measured, and renormalized overlaid at quarter speed.

### System and method

Both rods are 1 lb and 1 ft, as in experiment 1. Joint 1 carries the same PD controller ($K_P$ = 80, $K_D$ = 2) at 50 Hz control; joint 2 is free. The command is $(\pi/4)\sin(2\pi \cdot 4\ \text{Hz} \cdot t)$, whose 19.7 rad/s peak velocity matches experiment 1.

The reference is the exact continuous-time motion under the held command, integrated with DOP853 at `rtol = atol = 1e-12`. Its right-hand side is MuJoCo's own rigid-body dynamics, so the reference and the simulator describe the same mechanism and differ only in time discretization; `sanity_check.py` confirms the joint-space inertia against the analytic value for two rods and checks that the reference is converged.

The error metric is the joint-angle RMSE over the first second. Beyond that, chaos makes trajectory matching meaningless for every parameter set.

## Isaac Sim: the single pendulum in PhysX

Experiment 1 repeated in Isaac Sim 5.1 against the same reference solution, on the same $h$ grid, drawn by the same code (`figures.py`), so the two engines are directly comparable.

**PhysX has no RK4.** Its articulation solver offers only `TGS` and `PGS`, both first-order schemes with constraint iteration; a high-order explicit integrator does not fit a pipeline built around solving contacts and constraints each step. The three Isaac schemes therefore vary the solver and how the PD force is applied:

| | solver | PD force |
|---|---|---|
| explicit PD + TGS | TGS | torque computed in Python each physics step, as legged_gym does |
| PhysX drive + TGS | TGS | PhysX joint drive: an implicit spring, damping $K_D + \\delta K_P$ at $\\delta = h/32$ |
| PhysX drive + PGS | PGS | the same drive at $\\delta = h$ |

The model is validated first: the simulated error falls as $1/f$ towards the reference — 0.0284, 0.0149, 0.0038, 0.00095 rad at 200, 400, 1600, 6400 Hz — so Isaac and MuJoCo describe the same mechanism.

### Flow of the effective parameters

![Isaac flow](exp1_pendulum/results/fig_flow_isaac_single.png)

Four parameters are fitted: a rotor inertia on the joint (the physical joint has none, so this plays the role of $\Delta J$), $K_P$, $K_D$, and a command delay. Each scheme is given its own counterterms rather than MuJoCo's, because each realizes a different step map.

**What PhysX actually computes.** PhysX 5's CPU code is BSD-3 open source ([NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX)) and this experiment runs the CPU pipeline, so the solver can simply be read. `DyCpuGpuArticulation.h` forms, once per position iteration, with $a = \delta(\delta K_P + K_D)$ and $x = 1/(1 + a/J)$:

$$F \leftarrow F(1-x) - x a \dot\theta + K_P x \delta (r - \theta)$$

whose fixed point is

$$\tau = \frac{K_P (r - \theta) - (\delta K_P + K_D)\dot\theta}{1 + \delta(\delta K_P + K_D)/J}$$

**so the damping the joint feels is $K_D + \delta K_P$, not $K_D$.** PGS runs this at $\delta = h$; TGS advances time inside its position iterations, so there $\delta = h/N$ with $N$ = `solverPositionIterationCount`, 32 by default. Backward error analysis of that map gives

$$J^\star - J = -\frac{\delta K_D}{2} + \delta^2\frac{J K_P + K_D^2}{12 J}, \qquad K_P^\star = K_P, \qquad K_D^\star - K_D = -\delta K_P + \delta^2 \frac{K_P K_D}{12 J}$$

For the explicitly held torque there is nothing to read, so the step map was measured instead: put the simulator in a known state, take one step, subtract the known $h^1$ part and extrapolate $h \to 0$ (`exp1_pendulum/probe_step_map.py`, which identifies MuJoCo's three arms to 0.07%, and `isaac/probe_step_map_isaac.py`). The answer is $M_{2\theta} = f$ and $M_{2v} = 0$ whatever the iteration count: **semi-implicit Euler**, so MuJoCo's Euler counterterms apply unchanged.

| | $c_1$ identified | $c_1$ predicted |
|---|---|---|
| explicit PD + TGS: rotor inertia$/J$ | +0.0634 per ms | +0.0712 ($K_D/2J$) |
| explicit PD + TGS: command delay | 0.56 per ms | 0.50 ($h/2$) |
| **PhysX drive + PGS: rotor inertia$/J$** | **−0.0706 per ms** | −0.0712 |
| **PhysX drive + PGS: $\Delta K_D/K_D$** | **−0.0399 per ms** | −0.0400 ($-K_P/K_D$) |
| PhysX drive + PGS: command delay | 0.50 per ms | 0.50 |
| PhysX drive + TGS: rotor inertia$/J$ | −0.00212 per ms | −0.00222 ($-K_D/2JN$) |
| PhysX drive + TGS: $\Delta K_D/K_D$ | −0.00132 per ms | −0.00125 ($-K_P/K_D N$) |

The PGS column matches to four digits with $R^2$ = 0.999, including the $h^2$ curvature of $\Delta K_D$: 4.66e−4 against the predicted 4.75e−4 per ms². $\Delta K_P$ is predicted not to flow and does not, in any scheme.

**The counterterm is a property of the discretization, not of the engine**, and the sign is a property of the scheme. An explicitly held torque gives $+hK_D/2$ in both engines; the implicit drive gives $-\delta K_D/2$, the same flip MuJoCo shows for `implicitfast`. TGS is not a different physics from PGS: it is the same drive run at a step 32 times smaller, which is also why it is the most accurate scheme out of the box — its measured-parameter error at $h$ = 1 ms is 1.8e−4 rad, 31× below PGS.

**Only well-conditioned control rates are plotted.** The pendulum's settling time is $2J/K_D$ = 14 ms. At $f_\mathrm{ctrl}$ = 100 Hz the control period is shorter than that and every sample carries information about the parameters; at 10 Hz the pendulum reaches the static equilibrium for each held command within the first 14 ms of a 100 ms period, so most of the trajectory says nothing and the fit is ill-conditioned — its error with the measured parameters is six times *smaller* there, which is the symptom, not a success.

### Behaviour

![Isaac behaviour](exp1_pendulum/results/fig_behaviour_isaac_single.png)

With the predicted parameters, and no fitting at all, the error at 200 Hz drops 22× and at 1 kHz 182×. Below 110 Hz the measured parameters diverge outright while the renormalized ones keep running — the same stability rescue MuJoCo showed.

The video, rendered with the MuJoCo renderer from Isaac's joint angles so both engines produce the same picture: [`exp1_pendulum/results/isaac_single.mp4`](exp1_pendulum/results/isaac_single.mp4).

### Where the second order is not enough

Once $\delta = h/N$ is used, TGS's first-order counterterms come out right, but its $h^2$ coefficient does not. Composing $N$ substeps predicts $c_2 \propto 1/N^2$; sweeping $N$ from 1 to 32 instead gives $c_1 \times N$ constant at −0.070 against the predicted −0.0712, while $c_2$ changes sign between $N$ = 1 and 4 and then saturates near −5e−4. Something inside a TGS step happens once per step rather than once per substep: it cancels at first order and survives at second. At $N$ = 1, TGS reduces exactly to PGS — $c_1$ = −0.0701 and $c_2$ = +1.75e−3 against −0.0712 and +2.16e−3.

## Reproduce

```bash
uv sync
cd exp1_pendulum
uv run python sanity_check.py     # MuJoCo model vs ground-truth ODE
uv run python run_gap.py          # error of each parameter set across resolutions
uv run python plot_pendulum.py    # the figure
uv run python run_flow.py         # effective parameters on a uniform grid of time steps
uv run python run_flow.py --delay # same, also identifying the command delay
uv run python plot_flow.py        # flow figure
uv run python video.py            # pendulum video (needs EGL)

cd ../exp2_double_pendulum
uv run python sanity_check.py     # inertia, reference convergence, chaos
uv run python run_flow.py         # effective parameters on a uniform grid of time steps
uv run python plot_flow.py        # flow figure
uv run python run_gap.py          # error of each parameter set across resolutions
uv run python plot_pendulum.py    # behaviour figure
uv run python video.py            # three pendulums overlaid (needs EGL)
```

Isaac Sim runs in its own Python 3.11 environment (`isaacsim[all]==5.1.0.0`), with `mujoco`
installed alongside so that it can call the same reference solution. Its figures are drawn by
the same `figures.py` as the MuJoCo ones, so the two cannot drift apart.

```bash
cd isaac
# flow: the MuJoCo grid, one 2 Hz command, as exp1/run_flow.py uses
$ISAAC_VENV/bin/python run_single.py --arm A --delay --points 8 \
    --h-min 0.25 --h-max 5 --single-drive          # also --arm B, C
# behaviour: five commands over f = 50-1000 Hz, as exp1/run_gap.py uses
$ISAAC_VENV/bin/python run_single.py --arm A --delay --points 8 --h-min 1 --h-max 20
$ISAAC_VENV/bin/python dump_traj.py single
uv run python plot_isaac_flow.py        # figures, in the project venv
uv run python plot_isaac_behaviour.py
uv run python video_isaac.py --exp single
```

`double_isaac.py` and `run_double.py` hold a validated Isaac model of the chaotic double
pendulum; that experiment has not been run to completion yet.

## Roadmap

Each step is one commit, reviewed before the next begins.

- [x] Plan
- [x] Experiment 1, MuJoCo: flow, gap, and the pendulum video
- [x] Experiment 2, MuJoCo: the chaotic double pendulum
- [x] Isaac, experiment 1: the single pendulum in PhysX
- [ ] Isaac, experiment 2: the chaotic double pendulum
