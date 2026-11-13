# R4R: Renormalization for Robotics

Experiments for *Effective Parameters, Real Behavior: Renormalization for Robotics — From Infinite Electron Mass to Sim-to-Real Gap*, by Youran Sun, Jiaxuan Guo, Xingyu Ren, Chugang Yi and Haizhao Yang.

[![arXiv](https://img.shields.io/badge/arXiv-2607.24079-b31b1b.svg)](https://arxiv.org/abs/2607.24079)
[![PDF](https://img.shields.io/badge/PDF-b31b1b.svg)](https://arxiv.org/pdf/2607.24079)
[![BibTeX](https://img.shields.io/badge/BibTeX-555555.svg)](#citation)

A simulator needs effective parameters, which change with the simulation frequency $f$. For PD control, an effective delay $\delta t \propto 1/f$ shifts them as follows:

$$
K_D^{\mathrm{sim}} = K_D^{\mathrm{real}} + \delta t \cdot K_P^{\mathrm{real}},
\qquad
J^{\mathrm{sim}} = J^{\mathrm{measured}} + \delta t \cdot K_D^{\mathrm{real}}.
$$

<p align="center">
  <img src="exp1_pendulum/results/pendulum.gif" width="49%" alt="The real pendulum and two 200 Hz simulators of it">
  <img src="exp2_double_pendulum/results/double_pendulum.gif" width="49%" alt="The real double pendulum and two 200 Hz simulators of it">
</p>

## Experiment 1: PD-controlled pendulum

The pendulum is a uniform rod, 1 lb and 1 ft long, under gravity. A PD controller drives it with the ANYmal-C gains of [legged_gym](https://github.com/leggedrobotics/legged_gym): $K_P$ = 80 N m/rad, $K_D$ = 2 N m s/rad. The commands are sines from 0.5 to 8 Hz, held at the control rate. The real pendulum is the exact continuous-time solution. MuJoCo holds an explicit PD torque over each physics step, so $\delta t = h/2$. For each time step $h$, least squares finds the $(J, K_P, K_D)$ that best fits the real trajectory.

![Result](exp1_pendulum/results/fig_pendulum_mujoco.png)

At the same simulation frequency, the first-order formula of the paper gives a 36× smaller error. For the same accuracy, it needs 36× less computation.

### Flow of the effective parameters

![Flow](exp1_pendulum/results/fig1_flow_mujoco.png)

At second order, backward error analysis gives the effective parameters of A (upper sign) and C (lower sign):

$$J_e = J \mp \frac{h K_D}{2} - h^2 \frac{J K_P + K_D^2}{12 J}, \qquad K_{P,e} = K_P, \qquad K_{D,e} = K_D - h^2 \frac{K_P K_D}{12 J}$$

The simulator needs the opposite shifts. For a linear plant, the three parameters absorb all of the $h^2$ error. Gravity leaves a remainder that no parameter can absorb. `derive_second_order.py` derives each step map and checks it numerically.

## Experiment 2: chaotic double pendulum

A second, free link hangs from the end of the pendulum of experiment 1. Its mass center is near its hinge, so it tumbles chaotically. The multi-body prediction of the paper is $\mathbf M^{\mathrm{sim}} = \mathbf M^{\mathrm{real}} + \delta t \cdot \mathbf K_D$. In MuJoCo, this is a rotor inertia (`armature`) on the driven joint.

![Result](exp2_double_pendulum/results/fig_double_pendulum_mujoco.png)

Chaos turns a small parameter error into a lost trajectory. At 200 Hz, the formula gives a 10× smaller error. For the same accuracy, it needs 11× less computation.

### Flow of the effective parameters

![Flow](exp2_double_pendulum/results/fig_flow_mujoco.png)

The theory gives $K_P$ no counterterm. A free $K_P$ moves the rotor inertia by up to 68%, and the error stays the same.

## Isaac Sim: the single pendulum in PhysX

This section repeats experiment 1 in Isaac Sim 5.1. PhysX has no RK4, so the three schemes change the solver and the PD force.

![Isaac behaviour](exp1_pendulum/results/fig_behaviour_isaac_single.png)

At 156 Hz, the formula gives a 22× smaller error. For the same accuracy, it needs 11× less computation.

### Flow of the effective parameters

![Isaac flow](exp1_pendulum/results/fig_flow_isaac_single.png)

The PhysX source is open ([NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX)), so the theory comes from the solver code. TGS substeps the drive but applies gravity once per step. With this fact, the TGS drive matches the theory within 3%.

## MuJoCo Warp: the same pendulum on the GPU

This experiment is a high-level sanity check of MuJoCo Warp. It checks whether MuJoCo Warp gives the same results as MuJoCo. It does, so you can safely move from MuJoCo to MuJoCo Warp, and vice versa.

![MuJoCo Warp flow](exp1_pendulum/results/fig1_flow_mjwarp.png)

Every flow agrees to four digits, and the trajectories agree to float32 round-off.

## Reproduce

```bash
uv sync
cd exp1_pendulum
uv run python sanity_check.py
uv run python run_gap.py
uv run python plot_pendulum.py
uv run python run_flow.py
uv run python run_flow.py --delay
uv run python plot_flow.py
uv run python video.py                 # needs EGL
uv run python derive_second_order.py
uv run python probe_step_map.py

cd ../exp2_double_pendulum
uv run python sanity_check.py
uv run python run_flow.py
uv run python run_flow.py --delay
uv run python plot_flow.py
uv run python run_gap.py
uv run python plot_pendulum.py
uv run python video.py                 # needs EGL
```

Isaac Sim needs its own Python 3.11 environment, with `isaacsim[all]==5.1.0.0` and `mujoco`.

```bash
cd isaac
$ISAAC_VENV/bin/python run_single.py --arm A    # also B and C; about 6 h each
$ISAAC_VENV/bin/python dump_traj.py single
$ISAAC_VENV/bin/python probe_step_map_isaac.py A 1 4 32
$ISAAC_VENV/bin/python probe_drive_law.py B
uv run python plot_isaac_flow.py
uv run python plot_isaac_behaviour.py
uv run python video_isaac.py --exp single

cd ../mjwarp
./launch.sh
cd ../exp1_pendulum
uv run python plot_flow.py --engine mjwarp
```

## Citation

```bibtex
@misc{sun2026effective,
      title={Effective Parameters, Real Behavior: Renormalization for Robotics - From Infinite Electron Mass to Sim-to-Real Gap},
      author={Youran Sun and Jiaxuan Guo and Xingyu Ren and Chugang Yi and Haizhao Yang},
      year={2026},
      eprint={2607.24079},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2607.24079},
}
```
