"""Dump Isaac trajectories at 200 Hz for the behaviour figures and the videos.

Writes one npz per system holding the reference, the measured-parameter run and the
renormalized-parameter run, for every arm.
"""

import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

ROOT = Path("/home/youran/RenormalizationRobotics")
sys.path.insert(0, str(ROOT / "isaac"))

DEC = 4  # 200 Hz physics at 50 Hz control


def dump_single():
    sys.path.insert(0, str(ROOT / "exp1_pendulum"))
    import common as mj
    from pendulum_isaac import F_CTRL, KD0, KP0, IsaacPendulum
    from isaacsim.core.api import World

    h = 1.0 / (F_CTRL * DEC)
    world = World(physics_dt=h, stage_units_in_meters=1.0, backend="numpy")
    out = {}
    for arm in ("A", "B", "C"):
        sim = IsaacPendulum(world, arm, 2) if arm == "A" else sim
        sim.arm = arm
        refs = mj.ref_sequence(F_CTRL, mj.F_REF)
        gt = mj.ground_truth(refs, F_CTRL, DEC)[DEC - 1::DEC]
        measured = np.array([0.0, KP0, KD0])
        renorm = np.array([h / 2 * KD0, KP0, KD0 + h / 2 * KP0])
        out[f"{arm}_measured"] = sim.rollout(refs, DEC, measured[None, :])[0]
        out[f"{arm}_renormalized"] = sim.rollout(refs, DEC, renorm[None, :])[0]
        out["reference"] = gt
        out["refs"] = refs
    np.savez_compressed(ROOT / "exp1_pendulum" / "results" / "isaac_traj.npz", **out)
    print("wrote isaac_traj.npz")


def dump_double():
    sys.path.insert(0, str(ROOT / "exp2_double_pendulum"))
    import common as mj
    from double_isaac import F_CTRL, KD0, KP0, IsaacDouble
    from isaacsim.core.api import World

    h = 1.0 / (F_CTRL * DEC)
    world = World(physics_dt=h, stage_units_in_meters=1.0, backend="numpy")
    sim = IsaacDouble(world, "A", 2)
    ref_sim = mj.Sim()
    refs = mj.ref_sequence(1.0)
    gt = ref_sim.ground_truth(refs)
    out = {"reference": gt, "refs": refs}
    for arm in ("A", "B", "C"):
        sim.arm = arm
        measured = np.array([0.0, KP0, KD0])
        renorm = np.array([h / 2 * KD0, KP0, KD0 + h / 2 * KP0])
        out[f"{arm}_measured"] = sim.rollout(refs, DEC, measured[None, :])[0]
        out[f"{arm}_renormalized"] = sim.rollout(refs, DEC, renorm[None, :])[0]
    np.savez_compressed(ROOT / "exp2_double_pendulum" / "results" / "isaac_traj.npz", **out)
    print("wrote isaac_traj.npz (double)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "single"
    (dump_single if which == "single" else dump_double)()
    app.close()
