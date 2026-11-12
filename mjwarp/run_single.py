"""Experiment 1 swept on MuJoCo Warp, using exp1_pendulum/run_flow.py unchanged.

The sweep, the grid, the starting points, the reference solution and the fitter all come from
the MuJoCo code; the only thing supplied here is a simulator with the same rollout signature
and a batched Jacobian, so the two engines cannot differ by anything but their stepping.

One (arm, nominal control frequency) chain per process, because a 1-DOF model leaves the GPU
launch-bound rather than compute-bound: several processes overlap almost perfectly, while a
single process stepping more worlds would not go any faster.

Usage (inside the MuJoCo Warp venv):
    python run_single.py --arm A --fctrl 50 [--delay]
    python run_single.py --merge [--delay]          # concatenate the per-chain files
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_flow as rf  # noqa: E402

NSTEP_MAX = int(round(4.0 / 0.25e-3)) + 1          # 4 s of command at the finest step
RESULTS = ROOT / "exp1_pendulum" / "results"
PARTS = RESULTS / "mjwarp_parts"


def flag(name, default=None):
    return next((a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
                 for a in sys.argv if a.split("=", 1)[0] == f"--{name}"), default)


ONLY_SEED = "--only-seed" in sys.argv


def seeded_starts(reference):
    """Start each fit from the other engine's answer for the same grid point.

    The two engines agree step for step to float32 round-off, so the minimum is already known
    to several digits; starting there turns twenty LM iterations into two or three.  The fit is
    still an independent one - it converges on MuJoCo Warp's own trajectories, and the measured
    and predicted parameter sets are kept as a safety net against a bad local minimum.
    """
    rows = json.loads(Path(reference).read_text())
    plain = rf.start_points

    def starts(h, predicted, prev, rng, with_delay):
        near = min(rows, key=lambda r: abs(r["h"] - h))
        base = plain(h, predicted, prev, rng, with_delay)
        seed = np.asarray(near["params"], float)
        if ONLY_SEED:
            return [seed]             # the minimum is already known; one start is enough
        return [seed] + base[:2]      # the reference, then measured and predicted

    return starts


def stem():
    return "flow_mjwarp_delay" if rf.WITH_DELAY else "flow_mjwarp"


def merge():
    # the [ABC] keeps "flow_mjwarp_*" from also matching "flow_mjwarp_delay_*"
    parts = sorted(PARTS.glob(f"{stem()}_[ABC]_*.json"))
    rows = [r for f in parts for r in json.loads(f.read_text())]
    out = RESULTS / f"{stem()}.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"merged {len(rows)} fits from {len(parts)} chains into {out}")


def main():
    if "--merge" in sys.argv:
        return merge()
    from pendulum_mjwarp import WarpPendulum, batch_for

    arm, f_nominal = flag("arm", "A"), int(flag("fctrl", "50"))
    ndim = 4 if rf.WITH_DELAY else 3
    # The fitter advances every starting point at once, each asking for 2 * ndim
    # finite-difference probes, so a round needs that many worlds.  Stepping sixty-four worlds
    # costs the same as stepping eight for a 1-DOF model, which is the whole point.
    rf.make_sim = lambda a: WarpPendulum(a, nworld=8 * 2 * ndim, nstep_max=NSTEP_MAX)
    rf.make_batch = batch_for
    ref = flag("seed")
    if ref:
        rows = json.loads(Path(ref).read_text())
        same = [r for r in rows if r["arm"] == arm and r["f_ctrl_nominal"] == f_nominal]
        if not same:
            raise SystemExit(f"no {arm} at f_ctrl {f_nominal} in {ref}")
        seed_file = PARTS / f".seed_{arm}_{f_nominal}.json"
        PARTS.mkdir(parents=True, exist_ok=True)
        seed_file.write_text(json.dumps(same))
        rf.start_points = seeded_starts(seed_file)
        print(f"seeding from {len(same)} MuJoCo fits in {ref}", flush=True)

    PARTS.mkdir(parents=True, exist_ok=True)
    part = PARTS / f"{stem()}_{arm}_{f_nominal:03d}.json"
    rf.on_row = lambda rows: part.write_text(json.dumps(rows, indent=1, default=float))
    t0 = time.perf_counter()
    rows = rf.chain((arm, f_nominal))
    print(f"wrote {len(rows)} fits in {(time.perf_counter() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
