"""The README animation: the real pendulum and two 200 Hz simulators of it, overlaid.

The command is the 2 Hz drive of the behaviour figure's panel (a).  One simulator is given the
measured parameters and the other the renormalized ones, and both are drawn in front of the real
pendulum, so a simulator that matches it lies on top of it rather than hiding inside it.  The
camera is orthographic, so drawing the rods at different depths does not change their size.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import F_REF, G, L, M, PARAM0, Sim, ground_truth, ref_sequence
from figures import save_gif
from run_gap import F_CTRL, renormalized

ARM = "B"
DEC = 4  # 200 Hz physics, the legged_gym default
T_SHOW = 2.0  # s, the window of the behaviour figure's panel (a)
FPS, SPEED = 25, 0.25
WIDTH, HEIGHT = 560, 440
VIEW_H, VIEW_Z = 0.62, -0.04  # m: height of the view and its centre
# name, colour, rod radius, depth (the camera looks along +y, so smaller y is in front)
TRACES = (("real", "0.80 0.79 0.77 1", 0.013, 0.04),
          ("measured", "0.92 0.41 0.20 1", 0.006, 0.0),
          ("renormalized", "0.16 0.47 0.84 1", 0.006, -0.04))
LABELS = (("real", (82, 81, 78)), ("without renormalization", (235, 104, 52)),
          ("renormalized", (42, 120, 214)))


def rod(name, rgba, size, y):
    return f"""
    <body name="{name}" pos="0 {y} 0">
      <joint name="{name}_hinge" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {-L / 2}" mass="{M}" diaginertia="1 1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 {-L}" size="{size}" rgba="{rgba}"
            contype="0" conaffinity="0"/>
    </body>"""


XML = f"""
<mujoco>
  <compiler inertiafromgeom="false"/>
  <option gravity="0 0 -{G}"/>
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
    <headlight ambient=".55 .55 .55" diffuse=".5 .5 .5" specular="0 0 0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>
  </asset>
  <worldbody>
    <camera name="side" pos="0 -1 {VIEW_Z}" xyaxes="1 0 0 0 0 1" projection="orthographic"
            fovy="{VIEW_H}"/>
    <geom type="cylinder" fromto="0 -0.07 0 0 0.07 0" size="0.012" rgba=".3 .3 .3 1"
          contype="0" conaffinity="0"/>
    {"".join(rod(*trace) for trace in TRACES)}
  </worldbody>
</mujoco>"""


def trajectories():
    """Theta at t = 0, h, 2h, ... for the real pendulum and the two simulators."""
    refs = ref_sequence(F_CTRL, F_REF, t_end=T_SHOW)
    h = 1.0 / (F_CTRL * DEC)
    sim = Sim(ARM)
    out = [ground_truth(refs, F_CTRL, DEC)]
    out += [sim.rollout(refs, F_CTRL, DEC, *p) for p in (PARAM0, renormalized(ARM, h))]
    return np.vstack([np.zeros((1, len(out))), np.stack(out, axis=1)])  # (sample, trace)


def main(out):
    traj = trajectories()
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    stride = round(F_CTRL * DEC * SPEED / FPS)  # physics steps per animation frame
    font = ImageFont.load_default(size=19)

    frames = []
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        for i in range(0, len(traj), stride):
            data.qpos[:] = traj[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="side")
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.text((16, 14), f"{F_CTRL * DEC} Hz physics, {F_REF:g} Hz command",
                      fill=(11, 11, 11), font=font)
            draw.text((16, 40), f"t = {i / (F_CTRL * DEC):4.2f} s   ({SPEED:g}x speed)",
                      fill=(82, 81, 78), font=font)
            for k, (label, rgb) in enumerate(LABELS):
                draw.text((WIDTH - 16, 14 + 26 * k), label, fill=rgb, font=font, anchor="ra")
            frames.append(np.asarray(img))
    save_gif(frames, out, FPS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "results" / "pendulum.gif"))
