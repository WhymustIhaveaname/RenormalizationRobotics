"""The README animation: three double pendulums at once, the real one and two simulators.

The real motion, a 200 Hz simulator given the measured parameters, and the same simulator given
the renormalized ones are overlaid on the 4 Hz command the behaviour figure scores, so the moment
each loses the real motion is visible directly.  Both simulators are drawn in front of the real
pendulum, so while one matches it, it lies on top of it rather than hiding inside it.  The camera
is orthographic, so drawing the chains at different depths does not change their size.

Underneath, a strip traces each simulator's error on the free joint as the clip runs.  The error
lives on the universal cover R of the joint's circle, not on the circle: each extra turn the
simulated link makes adds 2 pi to it, rather than folding it back to -pi.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (F_CTRL, F_DRIVE, G, HINGE_FRAC, L1, L2, M1, M2, PARAM0, Sim, ref_sequence,
                    renormalized)
from figures import error_strip, gif_font, save_gif

DEC = 4  # 200 Hz physics for both simulators, and the sampling rate of the animation
T_END = 2.5  # s, the window of the behaviour figure's panel (a)
FPS, SPEED = 25, 0.25
WIDTH, HEIGHT = 560, 440  # the scene; the error strip goes underneath
STRIP_H = 150
STRIP_BOX = (70, HEIGHT + 36, WIDTH - 20, HEIGHT + STRIP_H - 14)
ERR_LIM = (-2.0, 44.0)  # rad; holds the whole error, which reaches 13.6 pi
ERR_TICKS = tuple((k * np.pi, f"{k}π" if k else "0") for k in (0, 4, 8, 12))
VIEW_H, VIEW_Z = 0.74, -0.165  # m: height of the view and its centre
# name, colour, rod radius, depth (the camera looks along +y, so smaller y is in front)
TRACES = (("real", "0.80 0.79 0.77 1", 0.013, 0.04),
          ("measured", "0.92 0.41 0.20 1", 0.006, 0.0),
          ("renormalized", "0.16 0.47 0.84 1", 0.006, -0.04))
LABELS = (("real", (82, 81, 78)), ("without renormalization", (235, 104, 52)),
          ("renormalized", (42, 120, 214)))


def chain(name, rgba, size, y):
    a, b = HINGE_FRAC * L2, (1 - HINGE_FRAC) * L2
    return f"""
    <body name="{name}1" pos="0 {y} 0">
      <joint name="{name}_j1" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {-L1 / 2}" mass="{M1}" diaginertia="1 1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 {-L1}" size="{size}" rgba="{rgba}"
            contype="0" conaffinity="0"/>
      <body name="{name}2" pos="0 0 {-L1}">
        <joint name="{name}_j2" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 {(a - b) / 2}" mass="{M2}" diaginertia="1 1 1"/>
        <geom type="capsule" fromto="0 0 {a} 0 0 {-b}" size="{size}" rgba="{rgba}"
              contype="0" conaffinity="0"/>
      </body>
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
    <geom type="cylinder" fromto="0 -0.07 0 0 0.07 0" size="0.014" rgba=".3 .3 .3 1"
          contype="0" conaffinity="0"/>
    {"".join(chain(*trace) for trace in TRACES)}
  </worldbody>
</mujoco>"""


def trajectories():
    """Joint angles at t = 0, h, 2h, ... for the real pendulum and the two simulators."""
    sim = Sim()
    refs = ref_sequence(T_END)
    h = 1.0 / (F_CTRL * DEC)
    out = [sim.ground_truth(refs, sub=DEC)]
    out += [sim.rollout_dense(refs, DEC, *p) for p in (PARAM0, renormalized(h))]
    traj = np.stack(out, axis=1)  # (sample, trace, joint)
    return np.concatenate([np.zeros((1, *traj.shape[1:])), traj])  # every run starts at rest


def main(out):
    traj = trajectories()
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    stride = max(1, round(F_CTRL * DEC * SPEED / FPS))
    font, small = gif_font(17), gif_font(14)
    errors = [(traj[:, k, 1] - traj[:, 0, 1], rgb) for k, (_, rgb) in zip((1, 2), LABELS[1:])]

    frames = []
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        for i in range(0, len(traj), stride):
            data.qpos[:] = traj[i].reshape(-1)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="side")
            scene = renderer.render()
            img = Image.new("RGB", (WIDTH, HEIGHT + STRIP_H), tuple(scene[-1, 0]))
            img.paste(Image.fromarray(scene))
            draw = ImageDraw.Draw(img)
            error_strip(draw, STRIP_BOX, errors, i, ERR_LIM, ERR_TICKS,
                        "free joint θ₂, simulated − real (rad)", small)
            draw.text((16, 14), f"{F_CTRL * DEC} Hz physics, {F_DRIVE:g} Hz command",
                      fill=(11, 11, 11), font=font)
            draw.text((16, 40), f"t = {i / (F_CTRL * DEC):4.2f} s   ({SPEED:g}x speed)",
                      fill=(82, 81, 78), font=font)
            for k, (label, rgb) in enumerate(LABELS):
                draw.text((WIDTH - 16, 14 + 26 * k), label, fill=rgb, font=font, anchor="ra")
            frames.append(np.asarray(img))
    save_gif(frames, out, FPS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else str(Path(__file__).parent / "results" / "double_pendulum.gif"))
