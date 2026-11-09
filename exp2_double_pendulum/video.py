"""Three double pendulums at once: the real one and two simulators, drifting apart.

The real motion, a 200 Hz simulator given the measured parameters, and the same simulator given
the renormalized ones are overlaid in a single scene, so the moment each loses the real motion
is visible directly.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import (F_CTRL, G, HINGE_FRAC, L1, L2, M1, M2, PARAM0, Sim, ref_sequence,
                    renormalized)

DEC = 4  # 200 Hz physics for both simulators, and the sampling rate of the video
T_END = 2.5
FPS, SPEED = 50, 0.25
SIZE = 900
TRACES = (("real", "0.80 0.79 0.77 1", 0.013),
          ("measured", "0.92 0.41 0.20 1", 0.007),
          ("renormalized", "0.16 0.47 0.84 1", 0.007))
LABEL_RGB = ((82, 81, 78), (235, 104, 52), (42, 120, 214))


def chain(name, rgba, size):
    a, b = HINGE_FRAC * L2, (1 - HINGE_FRAC) * L2
    return f"""
    <body name="{name}1">
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
  <visual><global offwidth="{SIZE}" offheight="{SIZE}"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".86 .88 .91" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0 -1 1" dir="0 1 -1"/>
    <geom type="cylinder" fromto="0 -0.04 0 0 0.04 0" size="0.014" rgba=".3 .3 .3 1"
          contype="0" conaffinity="0"/>
    {"".join(chain(n, c, s) for n, c, s in TRACES)}
  </worldbody>
</mujoco>"""


def trajectories():
    sim = Sim()
    refs = ref_sequence(T_END)
    h = 1.0 / (F_CTRL * DEC)
    real = sim.ground_truth(refs, sub=DEC)
    out = [real]
    for params in (PARAM0, renormalized(h)):
        theta = sim.rollout_dense(refs, DEC, *params)
        out.append(theta)
    return np.stack(out, axis=1)  # (sample, trace, joint)


def main(out):
    traj = trajectories()
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    stride = max(1, round(F_CTRL * DEC * SPEED / FPS))

    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0, 0, -0.32)
    cam.distance, cam.azimuth, cam.elevation = 1.5, 90, 0
    font = ImageFont.load_default(size=26)

    with mujoco.Renderer(model, SIZE, SIZE) as renderer, \
            imageio.get_writer(out, fps=FPS, codec="libx264", quality=8) as writer:
        for i in range(0, len(traj), stride):
            data.qpos[:] = traj[i].reshape(-1)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            t = (i + 1) / (F_CTRL * DEC)
            draw.text((20, 18), f"t = {t:4.2f} / {T_END:g} s   ({SPEED:g}x speed)",
                      fill=(11, 11, 11), font=font)
            for k, ((name, _, _), rgb) in enumerate(zip(TRACES, LABEL_RGB)):
                draw.text((20, 56 + 32 * k), name, fill=rgb, font=font)
            writer.append_data(np.asarray(img))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else str(Path(__file__).parent / "results" / "double_pendulum.mp4"))
