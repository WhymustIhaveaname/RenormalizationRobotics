"""Render Isaac trajectories with the MuJoCo renderer, so both engines produce the same picture.

The mechanism is identical in the two engines; only the dynamics behind the joint angles differ.
Run in the project venv:  python video_isaac.py --exp single|double
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
G, M, L = 9.81, 0.45359237, 0.3048
HINGE_FRAC = 0.40
FPS, SIZE = 50, 800
TRACES = (("real", "0.80 0.79 0.77 1", 0.013),
          ("Isaac, measured", "0.92 0.41 0.20 1", 0.007),
          ("Isaac, renormalized", "0.16 0.47 0.84 1", 0.007))
LABEL_RGB = ((82, 81, 78), (235, 104, 52), (42, 120, 214))


def chain(name, rgba, size, double):
    a, b = HINGE_FRAC * L, (1 - HINGE_FRAC) * L
    inner = f"""
      <body name="{name}2" pos="0 0 {-L}">
        <joint name="{name}_j2" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 {(a - b) / 2}" mass="{M}" diaginertia="1 1 1"/>
        <geom type="capsule" fromto="0 0 {a} 0 0 {-b}" size="{size}" rgba="{rgba}"
              contype="0" conaffinity="0"/>
      </body>""" if double else ""
    return f"""
    <body name="{name}1">
      <joint name="{name}_j1" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {-L / 2}" mass="{M}" diaginertia="1 1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 {-L}" size="{size}" rgba="{rgba}"
            contype="0" conaffinity="0"/>{inner}
    </body>"""


def xml(double):
    return f"""
<mujoco>
  <compiler inertiafromgeom="false"/>
  <option gravity="0 0 -{G}"/>
  <visual><global offwidth="{SIZE}" offheight="{SIZE}"/></visual>
  <asset><texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".86 .88 .91"
         width="256" height="256"/></asset>
  <worldbody>
    <light pos="0 -1 1" dir="0 1 -1"/>
    <geom type="cylinder" fromto="0 -0.04 0 0 0.04 0" size="0.014" rgba=".3 .3 .3 1"
          contype="0" conaffinity="0"/>
    {"".join(chain(n, c, s, double) for n, c, s in TRACES)}
  </worldbody>
</mujoco>"""


def main(exp):
    double = exp == "double"
    d = ROOT / ("exp2_double_pendulum" if double else "exp1_pendulum") / "results"
    z = np.load(d / "isaac_traj.npz")
    traces = [z["reference"], z["A_measured"], z["A_renormalized"]]
    traj = np.stack([t.reshape(len(t), -1) for t in traces], axis=1)  # (sample, trace, dof)

    model = mujoco.MjModel.from_xml_string(xml(double))
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0, 0, -0.32 if double else -0.15)
    cam.distance, cam.azimuth, cam.elevation = (1.5 if double else 1.0), 90, 0
    font = ImageFont.load_default(size=24)
    out = d / f"isaac_{exp}.mp4"

    # The dumped trajectories are sampled at the 50 Hz control rate; slow down to see the motion.
    repeat = 4
    with mujoco.Renderer(model, SIZE, SIZE) as renderer, \
            imageio.get_writer(out, fps=FPS, codec="libx264", quality=8) as writer:
        for i in range(len(traj)):
            data.qpos[:] = traj[i].reshape(-1)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.text((20, 18), f"t = {(i + 1) / 50:4.2f} s   ({50 / (FPS / repeat):.0f}x slower)",
                      fill=(11, 11, 11), font=font)
            for k, ((name, _, _), rgb) in enumerate(zip(TRACES, LABEL_RGB)):
                draw.text((20, 54 + 30 * k), name, fill=rgb, font=font)
            frame = np.asarray(img)
            for _ in range(repeat):
                writer.append_data(frame)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="single", choices=("single", "double"))
    main(ap.parse_args().exp)
