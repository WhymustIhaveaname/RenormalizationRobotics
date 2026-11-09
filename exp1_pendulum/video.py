"""Render the ground-truth pendulum tracking the held 50 Hz command (0.5x speed)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import G, L, M, ground_truth, ref_sequence

F_CTRL, DEC = 50, 12  # ground truth sampled at 600 Hz
FPS, SPEED = 60, 0.5
SIZE = 720

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
    <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.012" rgba=".3 .3 .3 1"/>
    <body name="rod">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {-L / 2}" mass="{M}" diaginertia="1 1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 {-L}" size="0.008" rgba=".92 .41 .2 1"/>
    </body>
    <body name="target" mocap="true">
      <geom type="capsule" fromto="0 0 0 0 0 {-L}" size="0.014" rgba=".16 .47 .84 .3"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""


def main(out):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    refs = ref_sequence(F_CTRL)
    theta = np.concatenate([[0.0], ground_truth(refs, F_CTRL, DEC)])
    stride = round(F_CTRL * DEC * SPEED / FPS)  # GT samples per video frame

    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0, 0, -0.08)
    cam.distance, cam.azimuth, cam.elevation = 0.95, 90, 0
    font = ImageFont.load_default(size=24)

    with mujoco.Renderer(model, SIZE, SIZE) as renderer, \
            imageio.get_writer(out, fps=FPS, codec="libx264", quality=8) as writer:
        for i in range(0, len(theta), stride):
            t = i / (F_CTRL * DEC)
            r = refs[min(int(t * F_CTRL), len(refs) - 1)]
            data.qpos[0] = theta[i]
            data.mocap_quat[0] = (np.cos(r / 2), 0, np.sin(r / 2), 0)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.text((20, 18), f"t = {t:4.2f} s   ({SPEED:g}x speed)", fill=(11, 11, 11), font=font)
            draw.text((20, 50), "orange: real (continuous PD)", fill=(82, 81, 78), font=font)
            draw.text((20, 80), "blue: command, 50 Hz held", fill=(82, 81, 78), font=font)
            writer.append_data(np.asarray(img))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "results" / "pendulum.mp4"))
