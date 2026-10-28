#!/usr/bin/env python3
"""Create a camera-equipped scene with a variable number of unnamed-to-task cubes."""

import argparse
import colorsys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.count <= 12:
        parser.error("count must be between 0 and 12 (the tray itself holds eight)")
    source = Path(__file__).resolve().parents[1] / "models/ur5e_robotiq_vision_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(source))
    arm_qpos = []
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            address = model.jnt_qposadr[joint]
            end = model.jnt_qposadr[joint+1] if joint+1 < model.njnt else model.nq
            arm_qpos.extend(model.key_qpos[0, address:end])
    root = ET.parse(source).getroot()
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        if body.find("freejoint") is not None:
            world.remove(body)
    rng = np.random.default_rng(args.seed)
    grid = np.array([(x, y) for y in (-0.40, -0.28, -0.16)
                     for x in (0.34, 0.44, 0.54, 0.64)])
    rng.shuffle(grid)
    qpos = list(arm_qpos)
    for index, xy in enumerate(grid[:args.count]):
        xy = xy + rng.uniform(-0.004, 0.004, 2)
        color = colorsys.hsv_to_rgb(float(rng.random()), 0.65, 0.9)
        body = ET.SubElement(world, "body", name=f"parcel_{index:02d}",
                             pos=f"{xy[0]:.6f} {xy[1]:.6f} 0.48")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="box", size="0.03 0.03 0.03", mass="0.1",
                      rgba=" ".join(str(v) for v in (*color, 1)), friction="1 0.005 0.0001")
        qpos.extend((*xy, 0.48, 1, 0, 0, 0))
    for key in root.findall("./keyframe/key"):
        key.set("qpos", " ".join(str(v) for v in qpos))
    # The generated file may live outside the models directory.
    for element in root.iter():
        if "file" in element.attrib:
            element.set("file", str(source.parent / element.get("file")))
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(args.output, encoding="unicode")
    check = mujoco.MjModel.from_xml_path(str(args.output))
    assert check.nq == len(qpos)
    print(f"Generated {args.count} cubes: {args.output}")


if __name__ == "__main__":
    main()
