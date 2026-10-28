#!/usr/bin/env python3
"""
Render labelled MuJoCo images, then train a one-class YOLO box detector.

Simulator geometry/segmentation is used ONLY to generate offline labels.
The running detector consumes RGB, depth, CameraInfo and TF, never body poses.
"""

import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


PACKAGE = Path(__file__).resolve().parents[1]


def generate_dataset(directory, samples, seed, relabel_only=False, first_index=0, last_index=None):
    directory = Path(directory).resolve()
    scene = PACKAGE / "models/ur5e_robotiq_vision_scene.xml"
    root = ET.parse(scene).getroot()
    for element in root.iter():
        if "file" in element.attrib:
            element.set("file", str(scene.parent / element.get("file")))
    ET.SubElement(root.find("worldbody"), "camera", name="training_camera",
                  pos="0.5 -0.28 0.98", quat="1 0 0 0", fovy="42",
                  resolution="640 360")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # Multisampled ID colours blend at silhouettes and can decode to unrelated
    # geom IDs. Disable MSAA, then retain only the largest connected component.
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    # Training labels are found from physical geometry, without colour names.
    cubes = []
    for bid in range(model.nbody):
        jid = model.body_jntadr[bid]
        gids = np.where(model.geom_bodyid == bid)[0]
        if jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            cubes.append((model.jnt_qposadr[jid], gids))
    cam_id = model.camera("training_camera").id
    rng = np.random.default_rng(seed)
    renderer = mujoco.Renderer(model, height=360, width=640)
    for index in range(samples):
        split = "train" if index < int(samples * 0.8) else "val"
        image_dir = directory / "images" / split
        label_dir = directory / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        # Domain variation: counts, layout, yaw, colour, light and viewpoints.
        count = int(rng.integers(0, len(cubes) + 1))
        occupied = []
        for j, (address, gids) in enumerate(cubes):
            if j >= count:
                data.qpos[address:address + 3] = (0, 0, -3 - j)
                continue
            for _ in range(200):
                xy = rng.uniform((0.27, -0.45), (0.73, 0.25))
                # Keep objects clear of tray walls and each other.
                if -0.06 < xy[1] < 0.04:
                    continue
                if all(np.linalg.norm(xy - other) > 0.085 for other in occupied):
                    break
            occupied.append(xy)
            z = 0.50 if xy[1] > 0 else 0.48
            data.qpos[address:address + 3] = (*xy, z)
            angle = rng.uniform(-np.pi, np.pi)
            data.qpos[address + 3:address + 7] = (np.cos(angle / 2), 0, 0, np.sin(angle / 2))
            color = rng.uniform(0.05, 1.0, 3)
            if index % 5 == 0:
                color[:] = rng.uniform(0.15, 0.9)
            model.geom_rgba[gids, :3] = color
        model.cam_pos[cam_id] = rng.uniform((0.37, -0.34, 0.89), (0.63, 0.17, 1.08))
        rotation = Rotation.from_euler("xyz", (rng.uniform(-0.12, 0.12),
                                               rng.uniform(-0.12, 0.12),
                                               rng.uniform(-0.35, 0.35)))
        xyzw = rotation.as_quat()
        model.cam_quat[cam_id] = xyzw[[3, 0, 1, 2]]
        if model.nlight:
            model.light_diffuse[:] = rng.uniform(0.45, 0.95)
        mujoco.mj_forward(model, data)
        if index < first_index or (last_index is not None and index >= last_index):
            continue
        renderer.update_scene(data, camera=cam_id)
        rgb = None if relabel_only else renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        labels = []
        for _, gids in cubes:
            mask = (np.isin(segmentation[:, :, 0], gids) & (
                segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)).astype(np.uint8)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if count <= 1:
                continue
            x0, y0, width, height, area = stats[1:][np.argmax(stats[1:, cv2.CC_STAT_AREA])]
            if area < 60:
                continue
            x1, y1 = x0 + width, y0 + height
            if min(x1 - x0, y1 - y0) < 9:
                continue
            labels.append(f"0 {(x0 + x1) / 1280:.7f} {(y0 + y1) / 720:.7f} "
                          f"{(x1 - x0) / 640:.7f} {(y1 - y0) / 360:.7f}")
        if rgb is not None:
            cv2.imwrite(str(image_dir / f"{index:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        elif not (image_dir / f"{index:05d}.jpg").is_file():
            raise FileNotFoundError("Relabelling requires the corresponding rendered image")
        (label_dir / f"{index:05d}.txt").write_text("\n".join(labels) + "\n")
        if index % 100 == 0:
            print(f"Rendered {index}/{samples}", flush=True)
    renderer.close()
    config = directory / "dataset.yaml"
    config.write_text(f"path: {directory}\ntrain: images/train\n"
                      "val: images/val\nnames:\n  0: box\n")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--seed", type=int, default=2702026)
    parser.add_argument("--directory", type=Path, default=Path("vision_data"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--relabel-only", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    args = parser.parse_args()
    config = (args.directory / "dataset.yaml").resolve() if args.reuse_data else generate_dataset(
        args.directory, args.samples, args.seed, relabel_only=args.relabel_only,
        first_index=args.start_index, last_index=args.end_index)
    if args.render_only or args.relabel_only:
        return
    from ultralytics import YOLO

    initial_weights = args.directory / "yolo11n.pt"
    model = YOLO(str(initial_weights) if initial_weights.is_file() else "yolo11n.pt")
    model.train(data=str(config), epochs=args.epochs, imgsz=416, batch=16,
                device=args.device, workers=2, seed=args.seed, patience=15,
                project="vision_runs/training", name="boxes", exist_ok=True,
                plots=True, verbose=False, cache=False, amp=False)
    destination = PACKAGE / "models" / "box_detector.pt"
    shutil.copy2(model.trainer.best, destination)
    metrics = model.val(data=str(config), device=args.device, imgsz=416, plots=False)
    metadata = {"architecture": "YOLO11n", "classes": ["box"], "seed": args.seed,
                "images": args.samples, "epochs_requested": args.epochs,
                "image_size": 416, "metrics": metrics.results_dict,
                "label_method": "MuJoCo segmentation, MSAA off, largest connected component",
                "scope": ("MuJoCo RGB images of single-layer 6 cm cubes; "
                          "not trained for real hardware")}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
