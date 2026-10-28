#!/usr/bin/env python3
"""
Audit recorded arm motion, including camera/tool tilt and IK-branch changes.

Uses measured joint angles and MuJoCo forward kinematics offline. No simulator
object poses are used by this audit or by the running vision task.
"""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def analyze(directory, scene):
    directory = Path(directory)
    with (directory / "controller_response.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The run contains no controller samples")
    q = np.array([[float(row[f"{name}_actual_rad"]) for name in NAMES] for row in rows])
    error = np.array([[float(row[f"{name}_error_rad"]) for name in NAMES] for row in rows])
    time_s = np.array([float(row["sim_time_s"]) for row in rows])
    time_s -= time_s[0]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    addresses = [model.joint(name).qposadr[0] for name in NAMES]
    site = model.site("ur5e/robotiq_2f85/pinch").id
    tilt = []
    for sample in q:
        data.qpos[addresses] = sample
        mujoco.mj_kinematics(model, data)
        rotation = data.site_xmat[site].reshape(3, 3)
        tilt.append(np.degrees(np.arccos(np.clip(-rotation[2, 2], -1.0, 1.0))))
    tilt = np.asarray(tilt)
    home_error = float(np.max(abs(q[-1] - model.key_qpos[0, addresses])))
    checks = {
        "elbow_stays_positive": bool(np.all(q[:, 2] >= 0.33)),
        "wrist_2_stays_on_negative_working_branch": bool(
            np.all((q[:, 4] >= -2.02) & (q[:, 4] <= -1.08))),
        "camera_tool_tilt_below_6_degrees": bool(tilt.max() <= 6.0),
        "joint_tracking_error_below_0_03_rad": bool(np.max(abs(error)) <= 0.03),
        "canonical_home_reached": home_error <= 0.05,
    }
    quality_path = directory / "motion_quality.json"
    if quality_path.exists():
        bounds = json.loads(quality_path.read_text()).get("joint_bounds_rad", {})
        checks["all_measured_joints_within_working_bounds"] = all(
            bool(np.all((q[:, i] >= bounds[name][0] - 0.03)
                        & (q[:, i] <= bounds[name][1] + 0.03)))
            for i, name in enumerate(NAMES) if name in bounds)
    report = {
        "run_directory": str(directory.resolve()), "samples": len(rows),
        "duration_s": float(time_s[-1]), "max_tool_camera_tilt_deg": float(tilt.max()),
        "worst_tilt_stage": rows[int(tilt.argmax())]["stage"],
        "max_joint_tracking_error_rad": float(np.max(abs(error))),
        "final_home_error_rad": home_error,
        "joint_ranges_rad": {name: [float(q[:, i].min()), float(q[:, i].max())]
                             for i, name in enumerate(NAMES)},
        "joint_travel_rad": dict(zip(NAMES, np.abs(np.diff(q, axis=0)).sum(axis=0).tolist())),
        "checks": checks, "passed": all(checks.values()),
    }
    return report, time_s, q, tilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--scene", type=Path, default=Path(__file__).resolve().parents[1]
                        / "models/ur5e_robotiq_vision_scene.xml")
    parser.add_argument("--verify", action="store_true",
                        help="exit nonzero if a stability criterion fails")
    args = parser.parse_args()
    current = analyze(args.run_directory, args.scene)
    report = current[0]
    data = [("Current", current, "#247a62")]
    if args.compare:
        previous = analyze(args.compare, args.scene)
        report["baseline"] = previous[0]
        data.insert(0, ("Previous", previous, "#b8564e"))
    (args.run_directory / "motion_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Measured robot motion: tool orientation and working posture")
    for label, (_, time_s, q, tilt), color in data:
        axes[0].plot(time_s, tilt, label=label, color=color, linewidth=1)
        axes[1].plot(time_s, np.degrees(q[:, 2]), color=color, linewidth=1)
        axes[2].plot(time_s, np.degrees(q[:, 4]), color=color, linewidth=1)
    axes[0].set_ylabel("Tool/camera tilt (deg)")
    axes[0].axhline(6, color="black", linestyle="--", linewidth=.7, label="6° tilt limit")
    axes[0].legend()
    axes[1].set_ylabel("Elbow angle (deg)")
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[2].set_ylabel("Wrist 2 angle (deg)")
    axes[2].axhline(-90, color="black", linestyle="--", linewidth=.7)
    axes[2].set_xlabel("Simulation time since first recorded sample (s)")
    for axis in axes:
        axis.grid(alpha=.25)
    for extension in ("png", "svg"):
        fig.savefig(args.run_directory / f"motion_comparison.{extension}", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    if args.verify and not report["passed"]:
        raise SystemExit("Motion stability criteria failed; inspect motion_validation.json")


if __name__ == "__main__":
    main()
