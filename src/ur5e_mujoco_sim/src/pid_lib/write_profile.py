#!/usr/bin/env python3
"""Generate ros2_control's effort-PID profile from the shared gains."""

import sys
import tempfile
from pathlib import Path

import yaml

if __package__:
    from .controller import ARM_GAINS
else:
    from controller import ARM_GAINS


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_profile.py <base_ros2_controllers.yaml>")
    base = Path(sys.argv[1])
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    params = data["ur5e_controller"]["ros__parameters"]
    expected = set(params["joints"])
    if expected != set(ARM_GAINS):
        raise RuntimeError("PID gain names do not match the arm controller joints")
    params["command_interfaces"] = ["effort"]
    params["open_loop_control"] = False
    params["gains"] = {
        name: {
            "p": gain.p, "i": gain.i, "d": gain.d,
            "i_clamp": gain.i_clamp, "ff_velocity_scale": 0.0,
        }
        for name, gain in ARM_GAINS.items()
    }
    output = Path(tempfile.mkdtemp(prefix="ur5e_pid_")) / "ros2_controllers_pid.yaml"
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
