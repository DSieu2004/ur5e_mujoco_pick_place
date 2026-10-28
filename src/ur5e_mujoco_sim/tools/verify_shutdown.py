#!/usr/bin/env python3
"""Exercise real Ctrl-C and X11 window-close shutdowns of the complete ROS stack."""

import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from ament_index_python.packages import get_package_prefix


class ClientMessage(ctypes.Structure):
    """X11 client-message ABI used by the normal window-manager close action."""

    _fields_ = [
        ('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
        ('display', ctypes.c_void_p), ('window', ctypes.c_ulong),
        ('message_type', ctypes.c_ulong), ('format', ctypes.c_int),
        ('data', ctypes.c_long * 5)]


class XEvent(ctypes.Union):
    """X11 event union, padded to the native XEvent size."""

    _fields_ = [('message', ClientMessage), ('pad', ctypes.c_long * 24)]


def close_simulator_window(name):
    info = subprocess.run(['xwininfo', '-name', name], capture_output=True, text=True,
                          check=True)
    match = re.search(r'Window id: (0x[0-9a-fA-F]+)', info.stdout)
    if match is None:
        raise RuntimeError(f'Cannot identify simulator window {name!r}')
    window = int(match.group(1), 16)
    x11 = ctypes.CDLL('libX11.so.6')
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XSendEvent.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long, ctypes.POINTER(XEvent)]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError('A working X11 display is required')
    try:
        event = XEvent()
        event.message.type = 33
        event.message.display = display
        event.message.window = window
        event.message.message_type = x11.XInternAtom(display, b'WM_PROTOCOLS', 0)
        event.message.format = 32
        event.message.data[0] = x11.XInternAtom(display, b'WM_DELETE_WINDOW', 0)
        if not x11.XSendEvent(display, window, 0, 0, ctypes.byref(event)):
            raise RuntimeError('Window manager did not accept the close message')
        x11.XFlush(display)
    finally:
        x11.XCloseDisplay(display)


def run_case(directory, mode, index, expected_library, active_task=False, preview=False,
             launch_file='vision_pick_place.launch.py'):
    path = directory / f'{index:02d}_{mode}.log'
    command = ['ros2', 'launch', 'ur5e_mujoco_sim', launch_file]
    if launch_file == 'vision_pick_place.launch.py':
        command += [f'run_task:={str(active_task).lower()}', f'preview:={str(preview).lower()}']
    window_name = ('MuJoCo : ur5e_realsense_vision' if launch_file.endswith('.py')
                   else 'MuJoCo : ur5e_robotiq_scene')
    with path.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 30
            native_pid = None
            while time.monotonic() < deadline:
                text = path.read_text()
                match = re.search(
                    r'\[ros2_control_node-\d+\]: process started with pid \[(\d+)\]', text)
                ready = re.search(r'\[spawner-\d+\]: process has finished cleanly', text)
                if active_task:
                    ready = ready and 'Timed trajectory camera scan overview' in text
                if match and ready:
                    native_pid = int(match.group(1))
                    break
                if process.poll() is not None:
                    raise RuntimeError(f'Launch exited during startup: {path}')
                time.sleep(.1)
            if native_pid is None:
                raise RuntimeError(f'Controller startup timed out: {path}')
            mappings = Path(f'/proc/{native_pid}/maps').read_text()
            if str(expected_library) not in mappings:
                raise RuntimeError('Simulator did not load the patched workspace library')
            # Let camera timers execute before destroying their callback groups.
            time.sleep(1.0)
            if mode == 'ctrl_c':
                process.send_signal(signal.SIGINT)
            else:
                close_simulator_window(window_name)
            process.wait(timeout=25)
            text = path.read_text()
            errors = re.findall(r'exit code (-?\d+)', text)
            crashed = (any(int(code) not in (0, -2, 130) for code in errors)
                       or any(marker in text for marker in
                              ('Segmentation fault', 'SIGSEGV', 'terminate called', 'Traceback',
                               "sending signal 'SIGTERM'", "sending signal 'SIGKILL'")))
            child_pids = [int(pid) for pid in re.findall(
                r'process started with pid \[(\d+)\]', text)]
            survivors = []
            for pid in child_pids:
                try:
                    os.kill(pid, 0)
                    survivors.append(pid)
                except ProcessLookupError:
                    pass
            partial_saved = True
            if active_task:
                recording = re.search(r'Recording PID feedback and controller commands in (.+)',
                                      text)
                partial_saved = False
                if recording:
                    run_directory = Path(recording.group(1).strip())
                    summary_path = run_directory / 'vision_summary.json'
                    if summary_path.is_file():
                        summary = json.loads(summary_path.read_text())
                        partial_saved = (summary.get('interrupted') is True
                                         and summary.get('completed') is False
                                         and (run_directory / 'controller_response.csv').is_file())
            result = {'mode': mode, 'launch_file': launch_file,
                      'active_task': active_task, 'preview': preview,
                      'partial_results_saved': partial_saved,
                      'log': str(path.resolve()),
                      'launch_returncode': process.returncode,
                      'child_exit_codes': [int(code) for code in errors],
                      'surviving_children': survivors,
                      'patched_library': str(expected_library),
                      'passed': not crashed and not survivors and partial_saved
                      and process.returncode in (0, -2, 130)}
            return result
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=3, help='repetitions of each close method')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--active-task', action='store_true', help='interrupt a moving task')
    parser.add_argument('--preview', action='store_true', help='also exercise the camera viewer')
    parser.add_argument('--launch-file', default='vision_pick_place.launch.py',
                        choices=('vision_pick_place.launch.py', 'mujoco.launch.xml',
                                 'mujoco_pid.launch.xml'))
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    if args.launch_file.endswith('.xml') and (args.active_task or args.preview):
        parser.error('--active-task and --preview apply to the vision launch')
    workspace = Path(__file__).resolve().parents[3]
    prefix = Path(get_package_prefix('mujoco_ros2_control'))
    if not prefix.resolve().is_relative_to(workspace / 'install'):
        parser.error('Source the workspace overlay containing the patched MuJoCo driver first')
    # Never close an unrelated simulator already running on the desktop.
    for title in ('MuJoCo : ur5e_realsense_vision', 'MuJoCo : ur5e_robotiq_scene'):
        existing = subprocess.run(['xwininfo', '-name', title], capture_output=True)
        if existing.returncode == 0:
            parser.error('Close the existing MuJoCo simulation before running this test')
    output = args.output or workspace / 'trajectory_runs' / (
        'shutdown_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    library = (prefix / 'lib/libmujoco_ros2_control.so').resolve()
    results = []
    for index in range(args.repeat):
        for mode in ('ctrl_c', 'window_close'):
            result = run_case(output, mode, len(results), library, args.active_task, args.preview,
                              args.launch_file)
            results.append(result)
            print(f"{index + 1}/{args.repeat} {mode}: "
                  f"{'PASS' if result['passed'] else 'FAIL'}", flush=True)
            (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    passed = all(result['passed'] for result in results)
    print(f"{sum(result['passed'] for result in results)}/{len(results)} shutdown checks passed; "
          f"results: {output}")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
