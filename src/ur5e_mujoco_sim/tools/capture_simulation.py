#!/usr/bin/env python3
"""Capture only the MuJoCo X11 window, even when other desktop windows overlap it."""

import argparse
import ctypes
import json
from pathlib import Path
import re
import subprocess
import time

import cv2
import numpy as np


class XImage(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ("width", "height", "xoffset", "format")] + [
        ("data", ctypes.c_void_p)] + [(name, ctypes.c_int) for name in (
            "byte_order", "bitmap_unit", "bitmap_bit_order", "bitmap_pad", "depth",
            "bytes_per_line", "bits_per_pixel")] + [
        ("red_mask", ctypes.c_ulong), ("green_mask", ctypes.c_ulong),
        ("blue_mask", ctypes.c_ulong)]


def retime_video(path, fps):
    """Keep playback at capture wall time when X11 readback is slower than requested."""
    path = Path(path)
    temporary = path.with_name(path.stem + ".retimed.mp4")
    reader = cv2.VideoCapture(str(path))
    size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not reader.isOpened() or not writer.isOpened():
        reader.release()
        writer.release()
        raise RuntimeError("Could not preserve the video playback timing")
    count = 0
    try:
        while True:
            available, frame = reader.read()
            if not available:
                break
            writer.write(frame)
            count += 1
    finally:
        reader.release()
        writer.release()
    if not count:
        raise RuntimeError("Captured video contains no frames")
    temporary.replace(path)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()
    info = subprocess.check_output(
        ["xwininfo", "-name", "MuJoCo : ur5e_realsense_vision"], text=True)
    window = int(re.search(r"Window id: (0x[0-9a-fA-F]+)", info).group(1), 16)
    width = int(re.search(r"Width: (\d+)", info).group(1))
    height = int(re.search(r"Height: (\d+)", info).group(1))
    x11 = ctypes.CDLL("libX11.so.6")
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError("An X11 display is required to capture the simulator window")
    error_callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    error_callback = error_callback_type(lambda *_: 0)
    x11.XSetErrorHandler.argtypes = [error_callback_type]
    x11.XSetErrorHandler(error_callback)
    x11.XGetImage.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
    x11.XGetImage.restype = ctypes.POINTER(XImage)
    x11.XDestroyImage.argtypes = [ctypes.POINTER(XImage)]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames = 0
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            pointer = x11.XGetImage(display, window, 0, 0, width, height,
                                    ctypes.c_ulong(-1).value, 2)
            if not pointer:
                raise RuntimeError("The MuJoCo window is no longer available")
            native = pointer.contents
            try:
                if (native.bits_per_pixel != 32 or native.byte_order != 0
                        or native.red_mask != 0xff0000):
                    raise RuntimeError("Unsupported X11 pixel format")
                pixels = ctypes.string_at(native.data, native.bytes_per_line * native.height)
                frame = np.ndarray(
                    (native.height, native.width, 4), dtype=np.uint8,
                    buffer=pixels, strides=(native.bytes_per_line, 4, 1))[:, :, :3].copy()
            finally:
                x11.XDestroyImage(pointer)
            if args.output.suffix.lower() == ".png":
                if not cv2.imwrite(str(args.output), frame):
                    raise RuntimeError("Could not save the simulator image")
                frames = 1
                break
            size = (args.width, int(round(height * args.width / width / 2)) * 2)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
                if not writer.isOpened():
                    raise RuntimeError("Could not open the MP4 encoder")
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            frames += 1
            delay = started + frames / args.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    finally:
        elapsed = time.monotonic() - started
        if writer is not None:
            writer.release()
        x11.XCloseDisplay(display)
    if args.output.suffix.lower() != ".png" and frames:
        effective_fps = frames / elapsed
        if abs(effective_fps - args.fps) > args.fps * .05:
            retime_video(args.output, effective_fps)
        args.output.with_suffix(".json").write_text(json.dumps({
            "frames": frames, "capture_wall_time_s": elapsed,
            "requested_fps": args.fps, "effective_capture_fps": effective_fps,
            "source": "MuJoCo X11 window; original simulation speed",
        }, indent=2) + "\n")
    print(f"Saved {frames} simulator-window frames to {args.output}")


if __name__ == "__main__":
    main()
