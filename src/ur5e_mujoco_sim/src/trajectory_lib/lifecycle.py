"""Keep the ROS context alive until task cancellation and recording finish."""

import signal

import rclpy
from rclpy.signals import SignalHandlerOptions


def init_task_context():
    """Turn stop signals into an interruption before shutting down ROS handles."""
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
