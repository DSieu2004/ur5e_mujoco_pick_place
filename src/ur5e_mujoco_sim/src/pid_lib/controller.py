"""
Clamped PID control law used for gain design and offline comparison.

The live robot uses ros2_control's real-time JointTrajectoryController PID
adapter with the same gains exported by this package. A Python callback is
not placed in the 100 Hz actuator loop.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PIDGains:
    p: float
    i: float
    d: float
    i_clamp: float
    effort_limit: float


class PIDController:
    """Reference discrete PID with output saturation and integral anti-windup."""

    def __init__(self, gains):
        self.gains = gains
        self.integral_output = 0.0

    def reset(self):
        self.integral_output = 0.0

    def update(self, target_position, measured_position, target_velocity,
               measured_velocity, dt):
        if dt <= 0:
            raise ValueError("PID update period must be positive")
        error = target_position - measured_position
        derivative = target_velocity - measured_velocity
        proportional = self.gains.p * error
        derivative_output = self.gains.d * derivative
        tentative_integral = max(
            -self.gains.i_clamp,
            min(self.gains.i_clamp,
                self.integral_output + self.gains.i * error * dt),
        )
        tentative = proportional + derivative_output + tentative_integral
        if abs(tentative) <= self.gains.effort_limit or tentative * error < 0:
            self.integral_output = tentative_integral
        output = proportional + derivative_output + self.integral_output
        return max(-self.gains.effort_limit,
                   min(self.gains.effort_limit, output))


# Effort control runs at 100 Hz and must use lower gains than MuJoCo's former
# internal position servo. The integral term is small and clamped.
ARM_GAINS = {
    "shoulder_pan_joint": PIDGains(150.0, 1.0, 30.0, 5.0, 150.0),
    "shoulder_lift_joint": PIDGains(150.0, 1.0, 30.0, 5.0, 150.0),
    "elbow_joint": PIDGains(150.0, 1.0, 30.0, 5.0, 150.0),
    "wrist_1_joint": PIDGains(40.0, 0.25, 8.0, 1.0, 28.0),
    "wrist_2_joint": PIDGains(40.0, 0.25, 8.0, 1.0, 28.0),
    "wrist_3_joint": PIDGains(40.0, 0.25, 8.0, 1.0, 28.0),
}
