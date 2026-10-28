# RealSense D435i wrist camera

These OBJ meshes and their colours are copied from MuJoCo Menagerie,
commit `6da8b971844845de6ab27315aacc3a3fea4ddb8e`:
[realsense_d435i](https://github.com/google-deepmind/mujoco_menagerie/tree/6da8b971844845de6ab27315aacc3a3fea4ddb8e/realsense_d435i).
Menagerie derived them from the manufacturer's public RealSense D435 mesh.
See [upstream attribution](UPSTREAM.md) and [Apache-2.0 license](LICENSE).

The real product is documented by [Intel](https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html)
and its [ROS description](https://github.com/realsenseai/realsense-ros/tree/ros2-master/realsense2_description).
The approximate 72 g mass is from that description; simulation inertia is a
uniform-box approximation, not a measured manufacturer inertia.

The mesh has a 90 x 25 x 25 mm housing, with its front at local Z=0 looking
towards +Z. A compact saddle attaches to a collar around the wrist flange,
just behind the gripper body, following the supplied wrist-mounted camera
reference. The housing centre is 68 mm along tool0 X; its front plane is
145 mm behind the pinch point. The short riser and rear plate replace the
previous 140 mm cantilever. The housing is rotated 90 degrees about tool0 Z
so its long dimension runs across the gripper jaws. Its lens looks along
tool0 +Z, towards the table when the gripper points down. This is a custom
simulated saddle, not the manufacturer's mounting kit or the exact camera
shown in the reference photograph. The bracket mass is approximated as 40 g.
The simulated optical plane is at mesh `(0.0325, 0, 0.001)` m.
It is 1 mm outside the front face to prevent self-occlusion in rendering.
This gives `tool0 -> wrist_camera_color_optical_frame` translation
`(0.068, 0.0325, -0.144)` m and a 90-degree Z rotation. Both scene and URDF
use it. Launch validates these transforms and rejects stale generated scenes.

The overview camera centre is `(0.50, -0.0925, 1.20)` m in world. The wide
image axis spans the table and tray together. The corresponding tool0
position is `(0.432, -0.06, 1.056)` m with a top-down Rx(pi) orientation.
The arm keeps this downward tool direction during work and the return home.

RGB and aligned ideal depth share a 42 degree vertical FOV at 640 x 360,
published at 5 Hz. This represents the colour projection, rather than the
wider native stereo-depth FOV. No IMU, stereo noise, distortion or real
hand-eye calibration is simulated.
