#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

import argparse
import time

import numpy as np

from pylibfranka import ControllerMode, CartesianPose, Robot

def R_from_T_colmajor(T):
    """T: length-16, column-major. Return 3x3 rotation."""
    return np.array([
        [T[0],  T[4],  T[8]],
        [T[1],  T[5],  T[9]],
        [T[2],  T[6],  T[10]],
    ], dtype=float)

def set_R_in_T_colmajor(T, R):
    """Write 3x3 rotation R back into length-16 column-major T (in-place)."""
    T[0],  T[1],  T[2]  = R[0,0], R[1,0], R[2,0]
    T[4],  T[5],  T[6]  = R[0,1], R[1,1], R[2,1]
    T[8],  T[9],  T[10] = R[0,2], R[1,2], R[2,2]

def quat_from_R(R):
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    # Robust branch-based conversion
    t = np.trace(R)
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        # Find the major diagonal element
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)

def R_from_quat(q):
    """q = [w, x, y, z] -> rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=float)
    return R

def slerp(q0, q1, s):
    """Spherical linear interpolation between unit quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    # shortest path
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        # very small angle -> lerp + normalize
        q = q0 + s*(q1 - q0)
        return q / np.linalg.norm(q)

    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    theta = theta0 * s
    sin_theta = np.sin(theta)

    a = np.sin(theta0 - theta) / sin_theta0
    b = sin_theta / sin_theta0
    q = a*q0 + b*q1
    return q / np.linalg.norm(q)

def minimum_jerk_scaling(t, T):
    """
    Minimum jerk time-scaling.
    t: current time
    T: total duration
    """
    if t <= 0.0:
        return 0.0
    if t >= T:
        return 1.0
    tau = t / T
    return 10*tau**3 - 15*tau**4 + 6*tau**5


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.90.90.10", help="Robot IP address")
    parser.add_argument("--T", type=float, default=3.0, help="Total motion duration in seconds")
    parser.add_argument(
        "--goal",
        type=float,
        nargs=16,
        default=[0.7074, -0.7068, 0.0001, 0.0, -0.7068, -0.7074, 0.0001, 0.0, -0.0, -0.0001, -1.0, 0.0, 0.3071, -0.0, 0.5904, 1.0],
        #default=[0.6377, -0.4868, -0.5969, 0.0, -0.6708, -0.732, -0.1196, 0.0, -0.3787, 0.4767, -0.7933, 0.0, 0.4905, 0.035, 0.2627, 1.0]
    )
    args = parser.parse_args()

    # Connect to robot
    robot = Robot(args.ip)

    try:
        # Set collision behavior
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        # First move the robot to a suitable joint configuration
        print("WARNING: This example will move the robot!")
        print("Please make sure to have the user stop button at hand!")
        input("Press Enter to continue...")

        # Start cartesian pose control with external control loop
        active_control = robot.start_cartesian_pose_control(ControllerMode.CartesianImpedance)

        time_elapsed = 0.0
        T = args.T
        motion_finished = False

        robot_state, duration = active_control.readOnce()
        initial_cartesian_pose = robot_state.O_T_EE.copy()
        target_cartesian_pose = np.array(args.goal, dtype=float)

        R0 = R_from_T_colmajor(initial_cartesian_pose)
        R1 = R_from_T_colmajor(target_cartesian_pose)
        q0 = quat_from_R(R0)
        q1 = quat_from_R(R1)

        # External control loop
        while not motion_finished:
            # Read robot state and duration
            robot_state, duration = active_control.readOnce()

            # Update time
            time_elapsed += duration.to_sec()

            # Minimum jerk scaling in [0,1]
            s = minimum_jerk_scaling(time_elapsed, T)

            # Interpolate translation (keep orientation fixed)
            new_cartesian_pose = initial_cartesian_pose.copy()
            for idx in (12, 13, 14):  # x, y, z
                new_cartesian_pose[idx] = (
                    initial_cartesian_pose[idx]
                    + s * (target_cartesian_pose[idx] - initial_cartesian_pose[idx])
                )

            # SLERP

            q  = slerp(q0, q1, s)
            R  = R_from_quat(q)

            set_R_in_T_colmajor(new_cartesian_pose, R)


            # set Cartesian pose
            cartesian_pose = CartesianPose(new_cartesian_pose)

            # Set motion_finished flag to True on the last update
            if time_elapsed >= T:
                s = 1.0
                cartesian_pose.motion_finished = True
                motion_finished = True
                print("Finished motion, shutting down example")

            # Send command to robot
            active_control.writeOnce(cartesian_pose)

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1


if __name__ == "__main__":
    main()
