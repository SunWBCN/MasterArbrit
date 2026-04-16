#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

from pylibfranka import Robot, ControllerMode, JointPositions


def cosine_s_curve(q0: np.ndarray, qf: np.ndarray, t: float, T: float) -> np.ndarray:
    """Cosine S-curve time-scaling: q(t) = q0 + s(t)*(qf-q0), s(t)=0.5*(1-cos(pi*t/T))."""
    if t >= T:
        return qf
    s = 0.5 * (1.0 - np.cos(np.pi * t / T))
    return q0 + s * (qf - q0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True, help="Robot IP address (e.g., 10.90.90.10)")
    parser.add_argument("--T", type=float, default=3.0, help="Total motion duration in seconds")
    parser.add_argument("--tol", type=float, default=1e-2, help="Stop tolerance in joint space (rad)")
    parser.add_argument(
        "--goal",
        type=float,
        nargs=7,
        default=[0.0, -0.785398163397, 0.0, -2.35619449019, 0.0, 1.57079632679, 0.785398163397],
        help="7 joint targets in rad (default: Franka start pose)",
    )
    args = parser.parse_args()

    robot = Robot(args.ip)

    try:
        # Safety: collision thresholds (tune for your setup)
        robot.set_collision_behavior([20.0] * 7, [40.0] * 7, [10.0] * 6, [20.0] * 6)

        q_goal = np.array(args.goal, dtype=float)

        print("WARNING: This program will move the robot.")
        input("Press Enter to continue...")

        # Start joint position control (external loop)
        active = robot.start_joint_position_control(ControllerMode.JointImpedance)

        # Define trajectory start from current state
        state, _ = active.readOnce()
        q_start = np.array(state.q, dtype=float)

        t = 0.0
        finished = False

        while not finished:
            state, duration = active.readOnce()
            dt = duration.to_sec()
            t += dt

            q_cmd = cosine_s_curve(q_start, q_goal, t, args.T)
            cmd = JointPositions(q_cmd.tolist())

            # Finish criteria: time reached OR close enough
            if t >= args.T or np.linalg.norm(q_goal - np.array(state.q, dtype=float)) < args.tol:
                cmd.motion_finished = True
                finished = True
                print("Finished motion.")

            active.writeOnce(cmd)

        robot.stop()
        return 0

    except Exception as e:
        print(f"Error occurred: {e}")
        try:
            robot.stop()
        except Exception:
            pass
        return -1


if __name__ == "__main__":
    raise SystemExit(main())