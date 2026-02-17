#!/usr/bin/env python3
# Copyright (c) 2025 Franka Robotics GmbH
# Apache-2.0

import argparse
import time
import numpy as np
import os
import matplotlib.pyplot as plt

from pylibfranka import Robot, Torques

def p_from_T_colmajor(T):
    return np.array([T[12], T[13], T[14]], dtype=float)

def R_from_T_colmajor(T):
    return np.array([
        [T[0],  T[4],  T[8]],
        [T[1],  T[5],  T[9]],
        [T[2],  T[6],  T[10]],
    ], dtype=float)

def set_R_in_T_colmajor(T, R):
    T[0],  T[1],  T[2]   = R[0,0], R[1,0], R[2,0]
    T[4],  T[5],  T[6]   = R[0,1], R[1,1], R[2,1]
    T[8],  T[9],  T[10]  = R[0,2], R[1,2], R[2,2]

def quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def R_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def slerp(q0, q1, s):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    theta = theta0 * s
    return (
        np.sin(theta0 - theta) / sin0 * q0 +
        np.sin(theta) / sin0 * q1
    )


def log_SO3(R):
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    return (
        theta / (2*np.sin(theta)) *
        np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1],
        ])
    )

def pose_error(T_des, T):
    ep = p_from_T_colmajor(T_des) - p_from_T_colmajor(T)
    eo = log_SO3(R_from_T_colmajor(T_des) @ R_from_T_colmajor(T).T)
    return np.hstack([ep, eo])

def minimum_jerk(t, T):
    if t <= 0: return 0.0
    if t >= T: return 1.0
    s = t / T
    return 10*s**3 - 15*s**4 + 6*s**5

def clamp(x, max_abs):
    return np.clip(x, -max_abs, +max_abs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.90.90.10")
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument(
        "--goal",
        type=float,
        nargs=16,
        default=[0.7074, -0.7068, 0.0001, 0.0, -0.7068, -0.7074, 0.0001, 0.0, -0.0, -0.0001, -1.0, 0.0, 0.3071, -0.0, 0.5904, 1.0],
        #default=[0.6377, -0.4868, -0.5969, 0.0, -0.6708, -0.732, -0.1196, 0.0, -0.3787, 0.4767, -0.7933, 0.0, 0.4905, 0.035, 0.2627, 1.0]
    )    
    parser.add_argument("--plot", action="store_true", help="Save plots after motion")
    parser.add_argument("--outdir", type=str, default="cartesian_impedance", help="Output directory for plots/logs")
    args = parser.parse_args()

    robot = Robot(args.ip)

    print("WARNING: Torque control will move the robot!")
    input("Press Enter to continue...")

    active = robot.start_torque_control()

    state, _ = active.readOnce()
    model = robot.load_model()
    initial_cartesian_pose = state.O_T_EE.copy()
    target_cartesian_pose = np.array(args.goal, dtype=float)

    R0 = R_from_T_colmajor(initial_cartesian_pose) # R0: initial R
    R1 = R_from_T_colmajor(target_cartesian_pose) # R1: target R

    q0 = quat_from_R(R0)
    q1 = quat_from_R(R1)

    # Cartesian impedance parameters
    Kp = np.diag([800, 800, 800, 15, 15, 15])
    Dp = np.diag([40, 40, 40, 5, 5, 5])
    tau_max = np.array([40]*7)

    # logging
    ts = []
    p_des_log, p_act_log = [], []
    ep_log, eo_log = [], []
    dx_log, tau_log = [], []

    # time
    t = 0.0
    T = args.T
    hold_time = 5.0

    # slow integral only in z
    #Ki = 3.0
    Ki = [60.0, 60.0, 60.0, 80.0, 40.0, 40.0]
    e_int = 0.0
    e_int_max = 0.03

    try:
        # Set default behavior
        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        while True:
            state, dt = active.readOnce()
            t += dt.to_sec()

            q = np.array(state.q)
            dq = np.array(state.dq)
            current_cartesian_pose = state.O_T_EE.copy()

            if t < T:
                s = minimum_jerk(t, T)
                new_cartesian_pose = initial_cartesian_pose.copy()
                for i in (12, 13, 14):
                    new_cartesian_pose[i] = initial_cartesian_pose[i] + s*(target_cartesian_pose[i]-initial_cartesian_pose[i])
                R = R_from_quat(slerp(q0, q1, s))
                set_R_in_T_colmajor(new_cartesian_pose, R)
            else:
                new_cartesian_pose = target_cartesian_pose

            e = pose_error(new_cartesian_pose, current_cartesian_pose)

            J = np.array(model.zero_jacobian(state), dtype=float).reshape((6, 7), order='F')
            coriolis = np.array(model.coriolis(state))
            #gravity  = np.array(model.gravity(state))
            dx = J @ dq

            # slow integral
            dt_e = dt.to_sec()
            e_int += e * dt_e 
            e_int = np.clip(e_int, -e_int_max, e_int_max)

            F = Kp @ e - Dp @ dx
            F += Ki * e_int
            tau = J.T @ F + coriolis
            #alpha = min(1.0, t / T)
            #tau = J.T @ F + coriolis + alpha * gravity
            tau = clamp(tau, tau_max)

            # log
            ts.append(t)
            p_des_log.append(p_from_T_colmajor(new_cartesian_pose))
            p_act_log.append(p_from_T_colmajor(current_cartesian_pose))
            ep_log.append(e[:3])
            eo_log.append(e[3:])
            dx_log.append(dx)
            tau_log.append(tau)

            active.writeOnce(Torques(tau.tolist()))

            if t >= args.T + hold_time:
                break

    except KeyboardInterrupt:
        print("\nStopping robot...")
        #robot.stop()

    finally:
        robot.stop()

        os.makedirs(args.outdir, exist_ok=True)
        np.savez(
            f"{args.outdir}/log.npz",
            t=np.array(ts),
            p_des=np.vstack(p_des_log),
            p_act=np.vstack(p_act_log),
            ep=np.vstack(ep_log),
            eo=np.vstack(eo_log),
            dx=np.vstack(dx_log),
            tau=np.vstack(tau_log),
        )

        if args.plot:
            t = np.array(ts)
            p_des = np.vstack(p_des_log)
            p_act = np.vstack(p_act_log)
            ep = np.vstack(ep_log)
            eo = np.vstack(eo_log)

            plt.figure()
            plt.plot(t, p_des, "--")
            plt.plot(t, p_act)
            plt.title("Position: desired vs actual")
            plt.savefig(f"{args.outdir}/pos.png", dpi=200)

            plt.figure()
            plt.plot(t, eo)
            plt.title("Orientation error logSO(3)")
            plt.savefig(f"{args.outdir}/ori_error.png", dpi=200)

            plt.figure()
            plt.plot(t, np.linalg.norm(ep, axis=1), label="||ep||")
            plt.plot(t, np.linalg.norm(eo, axis=1), label="||eo||")
            plt.legend()
            plt.title("Error norms")
            plt.savefig(f"{args.outdir}/error_norm.png", dpi=200)

        print(f"logs & plots saved in: {args.outdir}")

    return 0

if __name__ == "__main__":
    main()