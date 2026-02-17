#!/usr/bin/env python3
# Two-stage contact experiment (NO gravity compensation)
# Plan B: Add I term on POSITION ONLY (anti-windup), freeze integrator during press
# 1) Move to goal pose (min-jerk + SLERP)  [PI-D in task space]
# 2) Wait at goal                           [PI-D in task space]
# 3) Press down (penetration ramp + hold)   [PD in task space; integrator frozen]
# Logging:
#   log_root/
#     YYYY-mm-dd_HH-MM-SS/
#        meta.txt
#        log.npz
#        plots/
#           z.png
#           tau_norm.png
#           pos_xyz.png
#           Fz.png

import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pylibfranka import Robot, Torques

# --------------------------------------------------
# SE(3) helpers (Franka column-major 16 format)
# --------------------------------------------------

def p_from_T(T):
    return np.array([T[12], T[13], T[14]], dtype=float)

def R_from_T(T):
    return np.array([
        [T[0],  T[4],  T[8]],
        [T[1],  T[5],  T[9]],
        [T[2],  T[6],  T[10]],
    ], dtype=float)

def quat_from_R(R):
    # robust enough for typical robot orientations
    t = np.trace(R)
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
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
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=float)

def slerp(q0, q1, s):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = q0 + s*(q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    theta = theta0 * s
    return (np.sin(theta0-theta)/sin0)*q0 + (np.sin(theta)/sin0)*q1

def log_SO3(R):
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    return (
        theta / (2.0*np.sin(theta)) *
        np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1],
        ], dtype=float)
    )

def minimum_jerk(t, T):
    if t <= 0.0: return 0.0
    if t >= T:   return 1.0
    s = t / T
    return 10*s**3 - 15*s**4 + 6*s**5

def clamp(x, lim):
    return np.clip(x, -lim, lim)

def rate_limit_tau(tau_des, tau_prev, dt, max_tau_rate):
    # per-joint |dτ/dt| <= max_tau_rate [Nm/s]
    max_step = max_tau_rate * dt
    return tau_prev + np.clip(tau_des - tau_prev, -max_step, +max_step)

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", type=str, default="10.90.90.10")

    parser.add_argument(
        "--goal",
        type=float,
        nargs=16,
        default=[0.7074, -0.7068, 0.0001, 0.0, -0.7068, -0.7074, 0.0001, 0.0, -0.0, -0.0001, -1.0, 0.0, 0.3071, -0.0, 0.5904, 1.0],
        #default=[0.7074, -0.7068, 0.0001, 0.0, -0.7068, -0.7074, 0.0001, 0.0, -0.0, -0.0001, -1.0, 0.0, 0.5005, 0.0133, 0.0866, 1.0]
    )  

    # timing
    parser.add_argument("--Tmove", type=float, default=4.0)
    parser.add_argument("--wait_time", type=float, default=1.0)
    parser.add_argument("--penetration", type=float, default=-0.01, help="m, negative = press down")
    parser.add_argument("--ramp_time", type=float, default=3.0)
    parser.add_argument("--hold_time", type=float, default=8.0)

    # impedance gains
    parser.add_argument("--Kp_xyz", type=float, default=600.0, help="N/m")
    parser.add_argument("--Dp_xyz", type=float, default=70.0,  help="Ns/m")
    parser.add_argument("--Kp_ori", type=float, default=20.0,  help="Nm/rad")
    parser.add_argument("--Dp_ori", type=float, default=6.0,   help="Nms/rad")

    # integrator (position only) for better steady-state accuracy during move+wait
    parser.add_argument("--Ki_x", type=float, default=80.0,  help="N/(m*s)")
    parser.add_argument("--Ki_y", type=float, default=80.0,  help="N/(m*s)")
    parser.add_argument("--Ki_z", type=float, default=120.0, help="N/(m*s)  (often larger for vertical)")
    parser.add_argument("--eint_max", type=float, default=0.03,
                        help="Integrator clamp on ∫ep dt [m*s] (anti-windup)")

    # safety / smoothness
    parser.add_argument("--tau_max", type=float, default=25.0, help="Nm per joint abs limit")
    parser.add_argument("--max_tau_rate", type=float, default=350.0, help="Nm/s per joint rate limit")

    # logging structure
    parser.add_argument("--log_root", type=str, default="logs",
                        help="Root folder. Each run creates logs/<timestamp>/...")

    args = parser.parse_args()

    # -------------------------
    # Create per-run folder
    # -------------------------
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.abspath(os.path.join(args.log_root, run_stamp))
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # record params
    with open(os.path.join(run_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write("Two-stage contact experiment (NO gravity)\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write(f"run_dir: {run_dir}\n")

    # -------------------------
    # Robot setup
    # -------------------------
    robot = Robot(args.ip)
    print("WARNING: Robot will move to goal, wait, then press down.")
    print(f"Logs will be saved to:\n  {run_dir}")
    input("Press Enter to continue...")

    active = robot.start_torque_control()
    state0, _ = active.readOnce()
    model = robot.load_model()

    # initial pose
    T_init = state0.O_T_EE.copy()
    p_init = p_from_T(T_init)
    R_init = R_from_T(T_init)
    q_init = quat_from_R(R_init)

    # goal pose
    T_goal = np.array(args.goal, dtype=float)
    p_goal = p_from_T(T_goal)
    R_goal = R_from_T(T_goal)
    q_goal = quat_from_R(R_goal)

    # gains
    Kp_pos = np.diag([args.Kp_xyz, args.Kp_xyz, args.Kp_xyz])
    Dp_pos = np.diag([args.Dp_xyz, args.Dp_xyz, args.Dp_xyz])
    Kp_ori = float(args.Kp_ori)
    Dp_ori = float(args.Dp_ori)

    Ki_pos = np.diag([args.Ki_x, args.Ki_y, args.Ki_z])
    e_int = np.zeros(3, dtype=float)               # ∫ ep dt  (position only)
    e_int_max = float(args.eint_max)

    tau_max = np.array([args.tau_max]*7, dtype=float)
    tau_prev = np.zeros(7, dtype=float)

    # time markers
    t = 0.0
    t_move_end = args.Tmove
    t_wait_end = t_move_end + args.wait_time
    t_press_end = t_wait_end + args.ramp_time + args.hold_time

    # logs (in-memory, then saved to log.npz once)
    ts = []
    p_ref_log = []
    p_act_log = []
    eo_log = []
    ep_log = []
    e_int_log = []
    tau_log = []
    Fz_log = []   # uses Franka estimator (for diagnostics / KF tuning)
    phase_log = []

    def phase_id(tt):
        if tt <= t_move_end: return 1  # move
        if tt <= t_wait_end: return 2  # wait
        return 3                        # press

    try:
        robot.set_collision_behavior([80]*7, [80]*7, [80]*6, [80]*6)

        while True:
            state, dt_msg = active.readOnce()
            dt = dt_msg.to_sec()
            t += dt

            dq = np.array(state.dq, dtype=float)

            T = state.O_T_EE.copy()
            p = p_from_T(T)
            R = R_from_T(T)

            J = np.array(model.zero_jacobian(state), dtype=float).reshape((6, 7), order='F')
            coriolis = np.array(model.coriolis(state), dtype=float)
            dx = J @ dq  # [v; w]

            ph = phase_id(t)

            # -------------------------
            # Reference generation (continuous)
            # -------------------------
            if ph == 1:
                s = minimum_jerk(t, args.Tmove)
                p_ref = p_init + s*(p_goal - p_init)
                q_ref = slerp(q_init, q_goal, s)
                R_ref = R_from_quat(q_ref)
            elif ph == 2:
                p_ref = p_goal.copy()
                R_ref = R_goal
            else:
                t2 = t - t_wait_end
                alpha = min(1.0, max(0.0, t2 / args.ramp_time))
                p_ref = p_goal.copy()
                p_ref[2] = p_goal[2] + args.penetration * alpha
                R_ref = R_goal

            # -------------------------
            # Errors
            # -------------------------
            ep = (p_ref - p)                      # 3
            eo = log_SO3(R_ref @ R.T)             # 3

            # -------------------------
            # Plan B integrator:
            # - integrate only during move+wait (ph 1/2)
            # - freeze during press (ph 3) to avoid force drift / windup
            # -------------------------
            if ph in (1, 2):
                e_int += ep * dt
                e_int = np.clip(e_int, -e_int_max, +e_int_max)

            # -------------------------
            # Task-space wrench (PI-D on position, PD on orientation)
            # -------------------------
            F = np.zeros(6, dtype=float)
            F[:3] = (Kp_pos @ ep) - (Dp_pos @ dx[:3]) + (Ki_pos @ e_int)
            F[3:] = (Kp_ori * eo) - (Dp_ori * dx[3:])

            # Joint torques (NO gravity)
            tau_des = (J.T @ F) + coriolis
            tau_des = clamp(tau_des, tau_max)

            # Rate limiting to avoid discontinuity reflex
            tau_cmd = rate_limit_tau(tau_des, tau_prev, dt, args.max_tau_rate)
            tau_cmd = clamp(tau_cmd, tau_max)

            # Send command
            active.writeOnce(Torques(tau_cmd.tolist()))
            tau_prev = tau_cmd

            # -------------------------
            # Logging
            # -------------------------
            ts.append(t)
            p_ref_log.append(p_ref.copy())
            p_act_log.append(p.copy())
            ep_log.append(ep.copy())
            eo_log.append(eo.copy())
            e_int_log.append(e_int.copy())
            tau_log.append(tau_cmd.copy())
            phase_log.append(ph)

            # diagnostic (Franka estimator)
            try:
                Fz_log.append(float(state.O_F_ext_hat_K[2]))
            except Exception:
                Fz_log.append(np.nan)

            # IMPORTANT: stop AFTER writeOnce
            if t >= t_press_end:
                break

    except Exception as e:
        print(f"Control exception: {e}")

    finally:
        robot.stop()

        # -------------------------
        # Save log.npz
        # -------------------------
        t_arr = np.array(ts)
        p_ref_arr = np.vstack(p_ref_log) if len(p_ref_log) else np.zeros((0, 3))
        p_act_arr = np.vstack(p_act_log) if len(p_act_log) else np.zeros((0, 3))
        ep_arr = np.vstack(ep_log) if len(ep_log) else np.zeros((0, 3))
        eo_arr = np.vstack(eo_log) if len(eo_log) else np.zeros((0, 3))
        e_int_arr = np.vstack(e_int_log) if len(e_int_log) else np.zeros((0, 3))
        tau_arr = np.vstack(tau_log) if len(tau_log) else np.zeros((0, 7))
        phase_arr = np.array(phase_log, dtype=int)
        Fz_arr = np.array(Fz_log, dtype=float)

        np.savez(
            os.path.join(run_dir, "log.npz"),
            t=t_arr,
            p_ref=p_ref_arr,
            p_act=p_act_arr,
            ep=ep_arr,
            eo=eo_arr,
            e_int=e_int_arr,
            tau=tau_arr,
            phase=phase_arr,
            Fz=Fz_arr,
        )

        # -------------------------
        # Plots
        # -------------------------
        if len(t_arr) == 0:
            print("No samples collected; nothing to plot.")
            return

        # XYZ
        plt.figure()
        plt.plot(t_arr, p_ref_arr, "--")
        plt.plot(t_arr, p_act_arr)
        plt.xlabel("time [s]")
        plt.ylabel("position [m]")
        plt.title("XYZ position: reference (--) vs actual")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "pos_xyz.png"), dpi=200)
        plt.close()

        # Z only
        plt.figure()
        plt.plot(t_arr, p_ref_arr[:, 2], "--", label="z_ref")
        plt.plot(t_arr, p_act_arr[:, 2], label="z_act")
        plt.legend()
        plt.xlabel("time [s]")
        plt.ylabel("z [m]")
        plt.title("Z position")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "z.png"), dpi=200)
        plt.close()

        # torque norm
        plt.figure()
        plt.plot(t_arr, np.linalg.norm(tau_arr, axis=1))
        plt.xlabel("time [s]")
        plt.ylabel("||tau|| [Nm]")
        plt.title("Joint torque norm (commanded)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "tau_norm.png"), dpi=200)
        plt.close()

        # diagnostic Fz (Franka estimator)
        plt.figure()
        plt.plot(t_arr, Fz_arr)
        plt.xlabel("time [s]")
        plt.ylabel("Fz [N]")
        plt.title("Fz (Franka O_F_ext_hat_K[2]) - diagnostic only")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "Fz.png"), dpi=200)
        plt.close()

        print("\nSaved:")
        print(f"  run_dir : {run_dir}")
        print(f"  log     : {os.path.join(run_dir, 'log.npz')}")
        print(f"  plots   : {plots_dir}")

if __name__ == "__main__":
    main()