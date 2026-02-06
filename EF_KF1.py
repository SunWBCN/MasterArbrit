#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from pylibfranka import Robot, Torques
from filterpy.kalman import KalmanFilter
from scipy.linalg import expm


# ============================================================
# Utils
# ============================================================

def create_experiment_dir(root="experiments"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(root, timestamp)
    os.makedirs(exp_dir, exist_ok=False)
    return exp_dir


# ============================================================
# Kalman Momentum Observer (OFFLINE)
# ============================================================

class KalmanMomentumObserver:
    """
    x = [p; tau_ext]
    p_dot = tau_cmd + c - g + tau_ext
    tau_ext_dot = 0
    y = p = M(q) dq
    """

    def __init__(self, dt, nv=7, Q_scale=1e-3, R_scale=1e-3):
        self.nv = nv
        dt = float(dt)

        A = np.block([
            [np.zeros((nv, nv)), np.eye(nv)],
            [np.zeros((nv, nv)), np.zeros((nv, nv))]
        ])
        B = np.vstack([
            np.eye(nv),
            np.zeros((nv, nv))
        ])
        C = np.hstack([
            np.eye(nv),
            np.zeros((nv, nv))
        ])

        Qc = np.eye(2 * nv) * Q_scale
        R = np.eye(nv) * R_scale

        n = A.shape[0]
        H = np.block([
            [A, Qc],
            [np.zeros((n, n)), -A.T]
        ])

        Hd = expm(H * dt)
        Ad = expm(A * dt)
        Qd = Hd[:n, n:] @ Ad.T

        self.kf = KalmanFilter(dim_x=2 * nv, dim_z=nv)
        self.kf.x = np.zeros(2 * nv)
        self.kf.P = np.eye(2 * nv)
        self.kf.F = Ad
        self.kf.B = B
        self.kf.H = C
        self.kf.Q = Qd
        self.kf.R = R

    def step(self, u, p_meas):
        self.kf.predict(u=u)
        self.kf.update(p_meas)
        return self.kf.x[self.nv:].copy()


# ============================================================
# Phase 1: Online run (SAFE, 1 kHz)
# ============================================================

def run_and_log(robot_ip, seconds, damping, tau_max):
    nv = 7
    robot = Robot(robot_ip)

    print("⚠️ WARNING: Torque control will move the robot!")
    input("确认环境安全后按 Enter 开始运行...")

    active = robot.start_torque_control()

    log_t, log_q, log_dq, log_tau_cmd, log_tau_J = [], [], [], [], []

    tau_cmd = [0.0] * nv
    t_robot = 0.0

    try:
        while t_robot < seconds:
            state, dt = active.readOnce()
            dt_robot = dt.to_sec() if hasattr(dt, "to_sec") else float(dt)
            t_robot += dt_robot

            dq = state.dq
            for i in range(nv):
                v = -damping * float(dq[i])
                v = max(min(v, tau_max), -tau_max)
                tau_cmd[i] = v

            active.writeOnce(Torques(tau_cmd))

            log_t.append(t_robot)
            log_q.append(np.array(state.q, dtype=float))
            log_dq.append(np.array(state.dq, dtype=float))
            log_tau_cmd.append(np.array(tau_cmd, dtype=float))
            log_tau_J.append(np.array(state.tau_J, dtype=float))

        print(f"[OK] Online run finished: t = {t_robot:.3f}s")

    except Exception as e:
        print("[ABORTED]", e)

    finally:
        try:
            robot.stop()
        except Exception:
            pass

    return {
        "t": np.array(log_t),
        "q": np.vstack(log_q),
        "dq": np.vstack(log_dq),
        "tau_cmd": np.vstack(log_tau_cmd),
        "tau_J": np.vstack(log_tau_J),
    }


# ============================================================
# Phase 2: Offline Kalman + Plot
# ============================================================

def offline_analysis(robot_ip, data, exp_dir, dt_kf):
    nv = 7

    t = data["t"]
    q = data["q"]
    dq = data["dq"]
    tau_cmd = data["tau_cmd"]
    tau_J = data["tau_J"]

    # -------------------------
    # 0) sanity check
    # -------------------------
    if t.size == 0:
        print("[ERROR] No samples collected (t is empty). No figures will be generated.")
        return

    # Franka residual ALWAYS available
    tau_ext_franka = tau_J - tau_cmd

    # -------------------------
    # 1) ALWAYS save residual plots (no model needed)
    # -------------------------
    fig, axs = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
    for i in range(7):
        axs[i].plot(t, tau_ext_franka[:, i], label="Franka residual")
        axs[i].set_ylabel(f"J{i+1} [Nm]")
        axs[i].grid(True)
        if i == 0:
            axs[i].legend()
    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Franka residual torque (tau_J - tau_cmd)")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "franka_residual_all_joints.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, tau_ext_franka[:, 0], label="Franka residual")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint 1 torque [Nm]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "franka_residual_joint1.png"), dpi=200)
    plt.close()

    print("[OK] Residual figures saved.")

    # -------------------------
    # 2) TRY KF plots (may fail due to model API)
    # -------------------------
    try:
        robot = Robot(robot_ip)
        model = robot.load_model()
        kf = KalmanMomentumObserver(dt=dt_kf, nv=nv)

        tau_ext_kf = []
        print("[INFO] Running offline Kalman filter...")

        # NOTE: your setup may NOT support model.mass(q)
        for k in range(len(t)):
            M = np.array(model.mass(q[k]), dtype=float).reshape(7, 7, order="F")
            c = np.array(model.coriolis(q[k], dq[k]), dtype=float)
            g = np.array(model.gravity(q[k]), dtype=float)

            p_meas = M @ dq[k]
            u = tau_cmd[k] + c - g
            tau_ext_kf.append(kf.step(u, p_meas))

        tau_ext_kf = np.vstack(tau_ext_kf)

        # KF vs residual (all joints)
        fig, axs = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
        for i in range(7):
            axs[i].plot(t, tau_ext_franka[:, i], label="Franka residual")
            axs[i].plot(t, tau_ext_kf[:, i], label="Kalman estimate")
            axs[i].set_ylabel(f"J{i+1} [Nm]")
            axs[i].grid(True)
            if i == 0:
                axs[i].legend()
        axs[-1].set_xlabel("Time [s]")
        fig.suptitle("External Torque Estimation (KF vs Franka residual)")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "tau_all_joints.png"), dpi=200)
        plt.close()

        # joint1
        plt.figure(figsize=(10, 5))
        plt.plot(t, tau_ext_franka[:, 0], label="Franka residual")
        plt.plot(t, tau_ext_kf[:, 0], label="Kalman estimate")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint 1 torque [Nm]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "tau_joint1.png"), dpi=200)
        plt.close()

        print("[OK] KF figures saved.")

    except Exception as e:
        print("[WARN] KF skipped due to error:", repr(e))
        print("       Residual plots were saved successfully.")

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="10.90.90.10")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--damping", type=float, default=0.3)
    parser.add_argument("--tau-max", type=float, default=5.0)
    parser.add_argument("--dt-kf", type=float, default=0.005)
    args = parser.parse_args()

    exp_dir = create_experiment_dir()
    print(f"[INFO] Experiment dir: {exp_dir}")

    # ---------- Phase 1 ----------
    data = run_and_log(
        robot_ip=args.robot_ip,
        seconds=args.seconds,
        damping=args.damping,
        tau_max=args.tau_max,
    )

    # save raw data
    #np.savez(os.path.join(exp_dir, "raw_data.npz"), **data)

    # save meta info
    with open(os.path.join(exp_dir, "meta.txt"), "w") as f:
        f.write(f"robot_ip: {args.robot_ip}\n")
        f.write(f"seconds: {args.seconds}\n")
        f.write(f"damping: {args.damping}\n")
        f.write(f"tau_max: {args.tau_max}\n")
        f.write(f"dt_kf: {args.dt_kf}\n")

    # ---------- Phase 2 ----------
    offline_analysis(
        robot_ip=args.robot_ip,
        data=data,
        exp_dir=exp_dir,
        dt_kf=args.dt_kf,
    )

    print("✅ Experiment finished successfully.")


if __name__ == "__main__":
    main()
