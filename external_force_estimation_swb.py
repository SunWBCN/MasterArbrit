#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pylibfranka import Robot, Torques
from filterpy.kalman import KalmanFilter

from scipy.linalg import expm


# ============================================================
# Kalman Momentum Observer (FIXED dt, e.g. 1 kHz)
# ============================================================

class KalmanMomentumObserver:
    """
    State:
        x = [p; tau_ext]  (14D)

    Continuous model:
        p_dot       = tau_cmd + c - g + tau_ext
        tau_ext_dot = 0

    Measurement:
        y = p = M(q) dq
    """

    def __init__(self, dt_kf=0.001, nv=7, Q_scale=1e-3, R_scale=1e-3):
        self.nv = nv
        dt = float(dt_kf)

        # Continuous system matrices
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
        R  = np.eye(nv) * R_scale

        # -------- Van Loan discretization (ONCE) --------
        n = A.shape[0]
        H = np.block([
            [A, Qc],
            [np.zeros((n, n)), -A.T]
        ])

        Hd = expm(H * dt)
        Ad = expm(A * dt)
        Qd = Hd[:n, n:] @ Ad.T

        # -------- Kalman Filter --------
        self.kf = KalmanFilter(dim_x=2 * nv, dim_z=nv)
        self.kf.x = np.zeros(2 * nv)
        self.kf.P = np.eye(2 * nv)

        self.kf.F = Ad
        self.kf.B = B
        self.kf.H = C
        self.kf.Q = Qd
        self.kf.R = R

    def predict_fixed(self, u):
        """Prediction with fixed dt_kf"""
        self.kf.predict(u=u)

    def update_measurement(self, p_meas):
        """Measurement update (once per robot cycle)"""
        self.kf.update(p_meas)
        return self.kf.x[self.nv:].copy()


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="10.90.90.10")
    parser.add_argument("--seconds", type=float, default=5.0)

    # Robot control
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--tau-max", type=float, default=5.0)

    # Kalman filter
    parser.add_argument("--dt-kf", type=float, default=0.001, help="KF fixed timestep (e.g. 0.001 = 1kHz)")
    parser.add_argument("--Q", type=float, default=1e-3)
    parser.add_argument("--R", type=float, default=1e-3)
    parser.add_argument("--max-kf-steps", type=int, default=0, help="Max KF predict steps per robot cycle")

    parser.add_argument("--log", default="log_kf_dual_time.npz")
    args = parser.parse_args()

    nv = 7

    # -------- Robot connection --------
    robot = Robot(args.robot_ip)
    model = robot.load_model()

    print("⚠️ WARNING: Torque control will move the robot!")
    input("确认环境安全后按 Enter 开始...")

    active = robot.start_torque_control()

    # -------- Kalman observer (fixed dt) --------
    observer = KalmanMomentumObserver(
        dt_kf=args.dt_kf,
        nv=nv,
        Q_scale=args.Q,
        R_scale=args.R
    )

    # -------- Logs --------
    t_robot_log = []
    tau_ext_kf_log = []
    tau_ext_franka_log = []

    # -------- Time systems --------
    t_robot = 0.0          # robot real time
    acc = 0.0              # accumulator for KF
    dt_kf = args.dt_kf

    D_vec = np.ones(nv) * args.damping

    try:
        while True:
            state, dt = active.readOnce()
            dt_robot = dt.to_sec() if hasattr(dt, "to_sec") else float(dt)

            # -------- Robot time --------
            t_robot += dt_robot
            acc += dt_robot

            dq = np.array(state.dq, dtype=float)

            # -------- Safe damping control --------
            tau_cmd = -D_vec * dq
            tau_cmd = np.clip(tau_cmd, -args.tau_max, args.tau_max)

            # -------- Dynamics from libfranka --------
            M = np.array(model.mass(state), dtype=float).reshape(7, 7, order="F")
            c = np.array(model.coriolis(state), dtype=float)
            g = np.array(model.gravity(state), dtype=float)

            u = tau_cmd + c - g
            p_meas = M @ dq

            # -------- KF fixed-step prediction --------
            steps = 0
            while acc >= dt_kf and steps < args.max_kf_steps:
                observer.predict_fixed(u)
                acc -= dt_kf
                steps += 1

            # -------- KF measurement update (ONCE) --------
            tau_ext_hat = observer.update_measurement(p_meas)

            # -------- Franka residual torque --------
            tau_ext_franka = np.array(state.tau_J, dtype=float) - tau_cmd

            # -------- Send command --------
            active.writeOnce(Torques(tau_cmd.tolist()))

            # -------- Log --------
            t_robot_log.append(t_robot)
            tau_ext_kf_log.append(tau_ext_hat)
            tau_ext_franka_log.append(tau_ext_franka)

            if t_robot >= args.seconds:
                break

    except KeyboardInterrupt:
        print("\n[INFO] 手动停止")

    finally:
        try:
            robot.stop()
        except Exception:
            pass

        if len(t_robot_log) == 0:
            print("[WARN] 没有记录到数据")
            return
        
        print(f"time:{t_robot}")

        t = np.array(t_robot_log)
        tau_ext_kf = np.vstack(tau_ext_kf_log)
        tau_ext_franka = np.vstack(tau_ext_franka_log)

        np.savez(
            args.log,
            t_robot=t,
            tau_ext_kf=tau_ext_kf,
            tau_ext_franka=tau_ext_franka,
        )
        print(f"日志已保存到: {args.log}")

        # -------- Save plot (no GUI) --------
        fig, axs = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
        for i in range(7):
            axs[i].plot(t, tau_ext_kf[:, i], label="KF (fixed 1kHz)")
            axs[i].plot(t, tau_ext_franka[:, i], "--", label="Franka residual")
            axs[i].set_ylabel(f"J{i+1} [Nm]")
            axs[i].grid(True)
            if i == 0:
                axs[i].legend()

        axs[-1].set_xlabel("Robot time [s]")
        fig.suptitle("External torque: Dual-time Kalman vs Franka")
        plt.tight_layout()
        plt.savefig("kf_dual_time_vs_franka.png", dpi=200)
        plt.close("all")
        print("已保存图像: kf_dual_time_vs_franka.png")


if __name__ == "__main__":
    main()