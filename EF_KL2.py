#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import argparse
import numpy as np
from collections import deque

from pylibfranka import Robot, Torques
from filterpy.kalman import KalmanFilter
from scipy.linalg import expm


# ============================================================
# Kalman Momentum Observer (200 Hz)
# ============================================================

class KalmanMomentumObserver:
    """
    x = [p; tau_ext]
    p_dot = tau_cmd + c - g + tau_ext
    tau_ext_dot = 0
    y = p = M dq
    """

    def __init__(self, dt, nv=7, Q_scale=1e-3, R_scale=1e-3):
        self.nv = nv
        dt = float(dt)

        A = np.block([
            [np.zeros((nv, nv)), np.eye(nv)],
            [np.zeros((nv, nv)), np.zeros((nv, nv))]
        ])
        B = np.vstack([np.eye(nv), np.zeros((nv, nv))])
        C = np.hstack([np.eye(nv), np.zeros((nv, nv))])

        Qc = np.eye(2 * nv) * Q_scale
        R = np.eye(nv) * R_scale

        n = A.shape[0]
        H = np.block([[A, Qc],
                      [np.zeros((n, n)), -A.T]])
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="10.90.90.10")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--damping", type=float, default=0.3)
    parser.add_argument("--tau-max", type=float, default=5.0)
    args = parser.parse_args()

    nv = 7

    # -------- Shared buffer (lock-free style) --------
    buffer = deque(maxlen=5)   # 只存最新几帧
    stop_flag = threading.Event()

    # -------- Robot & model --------
    robot = Robot(args.robot_ip)
    model = robot.load_model()

    print("⚠️ WARNING: Torque control will move the robot!")
    input("确认安全后按 Enter...")

    active = robot.start_torque_control()

    # ==================================================
    # 1 kHz CONTROL THREAD
    # ==================================================
    def control_loop():
        D = args.damping
        tau_list = [0.0] * nv
        t_robot = 0.0

        try:
            while not stop_flag.is_set():
                state, dt = active.readOnce()
                dt_robot = dt.to_sec() if hasattr(dt, "to_sec") else float(dt)
                t_robot += dt_robot

                dq = state.dq
                for i in range(nv):
                    v = -D * float(dq[i])
                    v = max(min(v, args.tau_max), -args.tau_max)
                    tau_list[i] = v

                active.writeOnce(Torques(tau_list))

                # 只做“数据采样”，不做任何重计算
                buffer.append((tau_list.copy(), state))

                if t_robot >= args.seconds:
                    stop_flag.set()

        except Exception as e:
            print("[CONTROL ERROR]", e)
            stop_flag.set()

    # ==================================================
    # 200 Hz KALMAN THREAD
    # ==================================================
    def kf_loop():
        kf = KalmanMomentumObserver(dt=0.005, nv=nv)
        last_t = None

        while not stop_flag.is_set():
            if not buffer:
                time.sleep(0.001)
                continue

            tau_list, state = buffer[-1]

            tau_cmd = np.array(tau_list, dtype=float)
            dq = np.array(state.dq, dtype=float)

            M = np.array(model.mass(state), dtype=float).reshape(7, 7, order="F")
            c = np.array(model.coriolis(state), dtype=float)
            g = np.array(model.gravity(state), dtype=float)

            p_meas = M @ dq
            u = tau_cmd + c - g

            tau_ext_hat = kf.step(u, p_meas)

            # 这里只是示例：打印 / 记录 / 发布
            print("tau_ext_hat[0]:", tau_ext_hat[0])

            time.sleep(0.005)   # 200 Hz

    # -------- Run --------
    th_control = threading.Thread(target=control_loop, daemon=True)
    th_kf = threading.Thread(target=kf_loop, daemon=True)

    th_control.start()
    th_kf.start()

    th_control.join()
    stop_flag.set()
    th_kf.join()

    try:
        robot.stop()
    except Exception:
        pass

    print("Finished cleanly.")


if __name__ == "__main__":
    main()
