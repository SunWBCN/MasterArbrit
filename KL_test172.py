#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.linalg import expm

from pylibfranka import Robot, Torques, ControllerMode, CartesianPose

# ============================================================
# Parameter tuning
# ============================================================

DEFAULT_QP = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
DEFAULT_QF = [1e-1, 1e-1, 5e-2, 1e-2, 1e-2, 1e-2]
DEFAULT_AF = [0.1, 0.1, 0.05, 0.5, 0.5, 0.5]
DEFAULT_R  = [1e-8] * 7

T_GOAL = np.array([
    0.7074, -0.7068, 0.0001, 0.0,
   -0.7068, -0.7074, 0.0001, 0.0,
   -0.0,   -0.0001, -1.0,   0.0,
    0.5005, 0.0133,  0.0866, 1.0
], dtype=float)



# ============================================================
# Utils
# ============================================================

def create_experiment_dir(root="experiments"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(root, timestamp)
    os.makedirs(exp_dir, exist_ok=False)
    return exp_dir


def clamp(x, max_abs):
    return np.clip(x, -max_abs, +max_abs)


def parse_scalar_or_vector(s, n: int, name: str):
    """
    Accept either:
      - scalar:  "1e-2"
      - vector:  "1e-2,1e-2,1e0,1e-2,1e-2,1e-2"
    Return float or np.ndarray (n,)
    """
    if s is None:
        return None
    s = str(s).strip()
    if "," not in s:
        return float(s)
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != n:
        raise ValueError(f"{name} expects {n} values, got {len(parts)}: {parts}")
    return np.array([float(p) for p in parts], dtype=float)


# ============================================================
# Discretization (Wahrburg 2015)
# ============================================================

def discretize_A_B_paper(Ac, Bc, Ts):
    n = Ac.shape[0]
    m = Bc.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    Md = expm(M * Ts)
    return Md[:n, :n], Md[:n, n:]


def discretize_Q_van_loan_paper(Ac, Qc, Ts):
    n = Ac.shape[0]
    H = np.zeros((2 * n, 2 * n))
    H[:n, :n] = Ac
    H[:n, n:] = Qc
    H[n:, n:] = -Ac.T
    Hexp = expm(H * Ts)
    M12 = Hexp[:n, n:]
    M11 = Hexp[:n, :n]
    return M12 @ M11.T


def discretize_C_R_paper(Cc, Rc, Ts):
    return Cc.copy(), Rc / Ts


# ============================================================
# Kalman CCFE (x=[p(7); f(6)])
# ============================================================

class KalmanCCFE_Wahrburg2015:

    def __init__(self, Ts=0.005, Qc_p=1e-2, Qc_f=1e-1, Rc=1e-2, Af_alpha=0.0):
        self.N = 7
        self.nf = 6
        self.n = self.N + self.nf
        self.Ts = float(Ts)

        # --- Qc_p ---
        Qc_p = np.asarray(Qc_p, dtype=float)
        if Qc_p.ndim == 0:
            Qc_p = np.full((self.N,), Qc_p)

        # --- Qc_f ---
        Qc_f = np.asarray(Qc_f, dtype=float)
        if Qc_f.ndim == 0:
            Qc_f = np.full((self.nf,), Qc_f)

        self.Qc = np.zeros((self.n, self.n))
        self.Qc[:self.N, :self.N] = np.diag(Qc_p)
        self.Qc[self.N:, self.N:] = np.diag(Qc_f)

        # --- Rc ---
        Rc = np.asarray(Rc, dtype=float)
        if Rc.ndim == 0:
            Rc = np.full((self.N,), Rc)

        self.Rc = np.diag(Rc)
        #self.Rc = np.eye(self.N) * float(Rc)

        self.Cc = np.hstack([np.eye(self.N), np.zeros((self.N, self.nf))])

        # --- Af ---
        Af_alpha = np.asarray(Af_alpha, dtype=float)
        if Af_alpha.ndim == 0:
            Af_alpha = np.full((self.nf,), Af_alpha)
        self.Af = -np.diag(Af_alpha)

        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)

        self.C, self.R = discretize_C_R_paper(self.Cc, self.Rc, self.Ts)

    def _build_Ac_Bc(self, J):
        J = np.asarray(J, dtype=float).reshape((self.nf, self.N))
        Ac = np.zeros((self.n, self.n))
        Ac[:self.N, self.N:] = -J.T
        Ac[self.N:, self.N:] = self.Af
        Bc = np.vstack([np.eye(self.N), np.zeros((self.nf, self.N))])
        return Ac, Bc

    def step(self, J, u, y):
        Ac, Bc = self._build_Ac_Bc(J)
        Ak, Bk = discretize_A_B_paper(Ac, Bc, self.Ts)
        Qk = discretize_Q_van_loan_paper(Ac, self.Qc, self.Ts)

        # predict
        x_pred = Ak @ self.x + Bk @ u
        P_pred = Ak @ self.P @ Ak.T + Qk

        # update
        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (y - self.C @ x_pred)

        I = np.eye(self.n)
        self.P = (I - K @ self.C) @ P_pred @ (I - K @ self.C).T + K @ self.R @ K.T

        return self.x[self.N:].copy()


# ============================================================
# Phase 1: Online collect
# ============================================================

def run_collect(robot_ip,
                move_time=3.0,
                hold_time=5.0,
                tau_max=15.0,
                Fz_des=10.0,
                KpF=0.8,
                KiF=5.0,
                joint_damping=2.0):

    robot = Robot(robot_ip)
    model = robot.load_model()

    print("⚠️ Robot will move!")
    input("确认环境安全后按 Enter 开始...")

    states = []
    tau_cmd_log = []
    t_log = []
    wrench_franka_log = []

    t_robot = 0.0

    def get_T(state):
        return np.array(state.O_T_EE, dtype=float).copy()

    def min_jerk(s):
        return 10*s**3 - 15*s**4 + 6*s**5

    def interp_pose(T0, T1, a):
        T = T1.copy()
        p0 = np.array([T0[12], T0[13], T0[14]])
        p1 = np.array([T1[12], T1[13], T1[14]])
        p = (1-a)*p0 + a*p1
        T[12], T[13], T[14] = p.tolist()
        return T

    # =========================================================
    # Phase A — Cartesian Pose Control
    # =========================================================
    print("[Phase A] Move to goal pose...")

    active = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)

    T0 = None
    t_move = 0.0

    while t_move < move_time:

        state, dt = active.readOnce()
        dt_robot = dt.to_sec() if hasattr(dt, "to_sec") else float(dt)

        t_robot += dt_robot
        t_move += dt_robot

        s = min(1.0, t_move / move_time)
        a = min_jerk(s)

        T_des = interp_pose(T0, T_GOAL, a)

        active.writeOnce(CartesianPose(T_des))

    active.stop()
    print("[OK] Reached goal pose.")

    # =========================================================
    # Phase B — Torque Control (10N downward)
    # =========================================================
    print("[Phase B] Force hold 10N downward...")

    active = robot.start_torque_control()

    int_e = 0.0
    contact_found = False
    contact_thresh = 0.3 * abs(Fz_des)

    t_hold = 0.0

    while t_hold < hold_time:

        state, dt = active.readOnce()
        dt_robot = dt.to_sec() if hasattr(dt, "to_sec") else float(dt)

        t_robot += dt_robot
        t_hold += dt_robot

        dq = np.array(state.dq, dtype=float)
        wrench = np.array(state.O_F_ext_hat_K, dtype=float)
        Fz = wrench[2]

        e = Fz_des - Fz

        if (not contact_found) and abs(Fz) > contact_thresh:
            contact_found = True
            int_e = 0.0

        if contact_found:
            int_e += e * dt_robot
            F_cmd = Fz_des + KpF*e + KiF*int_e
        else:
            F_cmd = 0.2 * Fz_des

        wrench_cmd = np.array([0,0,F_cmd,0,0,0], dtype=float)

        J = np.array(model.zero_jacobian(state),
                     dtype=float).reshape((6,7), order="F")

        tau_cmd = J.T @ wrench_cmd - joint_damping * dq
        tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)

        active.writeOnce(Torques(tau_cmd.tolist()))

        states.append(state)
        tau_cmd_log.append(tau_cmd.copy())
        t_log.append(t_robot)
        wrench_franka_log.append(wrench.copy())

    robot.stop()

    print("[OK] Force hold finished.")

    return model, np.array(t_log), np.vstack(tau_cmd_log), states, np.vstack(wrench_franka_log)



# ============================================================
# Phase 2: Offline KF + plots
# ============================================================

def offline_kf_and_plots(model, t, tau_log, states, wrench_franka,
                         exp_dir, dt_kf, Qp, QF, R, Af, use_gravity):
    """
    Offline CCFE (Wahrburg 2015) with engineering-grade plotting & logging.
    Core logic: A (kf.step)
    Engineering: B (plots, checks, norms)
    """

    # -------------------------
    # Safety check
    # -------------------------
    if len(states) == 0 or len(t) == 0:
        print("[ERROR] No samples for offline KF.")
        return

    # -------------------------
    # Create KF (A logic)
    # -------------------------
    kf = KalmanCCFE_Wahrburg2015(
        Ts=dt_kf,
        Qc_p=Qp,
        Qc_f=QF,
        Rc=R,
        Af_alpha=Af,
    )

    # -------------------------
    # Downsample timeline
    # -------------------------
    t_end = float(t[-1])
    t_grid = np.arange(0.0, t_end, dt_kf)

    idx = 0
    N = len(t)

    wrench_kf = []
    wrench_franka_ds = []
    t_use = []

    # -------------------------
    # Main loop (A logic)
    # -------------------------
    for tg in t_grid:
        while idx < N - 1 and t[idx] < tg:
            idx += 1

        s = states[idx]
        dq = np.array(s.dq, dtype=float)

        # dynamics
        M = np.array(model.mass(s), dtype=float).reshape((7, 7), order="F")
        c = np.array(model.coriolis(s), dtype=float)
        g = np.array(model.gravity(s), dtype=float)
        J = np.array(model.zero_jacobian(s), dtype=float).reshape((6, 7), order="F")

        # measurement
        p_meas = M @ dq

        # input (paper style)
        u = tau_log[idx] + c
        if use_gravity:
            u -= g

        # ---- CCFE step (核心不动) ----
        f_hat = kf.step(J=J, u=u, y=p_meas)

        wrench_kf.append(f_hat)
        wrench_franka_ds.append(wrench_franka[idx])
        t_use.append(t[idx])

    wrench_kf = np.vstack(wrench_kf)
    wrench_franka_ds = np.vstack(wrench_franka_ds)
    t_use = np.array(t_use)

    # -------------------------
    # Save raw data
    # -------------------------
    np.savez(
        os.path.join(exp_dir, "log_offline_ccfe.npz"),
        t=t_use,
        wrench_kf=wrench_kf,
        wrench_franka=wrench_franka_ds,
        dt_kf=float(dt_kf),
        Qp=Qp,
        QF=QF,
        R=R,
        Af=Af,
        use_gravity=use_gravity,
    )

    # -------------------------
    # Plotting (B optimization)
    # -------------------------
    labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    units  = ["N", "N", "N", "Nm", "Nm", "Nm"]

    def save_compare(title, ylab, idxs, fname):
        plt.figure(figsize=(10, 5))
        for i in idxs:
            plt.plot(t_use, wrench_franka_ds[:, i], "--", alpha=0.6,
                     label=f"{labels[i]} Franka")
            plt.plot(t_use, wrench_kf[:, i], label=f"{labels[i]} CCFE-KF")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel(ylab)
        plt.grid(True)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, fname), dpi=200)
        plt.close()

    # XYZ force / torque
    save_compare("External FORCE comparison (O frame)", "Force [N]",
                 [0, 1, 2], "compare_force_xyz.png")
    save_compare("External TORQUE comparison (O frame)", "Torque [Nm]",
                 [3, 4, 5], "compare_torque_xyz.png")

    # Force norm
    fn_franka = np.linalg.norm(wrench_franka_ds[:, :3], axis=1)
    fn_kf = np.linalg.norm(wrench_kf[:, :3], axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(t_use, fn_franka, "--", alpha=0.7, label="||F|| Franka")
    plt.plot(t_use, fn_kf, label="||F|| CCFE-KF")
    plt.title("Force magnitude comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("N")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "compare_force_norm.png"), dpi=200)
    plt.close()

    # Per-axis plots
    for i in range(6):
        plt.figure(figsize=(10, 4))
        plt.plot(t_use, wrench_franka_ds[:, i], "--", alpha=0.7,
                 label=f"{labels[i]} Franka")
        plt.plot(t_use, wrench_kf[:, i],
                 label=f"{labels[i]} CCFE-KF")
        plt.title(f"{labels[i]} comparison")
        plt.xlabel("Time [s]")
        plt.ylabel(units[i])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"compare_{labels[i]}.png"), dpi=200)
        plt.close()

    print(f"[OK] Offline CCFE results saved to: {exp_dir}")
# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot-ip", default="10.90.90.10")
    parser.add_argument("--seconds", type=float, default=10.0)

    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--tau-max", type=float, default=15.0)

    parser.add_argument("--dt-kf", type=float, default=0.001)

    parser.add_argument("--Qp", type=str, default=None) 
    parser.add_argument("--QF", type=str, default=None)
    parser.add_argument("--Af", type=str, default=None)
    parser.add_argument("--R",  type=str, default=None)

    parser.add_argument("--use-gravity", action="store_true")
    parser.add_argument("--outroot", default="experiments")

    args = parser.parse_args()

    Qp = parse_scalar_or_vector(args.Qp, 7, "Qp")
    QF = parse_scalar_or_vector(args.QF, 6, "QF")
    Af = parse_scalar_or_vector(args.Af, 6, "Af")
    R  = parse_scalar_or_vector(args.R,  7, "R" )

    if Qp is None:
        Qp = np.array(DEFAULT_QP, dtype=float)
    if QF is None:
        QF = np.array(DEFAULT_QF, dtype=float)
    if Af is None:
        Af = np.array(DEFAULT_AF, dtype=float)
    if R is None:
        R  = np.array(DEFAULT_R,  dtype=float)

    exp_dir = create_experiment_dir(args.outroot)

    model, t, tau, states, wrench = run_collect(
        robot_ip=args.robot_ip,
        move_time=3.0,
        hold_time=args.seconds,
        tau_max=args.tau_max,
        Fz_des=10.0
    )

    offline_kf_and_plots(
        model, t, tau, states, wrench,
        exp_dir, args.dt_kf, Qp, QF, R, Af, args.use_gravity
    )

    print("✅ Done:", exp_dir)


if __name__ == "__main__":
    main()
