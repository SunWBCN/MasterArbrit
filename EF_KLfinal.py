#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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


def clamp(x, max_abs):
    return np.clip(x, -max_abs, +max_abs)

def discretize_A_B_paper(Ac: np.ndarray, Bc: np.ndarray, Ts: float):
    """
    Wahrburg 2015 eq. (23)
    [Ak Bk; 0 I] = exp([Ac Bc; 0 0] * Ts)
    """
    n = Ac.shape[0]
    m = Bc.shape[1]
    M = np.block([
        [Ac, Bc],
        [np.zeros((m, n + m))]
    ])
    Md = expm(M * Ts)
    Ak = Md[:n, :n]
    Bk = Md[:n, n:n+m]
    return Ak, Bk

def discretize_Q_van_loan_paper(Ac: np.ndarray, Qc: np.ndarray, Ts: float):
    """
    Wahrburg 2015 eq. (25a)-(25c)

    H = [Ac Qc; 0 -Ac^T]
    exp(H Ts) = [M11 M12; 0 M22]
    Qk = M12 * M11^T
    """
    n = Ac.shape[0]
    H = np.block([
        [Ac, Qc],
        [np.zeros((n, n)), -Ac.T]
    ])
    M = expm(H * Ts)
    M11 = M[:n, :n]
    M12 = M[:n, n:]
    Qk = M12 @ M11.T
    return Qk

def discretize_C_R_paper(Cc: np.ndarray, Rc: np.ndarray, Ts: float):
    """
    Wahrburg 2015 eq. (24)
    C = Cc
    R = (1/Ts) * Rc
    """
    C = Cc
    R = Rc / Ts
    return C, R

class KalmanCCFE_Wahrburg2015:
    """
    Rewritten to follow Wahrburg et al. 2015 exactly (Sec. IV-A/IV-B):
      Continuous:
        xdot = Ac x + Bc u + w,   w ~ N(0, Qc)
        y    = Cc x + v,          v ~ N(0, Rc)
      Discrete (time-varying):
        x_{k+1} = Ak x_k + Bk u_k + w_k,  w_k ~ N(0, Qk)
        y_k     = C x_k + v_k,            v_k ~ N(0, R)

    State:
      x = [p(7); f(6)]
    """

    def __init__(self, Ts=0.005, Qc_p=1e-2, Qc_f=1e-1, Rc=1e-2, Af_alpha=0.0):
        self.N = 7
        self.next = 6
        self.n = self.N + self.next
        self.Ts = float(Ts)

        # --- Continuous covariance matrices (paper uses Qc, Rc) ---
        # Qc = diag(Qc,p, Qc,f)  (eq. (21) text around (19)(18))
        self.Qc = np.zeros((self.n, self.n))
        self.Qc[:self.N, :self.N] = np.eye(self.N) * float(Qc_p)
        self.Qc[self.N:, self.N:] = np.eye(self.next) * float(Qc_f)

        # Rc: continuous measurement covariance (paper's notation)
        self.Rc = np.eye(self.N) * float(Rc)

        # --- Output matrix Cc (eq. (22)) ---
        self.Cc = np.hstack([np.eye(self.N), np.zeros((self.N, self.next))])

        # --- Fading matrix Af (eq. (18)) ---
        # paper: Af = 0 (default) OR Af = -diag(af,i) with af,i>0
        self.Af = -float(Af_alpha) * np.eye(self.next)

        # --- Kalman state init (paper suggests x0=0, P0=I) ---
        self.x = np.zeros((self.n,))
        self.P = np.eye(self.n)

        # Cached discrete C, R (depends only on Ts if Rc fixed)
        self.C, self.R = discretize_C_R_paper(self.Cc, self.Rc, self.Ts)

    def _build_Ac_Bc(self, J: np.ndarray):
        """
        Build Ac, Bc per eq. (21):
          Ac = [[0, -J^T],
                [0, Af]]
          Bc = [[I],
                [0]]
        """
        J = np.asarray(J, dtype=float).reshape((self.next, self.N))
        Ac = np.zeros((self.n, self.n))
        Ac[:self.N, self.N:] = -J.T
        Ac[self.N:, self.N:] = self.Af

        Bc = np.vstack([np.eye(self.N), np.zeros((self.next, self.N))])
        return Ac, Bc

    def step(self, J, u, y):
        """
        One KF step following paper IV-B steps 3)-7):
          3) discretize -> Ak, Bk, Qk
          4) predict (27)
          5) gain (28)
          6) update (29)
          7) extract fhat (30)
        Inputs:
          J: 6x7
          u: tau = tau_mot + C^T dq - G - tau_fric_hat   (eq. (9), Step 2)
          y: p = M dq                                    (Step 1)
        Returns:
          f_hat (6,)
        """
        u = np.asarray(u, dtype=float).reshape((self.N,))
        y = np.asarray(y, dtype=float).reshape((self.N,))

        # (3) Discretize using eq. (23) and (25)
        Ac, Bc = self._build_Ac_Bc(J)
        Ak, Bk = discretize_A_B_paper(Ac, Bc, self.Ts)            # eq. (23)
        Qk     = discretize_Q_van_loan_paper(Ac, self.Qc, self.Ts) # eq. (25)

        # (4) Predict (eq. 27)
        x_pred = Ak @ self.x + Bk @ u
        P_pred = Ak @ self.P @ Ak.T + Qk

        # (5) Kalman gain (eq. 28)
        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)

        # (6) Update (eq. 29)
        innov = y - self.C @ x_pred
        self.x = x_pred + K @ innov
        I = np.eye(self.n)
        # Joseph form (paper eq. 29b)
        self.P = (I - K @ self.C) @ P_pred @ (I - K @ self.C).T + K @ self.R @ K.T

        # (7) Extract fhat (eq. 30)
        f_hat = self.x[self.N:].copy()
        return f_hat


def van_loan_discretize(A, Qc, dt):
    """
    Van Loan method:
      Given continuous: xdot = A x + w,  E[w w^T] = Qc
      Return discrete Ad, Qd for x_{k+1} = Ad x_k + w_k, E[w_k w_k^T]=Qd
    """
    n = A.shape[0]
    H = np.block([
        [A, Qc],
        [np.zeros((n, n)), -A.T]
    ])
    Hd = expm(H * dt)
    Ad = expm(A * dt)
    Qd = Hd[:n, n:] @ Ad.T
    return Ad, Qd


# ============================================================
# KF: state x = [p(7); F(6)] (13D)
#
# continuous:
#   p_dot = u + J^T F
#   F_dot = 0
#
# measurement:
#   y = p_meas
#
# We do "time-varying F/B" each step because J changes with q.
# We keep a fixed dt_kf (e.g. 0.005 = 200Hz) and do:
#   predict with u_eff = [u; 0]   and F matrix depends on J
# ============================================================

class KalmanWrenchFromMomentum:
    def __init__(self, dt_kf=0.005,
                 Qp_scale=1e-2, QF_scale=1e-1,
                 R_scale=1e-2):
        self.np = 7
        self.nf = 6
        self.n = self.np + self.nf
        self.dt = float(dt_kf)

        # Measurement: y = p  => H = [I, 0]
        self.H = np.hstack([np.eye(self.np), np.zeros((self.np, self.nf))])

        # Initial filter
        self.kf = KalmanFilter(dim_x=self.n, dim_z=self.np)
        self.kf.x = np.zeros(self.n)
        self.kf.P = np.eye(self.n) * 1.0
        self.kf.H = self.H
        self.kf.R = np.eye(self.np) * float(R_scale)

        # Process noise (continuous)
        Qc = np.zeros((self.n, self.n))
        Qc[:self.np, :self.np] = np.eye(self.np) * float(Qp_scale)
        Qc[self.np:, self.np:] = np.eye(self.nf) * float(QF_scale)
        self.Qc = Qc

        # We'll set kf.F and kf.Q each step because J changes.

        # Input matrix for u (7D) -> xdot
        # xdot = A x + B u
        # p_dot includes +u ; F_dot no u
        self.B = np.vstack([np.eye(self.np), np.zeros((self.nf, self.np))])

    def set_dynamics_from_J(self, J):
        """
        Given J (6x7), set continuous A:
          [ 0   J^T ]
          [ 0    0  ]
        """
        A = np.zeros((self.n, self.n))
        A[:self.np, self.np:] = -J.T  # (7x6)
        #A[:self.np, self.np:] = -J.T  # (7x6)
        self.A = A

        # Discretize (fixed dt)
        Ad, Qd = van_loan_discretize(A, self.Qc, self.dt)
        self.kf.F = Ad
        self.kf.Q = Qd

    def predict(self, u):
        """
        u is 7D joint-space input for p_dot: u = tau_cmd + c - g
        We discretize input as x_{k+1} = Ad x_k + Bd u
        For small dt, Bd ≈ B*dt (good enough here)
        """
        u = np.asarray(u, dtype=float).reshape(self.np,)
        Bd = self.B * self.dt
        self.kf.predict(u=u, B=Bd)

    def update(self, p_meas):
        p_meas = np.asarray(p_meas, dtype=float).reshape(self.np,)
        self.kf.update(p_meas)
        return self.kf.x[self.np:].copy()  # F(6)


# ============================================================
# Phase 1: Online collect (1 kHz)
# ============================================================

def run_collect(robot_ip, seconds, damping, tau_max):
    nv = 7
    robot = Robot(robot_ip)
    model = robot.load_model()

    print("⚠️ WARNING: Torque control will move the robot!")
    input("确认环境安全后按 Enter 开始运行...")

    active = robot.start_torque_control()

    states = []
    tau_cmd_log = []
    t_log = []
    wrench_franka_log = []

    tau_cmd = np.zeros(nv)
    t_robot = 0.0

    try:
        while t_robot < seconds:
            state, dt = active.readOnce()
            dt_robot = dt.to_sec() if hasattr(dt, "to_sec") else float(dt)
            t_robot += dt_robot

            dq = np.array(state.dq, dtype=float)

            tau_cmd = -float(damping) * dq
            tau_cmd = clamp(tau_cmd, float(tau_max))

            active.writeOnce(Torques(tau_cmd.tolist()))

            t_log.append(t_robot)
            tau_cmd_log.append(tau_cmd.copy())
            states.append(state)

            # built-in 6D wrench (K frame)
            wrench_franka_log.append(np.array(state.O_F_ext_hat_K, dtype=float).copy())

        print(f"[OK] Online run finished: t={t_robot:.3f}s, N={len(states)}")

    except Exception as e:
        print("[ABORTED]", e)

    finally:
        try:
            robot.stop()
        except Exception:
            pass

    return model, np.array(t_log), np.vstack(tau_cmd_log), states, np.vstack(wrench_franka_log)


# ============================================================
# Phase 2: Offline KF (downsample to dt_kf) + plots
# ============================================================

def offline_kf_and_plots(model, t, tau_cmd_log, states, wrench_franka, exp_dir,
                         dt_kf=0.005, Qp=1e-2, QF=1e-1, R=1e-2):
    if len(states) == 0:
        print("[ERROR] No samples.")
        return

    # Create KF (fixed dt)
    kf = KalmanWrenchFromMomentum(dt_kf=dt_kf, Qp_scale=Qp, QF_scale=QF, R_scale=R)

    # We run KF at dt_kf by sampling the nearest robot state at those times
    t_end = float(t[-1])
    t_grid = np.arange(0.0, t_end, dt_kf)

    # index pointer for nearest sample
    idx = 0
    N = len(t)

    wrench_kf = []
    wrench_franka_ds = []
    t_use = []

    for tg in t_grid:
        # advance idx until t[idx] >= tg
        while idx < N - 1 and t[idx] < tg:
            idx += 1

        state = states[idx]
        tau_cmd = tau_cmd_log[idx]
        wrench_f = wrench_franka[idx]

        dq = np.array(state.dq, dtype=float)

        # dynamics from state
        M = np.array(model.mass(state), dtype=float).reshape((7, 7), order="F")
        c = np.array(model.coriolis(state), dtype=float)
        g = np.array(model.gravity(state), dtype=float)
        J = np.array(model.zero_jacobian(state), dtype=float).reshape((6, 7), order="F")

        p_meas = M @ dq
        u = tau_cmd + c
        #u = tau_cmd + c - g

        # update time-varying dynamics
        kf.set_dynamics_from_J(J)

        # KF predict+update once per dt_kf
        kf.predict(u=u)
        F_hat = kf.update(p_meas)

        wrench_kf.append(F_hat)
        wrench_franka_ds.append(wrench_f)
        t_use.append(t[idx])

    wrench_kf = np.vstack(wrench_kf)
    wrench_franka_ds = np.vstack(wrench_franka_ds)
    t_use = np.array(t_use)

    # -------------------------
    # Compare plots
    # -------------------------
    def save_compare_xyz(title, ylab, idxs, fname):
        plt.figure(figsize=(10, 5))
        for i in idxs:
            plt.plot(t_use, wrench_franka_ds[:, i], "--", alpha=0.6,
                     label=f"{labels[i]} Franka")
            plt.plot(t_use, wrench_kf[:, i], label=f"{labels[i]} KF")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel(ylab)
        plt.grid(True)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, fname), dpi=200)
        plt.close()

    labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    save_compare_xyz("External FORCE comparison (K frame)", "Force [N]", [0, 1, 2],
                     "compare_force_xyz.png")
    save_compare_xyz("External TORQUE comparison (K frame)", "Torque [Nm]", [3, 4, 5],
                     "compare_torque_xyz.png")

    fn_franka = np.linalg.norm(wrench_franka_ds[:, :3], axis=1)
    fn_kf = np.linalg.norm(wrench_kf[:, :3], axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(t_use, fn_franka, "--", alpha=0.7, label="||F|| Franka")
    plt.plot(t_use, fn_kf, label="||F|| KF")
    plt.title("Force magnitude comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("N")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "compare_force_norm.png"), dpi=200)
    plt.close()

    # Per-axis
    units = ["N", "N", "N", "Nm", "Nm", "Nm"]
    for i in range(6):
        plt.figure(figsize=(10, 4))
        plt.plot(t_use, wrench_franka_ds[:, i], "--", alpha=0.7, label=f"{labels[i]} Franka")
        plt.plot(t_use, wrench_kf[:, i], label=f"{labels[i]} KF")
        plt.title(f"{labels[i]} comparison")
        plt.xlabel("Time [s]")
        plt.ylabel(units[i])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"compare_{labels[i]}.png"), dpi=200)
        plt.close()

    print(f"[OK] Saved images to: {exp_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="10.90.90.10")
    parser.add_argument("--seconds", type=float, default=10.0)

    # online control (can set both to 0 to avoid motion)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--tau-max", type=float, default=0.0)

    # KF settings (run at dt_kf, e.g., 0.005=200Hz)
    parser.add_argument("--dt-kf", type=float, default=0.005)

    # KF tuning
    parser.add_argument("--Qp", type=float, default=1e-2)
    parser.add_argument("--QF", type=float, default=1e-1)
    parser.add_argument("--R", type=float, default=1e-2)

    parser.add_argument("--outroot", type=str, default="experiments")
    args = parser.parse_args()

    exp_dir = create_experiment_dir(args.outroot)
    print(f"[INFO] Experiment dir: {exp_dir}")

    with open(os.path.join(exp_dir, "meta.txt"), "w") as f:
        f.write(f"robot_ip: {args.robot_ip}\n")
        f.write(f"seconds: {args.seconds}\n")
        f.write(f"damping: {args.damping}\n")
        f.write(f"tau_max: {args.tau_max}\n")
        f.write(f"dt_kf: {args.dt_kf}\n")
        f.write("KF style: filterpy + Van Loan discretization\n")
        f.write("State: x=[p(7); F(6)]\n")
        f.write("Model: p_dot = u + J^T F, F_dot=0, y=p\n")
        f.write(f"Qp: {args.Qp}\n")
        f.write(f"QF: {args.QF}\n")
        f.write(f"R: {args.R}\n")
        f.write("Compare vs state.K_F_ext_hat_K\n")

    model, t, tau_cmd_log, states, wrench_franka = run_collect(
        robot_ip=args.robot_ip,
        seconds=args.seconds,
        damping=args.damping,
        tau_max=args.tau_max
    )

    offline_kf_and_plots(
        model=model,
        t=t,
        tau_cmd_log=tau_cmd_log,
        states=states,
        wrench_franka=wrench_franka,
        exp_dir=exp_dir,
        dt_kf=args.dt_kf,
        Qp=args.Qp,
        QF=args.QF,
        R=args.R
    )

    print("✅ Done.")


if __name__ == "__main__":
    main()
