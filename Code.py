#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.linalg import expm

import pinocchio as pin

from pylibfranka import Robot, ControllerMode, CartesianPose

FRANKA_URDF = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "../../../test/fr3.urdf"))

# ============================================================
# Parameter tuning
# ============================================================

DEFAULT_QP = [1e-1] * 7
DEFAULT_QF = [1000] * 6
DEFAULT_AF = [0.1, 0.1, 0.05, 0.5, 0.5, 0.5] # here is positive alpha, which means negative feedback in Af = -diag(Af_alpha)
DEFAULT_R  = [1e-3] * 7

T_GOAL = np.array([
    0.7219, -0.6920,  0.0, 0.0,
   -0.6920, -0.7219,  0.0, 0.0,
    0.0,     0.0,    -1.0, 0.0,
    0.5015,  0.0053,  0.0469, 1.0
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
    if s is None:
        return None
    s = str(s).strip()
    if "," not in s:
        return float(s)
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != n:
        raise ValueError(f"{name} expects {n} values, got {len(parts)}: {parts}")
    return np.array([float(p) for p in parts], dtype=float)


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10*s**3 - 15*s**4 + 6*s**5


def get_pos_from_T(T):
    return np.array([T[12], T[13], T[14]], dtype=float)


def set_pos_in_T(T, p):
    T2 = np.array(T, dtype=float).copy()
    T2[12], T2[13], T2[14] = p.tolist()
    return T2


def R_from_T_colmajor(T):
    return np.array([
        [T[0], T[4], T[8]],
        [T[1], T[5], T[9]],
        [T[2], T[6], T[10]],
    ], dtype=float)


def set_R_in_T_colmajor(T, R):
    T[0], T[1], T[2]  = R[0,0], R[1,0], R[2,0]
    T[4], T[5], T[6]  = R[0,1], R[1,1], R[2,1]
    T[8], T[9], T[10] = R[0,2], R[1,2], R[2,2]


def quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        w, x = 0.25 * S, (R[2,1] - R[1,2]) / S
        y, z  = (R[0,2] - R[2,0]) / S, (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
        w, x = (R[2,1] - R[1,2]) / S, 0.25 * S
        y, z  = (R[0,1] + R[1,0]) / S, (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
        w, x = (R[0,2] - R[2,0]) / S, (R[0,1] + R[1,0]) / S
        y, z  = 0.25 * S, (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
        w, x = (R[1,0] - R[0,1]) / S, (R[0,2] + R[2,0]) / S
        y, z  = (R[1,2] + R[2,1]) / S, 0.25 * S
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
        q1, dot = -q1, -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    theta = theta0 * s
    return np.sin(theta0 - theta) / sin0 * q0 + np.sin(theta) / sin0 * q1


def interp_pose(T0, T1, a):
    """Interpolate translation and orientation (slerp) between T0 and T1."""
    T = np.array(T0, dtype=float).copy()
    p0 = get_pos_from_T(T0)
    p1 = get_pos_from_T(T1)
    p = (1 - a) * p0 + a * p1
    T[12], T[13], T[14] = p.tolist()
    q0 = quat_from_R(R_from_T_colmajor(T0))
    q1 = quat_from_R(R_from_T_colmajor(T1))
    R = R_from_quat(slerp(q0, q1, a))
    set_R_in_T_colmajor(T, R)
    return T


# ============================================================
# Discretization (Wahrburg 2015)
# ============================================================

def discretize_A_B_paper(Ac, Bc, Ts):
    n = Ac.shape[0] # A : n x n
    m = Bc.shape[1] # B : n x m
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

        Qc_p = np.asarray(Qc_p, dtype=float)
        if Qc_p.ndim == 0:
            Qc_p = np.full((self.N,), Qc_p)

        Qc_f = np.asarray(Qc_f, dtype=float)
        if Qc_f.ndim == 0:
            Qc_f = np.full((self.nf,), Qc_f)

        self.Qc = np.zeros((self.n, self.n))
        self.Qc[:self.N, :self.N] = np.diag(Qc_p)
        self.Qc[self.N:, self.N:] = np.diag(Qc_f)

        Rc = np.asarray(Rc, dtype=float)
        if Rc.ndim == 0:
            Rc = np.full((self.N,), Rc)

        self.Rc = np.diag(Rc)
        self.Cc = np.hstack([np.eye(self.N), np.zeros((self.N, self.nf))])

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
        Ac[:self.N, self.N:] = J.T
        Ac[self.N:, self.N:] = self.Af
        Bc = np.vstack([np.eye(self.N), np.zeros((self.nf, self.N))])
        return Ac, Bc

    def step(self, J, u, y):
        Ac, Bc = self._build_Ac_Bc(J)
        Ak, Bk = discretize_A_B_paper(Ac, Bc, self.Ts)
        Qk = discretize_Q_van_loan_paper(Ac, self.Qc, self.Ts)

        x_pred = Ak @ self.x + Bk @ u
        P_pred = Ak @ self.P @ Ak.T + Qk

        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (y - self.C @ x_pred)

        I = np.eye(self.n)
        self.P = (I - K @ self.C) @ P_pred @ (I - K @ self.C).T + K @ self.R @ K.T

        return self.x[self.N:].copy()


# ============================================================
# High-Gain Observer (momentum-based)
# ============================================================

def err_mapping_func(err, alg):
    """Map momentum error to L1/L2 injection signals.

    alg = 'linear'  : pure high-gain (L1*err, L2*err)
    alg = 'sos'     : super-twisting / SOS (L1*|e|^0.5*sign(e), L2*sign(e))
    alg = 'sliding' : first-order sliding mode (L1*sign(e), L2*sign(e))
    """
    if alg == "linear":
        return err, err
    elif alg == "sos":
        err1 = np.sign(err) * np.sqrt(np.abs(err))
        err2 = np.sign(err)
        return err1, err2
    elif alg == "sliding":
        s = np.sign(err)
        return s, s
    else:
        raise ValueError(f"Unknown hg_alg: {alg!r}. Choose linear / sos / sliding.")


class HighGainObserver:
    """Momentum-based high-gain observer.

    Dynamics (continuous, Euler-integrated):
        p_hat_dot = tau + C^T v - g + tau_ext_hat + L1 * err1
        tau_ext_hat_dot = L2 * err2
    where err = M @ v - p_hat  (momentum error)
    """

    def __init__(self, dt, nv=7, bandwidth=10.0, alg="linear"):
        self.dt  = float(dt)
        self.nv  = nv
        self.alg = alg
        bw = np.full(nv, bandwidth)
        self.L1 = np.diag(2.0 * bw)
        self.L2 = np.diag(bw ** 2)
        self.est_gm      = np.zeros(nv)   # estimated generalized momentum
        self.est_ext_tau = np.zeros(nv)   # estimated external joint torque

    def update(self, v, M, C, g, tau_motor):
        err = M @ v - self.est_gm
        err1, err2 = err_mapping_func(err, self.alg)
        dgm          = tau_motor + C.T @ v - g + self.est_ext_tau + self.L1 @ err1
        dest_ext_tau = self.L2 @ err2
        self.est_gm      += self.dt * dgm
        self.est_ext_tau += self.dt * dest_ext_tau
        return self.est_ext_tau.copy(), self.est_gm.copy()

    def reset(self):
        self.est_gm[:]      = 0.0
        self.est_ext_tau[:] = 0.0


# ============================================================
# Phase 1: Online collect with impedance contact measurement
# ============================================================

def run_collect(robot_ip,
                move_time=3.0,
                search_time=4.0,
                hold_time=5.0,
                approach_speed=0.0015,   # m/s
                contact_thresh=2.0,      # N
                max_down_disp=0.050,     # m
                force_axis=2,            # 0 for Fx, 1 for Fy, 2 for Fz
                force_sign=1.0):         # 1.0 or -1.0, depending on robot orientation and sensor sign convention

    robot = Robot(robot_ip)
    model = robot.load_model()

    print("⚠️ Robot will move!")
    input("确认环境安全后按 Enter 开始...")

    states = []
    t_log = []
    wrench_franka_log = []
    tau_meas_log = []
    tau_des_log = []
    q_log = []
    dq_log = []
    z_ref_log = []
    phase_log = []

    t_robot = 0.0

    # =========================================================
    # Phase A — Move to pre-contact pose (own control session)
    # =========================================================
    print("[Phase A] Move to pre-contact pose...")

    active = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
    state, dt = active.readOnce()
    t_robot += dt.to_sec() if hasattr(dt, "to_sec") else float(dt)
    T0 = np.array(state.O_T_EE, dtype=float).copy()

    t_move = 0.0
    while True:
        state, dt = active.readOnce()
        dt_robot = dt.to_sec() if hasattr(dt, "to_sec") else float(dt)
        t_robot += dt_robot
        t_move  += dt_robot

        s     = min(1.0, t_move / move_time)
        a     = min_jerk(s)
        T_cmd = interp_pose(T0, T_GOAL, a)
        cp    = CartesianPose(T_cmd.tolist())
        done  = t_move >= move_time
        if done:
            cp.motion_finished = True
        active.writeOnce(cp)

        wrench = np.array(state.O_F_ext_hat_K, dtype=float)
        states.append(state)
        t_log.append(t_robot)
        wrench_franka_log.append(wrench.copy())
        tau_meas_log.append(np.array(state.tau_J,   dtype=float).copy())
        tau_des_log.append(np.array(state.tau_J_d, dtype=float).copy())
        q_log.append(np.array(state.q,   dtype=float).copy())
        dq_log.append(np.array(state.dq, dtype=float).copy())
        z_ref_log.append(T_cmd[14])
        phase_log.append("move")

        if done:
            break

    print("[OK] Reached pre-contact pose (trajectory finished).")

    # =========================================================
    # Phase B1+B2 — Contact search + hold (fresh control session)
    # Starting fresh avoids motion-generator continuity constraints
    # from Phase A's trajectory ending at T_GOAL.
    # =========================================================
    print("[Phase B1] Slow impedance contact search...")

    RAMP_TIME = 0.3   # s — ramp velocity from 0 to approach_speed

    active = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
    state, dt = active.readOnce()
    t_robot += dt.to_sec() if hasattr(dt, "to_sec") else float(dt)

    # Start from the robot's actual pose after Phase A
    T_ref   = np.array(state.O_T_EE, dtype=float).copy()
    p_ref   = get_pos_from_T(T_ref)
    z_start = p_ref[2]

    t_search      = 0.0
    t_hold        = 0.0
    contact_found = False
    phase         = "search"

    # Give controller an initial target before the loop starts
    active.writeOnce(CartesianPose(T_ref.tolist()))

    while True:
        state, dt = active.readOnce()
        dt_robot = dt.to_sec() if hasattr(dt, "to_sec") else float(dt)
        t_robot += dt_robot
        wrench = np.array(state.O_F_ext_hat_K, dtype=float)

        # ---- Phase B1: slow descend until contact ----
        if phase == "search":
            t_search += dt_robot
            F_meas = force_sign * wrench[force_axis]

            if F_meas >= contact_thresh:
                contact_found = True
                print(f"[OK] Contact detected. Measured force = {F_meas:.3f} N")
                print("[Phase B2] Hold current contact pose for measurement...")
                phase = "hold"
            elif z_start - p_ref[2] > max_down_disp:
                print("[WARN] Maximum downward displacement reached before contact.")
                print("[Phase B2] No reliable contact found, hold current pose anyway...")
                phase = "hold"
            elif t_search >= search_time:
                print("[Phase B2] No reliable contact found, hold current pose anyway...")
                phase = "hold"
            else:
                ramp = min(1.0, t_search / RAMP_TIME)
                p_ref[2] -= approach_speed * ramp * dt_robot
                T_ref = set_pos_in_T(T_ref, p_ref)

            active.writeOnce(CartesianPose(T_ref.tolist()))
            z_ref       = T_ref[14]
            phase_label = "search"

        # ---- Phase B2: hold position for measurement ----
        elif phase == "hold":
            t_hold += dt_robot
            done = t_hold >= hold_time
            cp   = CartesianPose(T_ref.tolist())
            if done:
                cp.motion_finished = True
            active.writeOnce(cp)
            z_ref       = T_ref[14]
            phase_label = "hold"

            if done:
                break

        states.append(state)
        t_log.append(t_robot)
        wrench_franka_log.append(wrench.copy())
        tau_meas_log.append(np.array(state.tau_J,   dtype=float).copy())
        tau_des_log.append(np.array(state.tau_J_d, dtype=float).copy())
        q_log.append(np.array(state.q,   dtype=float).copy())
        dq_log.append(np.array(state.dq, dtype=float).copy())
        z_ref_log.append(z_ref)
        phase_log.append(phase_label)

    robot.stop()

    print("[OK] Impedance contact measurement finished.")

    return {
        "model": model,
        "t": np.array(t_log, dtype=float),
        "states": states,
        "wrench_franka": np.vstack(wrench_franka_log) if len(wrench_franka_log) > 0 else np.zeros((0, 6)),
        "tau_meas": np.vstack(tau_meas_log) if len(tau_meas_log) > 0 else np.zeros((0, 7)),
        "tau_des": np.vstack(tau_des_log) if len(tau_des_log) > 0 else np.zeros((0, 7)),
        "q": np.vstack(q_log) if len(q_log) > 0 else np.zeros((0, 7)),
        "dq": np.vstack(dq_log) if len(dq_log) > 0 else np.zeros((0, 7)),
        "z_ref": np.array(z_ref_log, dtype=float),
        "phase": np.array(phase_log, dtype=object),
        "contact_found": bool(contact_found),
    }


# ============================================================
# Phase 2: Offline KF + plots
# ============================================================

def offline_kf_and_plots(model, t, tau_meas, states, wrench_franka,
                         exp_dir, dt_kf, Qp, QF, R, Af, use_gravity,
                         observer="ccfe", hg_bandwidth=10.0, hg_alg="linear"):
    if len(states) == 0 or len(t) == 0:
        print("[ERROR] No samples for offline KF.")
        return

    pin_model = pin.buildModelFromUrdf(FRANKA_URDF)
    pin_data  = pin_model.createData()

    if observer == "ccfe":
        obs = KalmanCCFE_Wahrburg2015(Ts=dt_kf, Qc_p=Qp, Qc_f=QF, Rc=R, Af_alpha=Af)
        obs_label = "CCFE-KF"
    elif observer == "hgo":
        obs = HighGainObserver(dt=dt_kf, nv=7, bandwidth=hg_bandwidth, alg=hg_alg)
        obs_label = f"HGO({hg_alg})"
    else:
        raise ValueError(f"Unknown observer: {observer!r}. Choose ccfe or hgo.")

    t_end = float(t[-1])
    t_grid = np.arange(0.0, t_end, dt_kf)

    idx = 0
    N = len(t)

    wrench_kf = []
    wrench_franka_ds = []
    t_use = []
    tau_used = []

    for tg in t_grid:
        while idx < N - 1 and t[idx] < tg:
            idx += 1

        s = states[idx]
        q  = np.array(s.q,  dtype=float)
        dq = np.array(s.dq, dtype=float)

        M = np.array(model.mass(s), dtype=float).reshape((7, 7), order="F")
        g = np.array(model.gravity(s), dtype=float)
        J = np.array(model.zero_jacobian(s), dtype=float).reshape((6, 7), order="F")

        # Coriolis matrix C(q, dq) via Pinocchio
        C_mat = pin.computeCoriolisMatrix(pin_model, pin_data, q, dq)

        p_meas = M @ dq

        # u = tau_meas + C^T*dq - g  (momentum formulation, gravity always subtracted)
        u = tau_meas[idx] + C_mat.T @ dq - g

        if observer == "ccfe":
            f_hat = obs.step(J=J, u=u, y=p_meas)
        else:  # hgo
            tau_ext_hat, _ = obs.update(v=dq, M=M, C=C_mat, g=g, tau_motor=tau_meas[idx])
            # Map joint-space torques to Cartesian wrench: f = J^{T+} @ tau_ext
            f_hat, _, _, _ = np.linalg.lstsq(J.T, tau_ext_hat, rcond=None)

        wrench_kf.append( - f_hat)
        wrench_franka_ds.append(wrench_franka[idx])
        t_use.append(t[idx])
        tau_used.append(tau_meas[idx].copy())

    wrench_kf = np.vstack(wrench_kf)
    wrench_franka_ds = np.vstack(wrench_franka_ds)
    t_use = np.array(t_use)

    np.savez(
        os.path.join(exp_dir, f"log_offline_{observer}.npz"),
        t=t_use,
        wrench_kf=wrench_kf,
        wrench_franka=wrench_franka_ds,
        tau_meas=np.vstack(tau_used),
        dt_kf=float(dt_kf),
        observer=observer,
        Qp=Qp,
        QF=QF,
        R=R,
        Af=Af,
        use_gravity=use_gravity,
    )

    labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    units  = ["N", "N", "N", "Nm", "Nm", "Nm"]

    def save_compare(title, ylab, idxs, fname):
        plt.figure(figsize=(10, 5))
        for i in idxs:
            plt.plot(t_use, wrench_franka_ds[:, i], "--", alpha=0.6,
                     label=f"{labels[i]} Franka")
            plt.plot(t_use, wrench_kf[:, i], label=f"{labels[i]} {obs_label}")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel(ylab)
        plt.grid(True)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, fname), dpi=200)
        plt.close()

    save_compare("External FORCE comparison (O frame)", "Force [N]",
                 [0, 1, 2], "compare_force_xyz.png")
    save_compare("External TORQUE comparison (O frame)", "Torque [Nm]",
                 [3, 4, 5], "compare_torque_xyz.png")

    fn_franka = np.linalg.norm(wrench_franka_ds[:, :3], axis=1)
    fn_kf = np.linalg.norm(wrench_kf[:, :3], axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(t_use, fn_franka, "--", alpha=0.7, label="||F|| Franka")
    plt.plot(t_use, fn_kf, label=f"||F|| {obs_label}")
    plt.title("Force magnitude comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("N")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "compare_force_norm.png"), dpi=200)
    plt.close()

    for i in range(6):
        plt.figure(figsize=(10, 4))
        plt.plot(t_use, wrench_franka_ds[:, i], "--", alpha=0.7,
                 label=f"{labels[i]} Franka")
        plt.plot(t_use, wrench_kf[:, i],
                 label=f"{labels[i]} {obs_label}")
        plt.title(f"{labels[i]} comparison")
        plt.xlabel("Time [s]")
        plt.ylabel(units[i])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"compare_{labels[i]}.png"), dpi=200)
        plt.close()

    print(f"[OK] Offline {obs_label} results saved to: {exp_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot-ip", default="10.90.90.10")
    parser.add_argument("--hold-time", type=float, default=5.0)
    parser.add_argument("--move-time", type=float, default=3.0)
    parser.add_argument("--search-time", type=float, default=4.0)

    parser.add_argument("--approach-speed", type=float, default=0.0015)
    parser.add_argument("--contact-thresh", type=float, default=2.0)
    parser.add_argument("--max-down-disp", type=float, default=0.010)
    parser.add_argument("--force-axis", type=int, default=2)
    parser.add_argument("--force-sign", type=float, default=1.0)

    parser.add_argument("--dt-kf", type=float, default=0.001)

    parser.add_argument("--Qp", type=str, default=None)
    parser.add_argument("--QF", type=str, default=None)
    parser.add_argument("--Af", type=str, default=None)
    parser.add_argument("--R",  type=str, default=None)

    parser.add_argument("--use-gravity", action="store_true")
    parser.add_argument("--outroot", default="experiments")

    parser.add_argument("--observer", choices=["ccfe", "hgo"], default="ccfe",
                        help="Observer type: ccfe (Kalman CCFE) or hgo (High-Gain)")
    parser.add_argument("--hg-bandwidth", type=float, default=50.0,
                        help="HGO bandwidth (rad/s), used when --observer hgo")
    parser.add_argument("--hg-alg", choices=["linear", "sos", "sliding"], default="linear",
                        help="HGO injection algorithm")

    args = parser.parse_args()

    Qp = parse_scalar_or_vector(args.Qp, 7, "Qp")
    QF = parse_scalar_or_vector(args.QF, 6, "QF")
    Af = parse_scalar_or_vector(args.Af, 6, "Af")
    R  = parse_scalar_or_vector(args.R,  7, "R")

    if Qp is None:
        Qp = np.array(DEFAULT_QP, dtype=float)
    if QF is None:
        QF = np.array(DEFAULT_QF, dtype=float)
    if Af is None:
        Af = np.array(DEFAULT_AF, dtype=float)
    if R is None:
        R  = np.array(DEFAULT_R, dtype=float)

    exp_dir = create_experiment_dir(args.outroot)

    logs = run_collect(
        robot_ip=args.robot_ip,
        move_time=args.move_time,
        search_time=args.search_time,
        hold_time=args.hold_time,
        approach_speed=args.approach_speed,
        contact_thresh=args.contact_thresh,
        max_down_disp=args.max_down_disp,
        force_axis=args.force_axis,
        force_sign=args.force_sign,
    )

    offline_kf_and_plots(
        logs["model"],
        logs["t"],
        logs["tau_meas"],
        logs["states"],
        logs["wrench_franka"],
        exp_dir,
        args.dt_kf,
        Qp, QF, R, Af, args.use_gravity,
        observer=args.observer,
        hg_bandwidth=args.hg_bandwidth,
        hg_alg=args.hg_alg,
    )

    print("✅ Done:", exp_dir)


if __name__ == "__main__":
    main()