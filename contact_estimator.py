import numpy as np
import yaml

"""
The external force estimator module provides a class for estimating external forces
using a generalized momentum observer. The observer assumes only one contact point
and estimates the external forces acting on a rigid body system.
"""
def soft_sign(x, k=100):
    kx = k*x
    return np.tanh(kx)

def sign(x, use_soft_sign=True):
    if use_soft_sign:
        return soft_sign(x)
    else:
        return np.sign(x)
    
def alpha_func(err, alpha, use_soft_sign=True):
    return sign(err, use_soft_sign) * (np.abs(err) ** alpha)
    
def q_func(err, k_hg, k_sliding, use_soft_sign=True):
    return k_sliding * alpha_func(err, 1/2, use_soft_sign) + k_hg * err

def err_mapping_func(err, alg="hg", k_hg=1, k_sliding=1, use_soft_sign=True):
    if alg == "hg":
        return k_hg*err, k_hg*err
    elif alg == "sliding":
        return alpha_func(k_sliding*err, 1/2, use_soft_sign), sign(err, use_soft_sign)
    elif alg == "mixing":
        return q_func(err, k_hg, k_sliding, use_soft_sign), sign(err, use_soft_sign) + q_func(err, k_hg, k_sliding, use_soft_sign)
    else:
        raise NotImplementedError

from functools import reduce
class high_gain_based_observer(object):
    def __init__(self, dt, alg, nv):
        # load yaml config
        with open("./config.yaml", "r") as f:
            config = yaml.safe_load(f)
            # Merge the list of dictionaries into a single dictionary
            bandwidth = config["bandwidth"]
            bandwidth = reduce(lambda a, b: a | b, bandwidth)
            alg = config["hg_alg"]
        
        self.dt = dt
        self.est_ext_tau = np.zeros(nv)
        self.est_gm = np.zeros(nv)
        self.alg = alg
        bandwidth = bandwidth[alg]
        bandwidth = [bandwidth] * nv
        bandwidth = np.array(bandwidth)
        self.L1 = np.diag(2 * bandwidth)
        self.L2 = np.diag(bandwidth ** 2)
        
    def update(self, v, M, C, g, tau_motor):
        gt_gm = M @ v
        err = gt_gm - self.est_gm
        err1, err2 = err_mapping_func(err, self.alg)
        dgm = tau_motor + C.T @ v - g + self.est_ext_tau + self.L1 @ err1
        dest_ext_tau = self.L2 @ err2
        
        self.est_gm += self.dt * dgm
        self.est_ext_tau += self.dt * dest_ext_tau
        
        return self.est_ext_tau, self.est_gm
    
    def reset(self):
        self.est_ext_tau = np.zeros_like(self.est_ext_tau)
        self.est_gm = np.zeros_like(self.est_gm)
    
# ============================================================================ 
# Discretization functions for state space models
# ============================================================================ 
def discretize_state_space_model(A, B, dt, full_tau=False):
    from scipy.linalg import expm
    N = B.shape[1]
    M = np.block([
                    [A, B],
                    [np.zeros((N, N+N)), np.zeros((N, N))],
                ])
    
    M_dis = expm(M * dt)
    A_dis = M_dis[:A.shape[0], :A.shape[1]]
    B_dis = M_dis[:A.shape[0], A.shape[1]:]
    return A_dis, B_dis
    
def discretize_measurement_model(C, R, dt):
    return C, R / dt

def discretize_process_noise_model(A, Q, dt):
    from scipy.linalg import expm
    N_plus_n_ext = A.shape[0]
    H = np.block([
                    [A, Q],
                    [np.zeros((N_plus_n_ext, N_plus_n_ext)), -A.T]
                 ])
    H_dis = expm(H * dt)
    M_11 = H_dis[:N_plus_n_ext, :N_plus_n_ext]
    M_12 = H_dis[:N_plus_n_ext, N_plus_n_ext:]
    M_22 = H_dis[N_plus_n_ext:, N_plus_n_ext:]
    Q_dis = M_12 @ M_11.T
    return Q_dis

def euler_discretize_state_space_model(A, B, dt):
    A_k = np.eye(A.shape[0]) + A * dt
    B_k = B * dt
    return A_k, B_k

def euler_discretize_measurement_model(C, R, dt):
    return C, R / dt   
    
from filterpy.kalman import KalmanFilter
class kalman_disturbance_observer(object):
    def __init__(self, dt, nv):
        # load yaml config
        with open("./config.yaml", "r") as f:
            config = yaml.safe_load(f)
            # Merge the list of dictionaries into a single dictionary
            P_init = config["P_init"]
            Q = config["Q"]
            R = config["R"]
        
        self.dt = dt
        self.nv = nv
        self.kf = KalmanFilter(dim_x=nv + nv, dim_z=nv)
        self.kf.x = np.zeros(nv + nv)
        self.P = np.eye(nv + nv) * P_init
        self.Q = np.eye(nv + nv) * Q
        self.R = np.eye(nv) * R

        A_f = np.zeros((nv, nv))
        A = np.block([
                        [np.zeros((nv, nv)), np.identity(nv)],
                        [np.zeros((nv, nv)), A_f]
                        ])
        B = np.vstack([
                np.identity(nv),
                np.zeros((nv, nv))
                ])
        C = np.hstack([
                        np.identity(nv),
                        np.zeros((nv, nv))
                        ])
    
        A_dis, B_dis = discretize_state_space_model(A, B, dt)
        C_dis, R_dis = discretize_measurement_model(C, self.R, dt)
        Q_dis = discretize_process_noise_model(A, self.Q, dt)
    
        self.A_k = A_dis
        self.B_k = B_dis
        self.C_k = C_dis
        self.R_k = R_dis
        self.Q_k = Q_dis
        
        self.est_ext_tau = np.zeros(nv)
        self.est_gm = np.zeros(nv)
        
    def update(self, v, M, C, g, tau_motor):
        u = tau_motor + C.T @ v - g
        x = np.hstack([self.est_gm, self.est_ext_tau])
        gm_measured = M @ v
        
        self.kf.F = self.A_k
        self.kf.H = self.C_k
        self.kf.Q = self.Q_k
        self.kf.R = self.R_k
        self.kf.P = self.P
        self.kf.x = x
        self.kf.B = self.B_k
        
        self.kf.predict(u=u, B=self.B_k)
        self.kf.update(z=gm_measured)
        
        self.est_gm = self.kf.x[:self.nv]
        self.est_ext_tau = self.kf.x[self.nv:]
        self.P = self.kf.P
        return self.est_ext_tau, self.est_gm
    
if __name__ == "__main__":
    # Example usage
    dt = 0.01
    nv = 6
    observer = high_gain_based_observer(dt, "mixing", nv)