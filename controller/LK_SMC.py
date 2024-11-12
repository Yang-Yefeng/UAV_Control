import numpy as np
import pandas as pd


class lk_smc_param:
    def __init__(self,
                 eta: np.ndarray = np.array([0.5, 0.5, 0.5]).astype(float),
                 gamma2: np.ndarray = np.zeros(3).astype(float),
                 gamma3: np.ndarray = np.zeros(3).astype(float),
                 Ts: np.ndarray = np.array([5., 5., 5.]).astype(float),
                 Tu: np.ndarray = np.array([1., 1., 1.]).astype(float),
                 k2: np.ndarray = np.zeros(3).astype(float),
                 k3: np.ndarray = np.zeros(3).astype(float),
                 delta0: np.ndarray = np.zeros(3).astype(float),
                 w0: np.ndarray = np.array([1., 1., 1.]).astype(float),
                 dim: int = 3,
                 dt: float = 0.01):
        self.eta = eta
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.Ts = Ts
        self.Tu = Tu
        self.k2 = k2
        self.k3 = k3
        self.gamma1 = np.sqrt(self.gamma2 * self.gamma3 / 4) + 1
        self.k1 = np.sqrt(self.k2 * self.k3)
        self.d_delta = np.zeros(3).astype(float)
        self.delta = delta0
        self.c1 = 1.5 * np.ones(3)
        self.c2 = 1 / (2 ** ((1 + self.eta) / 2))
        self.c3 = 2 ** (self.eta - 2)
        self.w0 = w0
        self.dim = dim
        self.dt = dt


class lk_smc:
    def __init__(self,
                 param: lk_smc_param = lk_smc_param(),
                 is_ideal: bool = False):
        self.eta = param.eta
        self.gamma2 = param.gamma2
        self.gamma3 = param.gamma3
        self.Ts = param.Ts
        self.Tu = param.Tu
        self.k1 = param.k1
        self.k2 = param.k2
        self.k3 = param.k3
        self.gamma1 = param.gamma1
        self.c1 = param.c1
        self.c2 = param.c2
        self.c3 = param.c3
        self.dim = param.dim
        self.dt = param.dt
        
        self.s = np.zeros(self.dim)
        self.d_delta = np.zeros(self.dim)
        self.delta = param.delta
        self.h1 = 1.
        self.h2 = 1.
        self.h3 = (3 - self.eta) / (2 - self.eta) * 2 ** (self.eta - 2)
        self.w0 = param.w0
        
        self.is_ideal = is_ideal
        
        self.control_in = np.zeros(self.dim)
        self.control_out = np.zeros(self.dim)
        self.index = 0
        
        self.rec_t = np.empty(shape=[0, 1], dtype=float)
        self.rec_delta = np.empty(shape=[0, 3], dtype=float)
        self.rec_d_delta = np.empty(shape=[0, 3], dtype=float)
    
    @staticmethod
    def sig(x, a, kt=5):
        return np.fabs(x) ** a * np.tanh(kt * x)
    
    @staticmethod
    def singu(x, a, th=0.01):
        # 专门处理奇异的函数
        x = np.fabs(x)
        res = []
        for _x, _a in zip(x, a):
            if _x > th:
                res.append(1.0)
            else:
                res.append(np.sin(np.pi / 2 / th) * _x ** (-_a))
        return np.array(res)
    
    def cal_d_delta(self):
        if not self.is_ideal:
            self.d_delta = -(np.pi / (self.Tu * self.k1 * (1 - self.eta)) *
                             (2 * self.h1 * self.k1 * self.delta + self.h2 * self.k2 * self.delta + self.h3 * self.k3 * np.fabs(self.delta) ** (2 - self.eta))
                             + self.s * np.tanh(self.s / self.w0))
        else:
            self.d_delta = np.zeros(3)
    
    def control_update_inner(self,
                             e_rho: np.ndarray,
                             dot_e_rho: np.ndarray,
                             dd_ref: np.ndarray,
                             A_rho: np.ndarray,
                             B_rho: np.ndarray,
                             obs: np.ndarray,
                             att_only: bool = False,
                             delta_est: bool = False):
        if not att_only:
            dd_ref = np.zeros(self.dim)
        self.s = (dot_e_rho + np.pi / (self.Ts * self.gamma1 * (1 - self.eta)) *
                  (2 * self.gamma1 * e_rho + self.gamma2 * self.sig(e_rho, self.eta) + self.gamma3 * self.sig(e_rho, 2 - self.eta)))
        if delta_est:
            self.cal_d_delta()
            self.delta = self.delta + self.dt * self.d_delta
        A0 = (np.pi / (self.Ts * self.gamma1 * (1 - self.eta)) *
              (2 * self.gamma1 * dot_e_rho
               + self.gamma2 * self.eta * self.singu(e_rho, self.eta - 1) * dot_e_rho
               + self.gamma3 * (2 - self.eta) * self.sig(e_rho, 1 - self.eta) * dot_e_rho)
              )
        
        tau_eq = A_rho - dd_ref + A0
        if self.is_ideal:
            obs = np.zeros(3)
        tau_sw = (np.pi / (self.Tu * self.k1 * (1 - self.eta)) *
                  (2 * self.c1 * self.k1 * self.s + self.c2 * self.k2 * self.sig(self.s, self.eta) + self.c3 * self.k3 * self.sig(self.s, 2 - self.eta))
                  + self.delta * np.tanh(self.s / self.w0)
                  + obs)
        self.control_in = -np.dot(np.linalg.inv(B_rho), tau_eq + tau_sw)
        
        self.rec_t = np.append(self.rec_t, np.array([[self.index * self.dt]]), axis=0)
        self.rec_d_delta = np.append(self.rec_d_delta, [self.d_delta], axis=0)  # 添加整行元素，axis=1添加整列元素
        self.rec_delta = np.append(self.rec_delta, [self.delta], axis=0)
        self.index += 1
    
    def control_update_outer(self,
                             e_eta: np.ndarray,
                             dot_e_eta: np.ndarray,
                             dot_eta: np.ndarray,
                             kt: float,
                             m: float,
                             dd_ref: np.ndarray,
                             obs: np.ndarray,
                             e_m: float = 0.5,
                             de_m: float = 1.5,
                             delta_est: bool = False):
        e_eta = np.clip(e_eta, -e_m, e_m)
        dot_e_eta = np.clip(dot_e_eta, -de_m, de_m)
        self.s = (dot_e_eta + np.pi / (self.Ts * self.gamma1 * (1 - self.eta)) *
                  (2 * self.gamma1 * e_eta + self.gamma2 * self.sig(e_eta, self.eta) + self.gamma3 * self.sig(e_eta, 2 - self.eta)))
        if delta_est:
            self.cal_d_delta()
            self.delta += self.dt * self.d_delta
        A0 = (np.pi / (self.Ts * self.gamma1 * (1 - self.eta)) *
              (2 * self.gamma1 * dot_e_eta
               + self.gamma2 * self.eta * self.singu(e_eta, self.eta - 1) * dot_e_eta
               + self.gamma3 * (2 - self.eta) * self.sig(e_eta, 1 - self.eta) * dot_e_eta)
              )
        u_eq = kt / m * dot_eta - dd_ref + A0
        u_sw = (np.pi / (self.Tu * self.k1 * (1 - self.eta)) *
                (2 * self.c1 * self.k1 * self.s + self.c2 * self.k2 * self.sig(self.s, self.eta) + self.c3 * self.k3 * self.sig(self.s, 2 - self.eta))
                + self.delta * np.tanh(self.s / self.w0)
                + obs)
        self.control_out = -(u_eq + u_sw)
        
        self.rec_t = np.append(self.rec_t, np.array([[self.index * self.dt]]), axis=0)
        self.rec_d_delta = np.append(self.rec_d_delta, [self.d_delta], axis=0)  # 添加整行元素，axis=1添加整列元素
        self.rec_delta = np.append(self.rec_delta, [self.delta], axis=0)
        self.index += 1

    def save_ctrl_adaptive(self, path:str, flag:str):
        pd.DataFrame(np.hstack((self.rec_t, self.rec_d_delta, self.rec_delta)),
                     columns=['time',
                              'd_delta_roll', 'd_delta_pitch', 'd_delta_yaw',
                              'delta_roll', 'delta_pitch', 'delta_yaw']). \
            to_csv(path + flag + '_ctrl_adaptive.csv', sep=',', index=False)
