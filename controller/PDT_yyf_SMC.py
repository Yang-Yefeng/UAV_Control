import numpy as np
from scipy.special import gamma

class pdt_yyf_smc_param:
    def __init__(self,
                 a_s: np.ndarray = np.array([0.5, 0.5, 0.5]).astype(float),     # 滑模里面的 alpha
                 b1_s: np.ndarray = np.ones(3).astype(float),                    # 滑模里面的 beta1
                 b2_s: np.ndarray = np.ones(3).astype(float),                    # 滑模里面的 beta2
                 b3_s: np.ndarray = np.ones(3).astype(float),                    # 滑模里面的 beta3
                 Ts: np.ndarray = np.array([5., 5., 5.]).astype(float),             # 滑模里面的预设时间
                 k1_s: np.ndarray = np.array([1., 1., 1.]).astype(float),           # 滑模里面的 kappa1
                 a_c: np.ndarray = np.array([0.5, 0.5, 0.5]).astype(float),     # 控制器里面的 alpha
                 b1_c: np.ndarray = np.ones(3).astype(float),                    # 控制器里面的 beta1
                 b2_c: np.ndarray = np.ones(3).astype(float),                    # 控制器里面的 beta2
                 b3_c: np.ndarray = np.ones(3).astype(float),                    # 控制器里面的 beta3
                 Tc: np.ndarray = np.array([10., 10., 10.]).astype(float),             # 控制器里面的预设时间
                 k1_c: np.ndarray = np.array([1., 1., 1.]).astype(float),           # 控制器里面的 kappa1
                 k2:np.ndarray=np.array([0., 0., 0.]).astype(float),        # 补偿
                 dim: int = 3,
                 dt: float = 0.01):
        self.a_s = a_s
        self.b1_s = b1_s
        self.b2_s = b2_s
        self.b3_s = b3_s
        self.k1_s = k1_s
        self.Ts = Ts

        self.a_c = a_c
        self.b1_c = b1_c
        self.b2_c = b2_c
        self.b3_c = b3_c
        self.k1_c = k1_c
        self.Tc = Tc
        self.k2 = k2

        self.dim = dim
        self.dt = dt


class pdt_yyf_smc:
    def __init__(self,
                 param: pdt_yyf_smc_param = None,
                 a_s: np.ndarray = np.array([0.5, 0.5, 0.5]).astype(float),
                 b1_s: np.ndarray = np.ones(3).astype(float),
                 b2_s: np.ndarray = np.ones(3).astype(float),
                 b3_s: np.ndarray = np.ones(3).astype(float),
                 Ts: np.ndarray = np.array([5., 5., 5.]).astype(float),
                 k1_s: np.ndarray = np.array([1., 1., 1.]).astype(float),
                 a_c: np.ndarray = np.array([0.5, 0.5, 0.5]).astype(float),
                 b1_c: np.ndarray = np.ones(3).astype(float),
                 b2_c: np.ndarray = np.ones(3).astype(float),
                 b3_c: np.ndarray = np.ones(3).astype(float),
                 Tc: np.ndarray = np.array([5., 5., 5.]).astype(float),
                 k1_c: np.ndarray = np.array([1., 1., 1.]).astype(float),
                 k2: np.ndarray = np.array([0., 0., 0.]).astype(float),
                 dim: int = 3,
                 dt: float = 0.01
                 ):
        self.a_s = a_s if param is None else param.a_s
        self.b1_s = b1_s if param is None else param.b1_s
        self.b2_s = b2_s if param is None else param.b2_s
        self.b3_s = b3_s if param is None else param.b3_s
        self.Ts = Ts if param is None else param.Ts
        self.k1_s = k1_s if param is None else param.k1_s

        a1s = (1 - self.a_s * self.k1_s) / (2 - 2 * self.a_s)
        a2s = self.k1_s - a1s
        self.k0_s = gamma(a1s) * gamma(a2s) / gamma(self.k1_s) / (2 - 2 * self.a_s) / self.b2_s ** self.k1_s * (self.b2_s / self.b3_s) ** a1s / self.Ts

        self.a_c = a_c if param is None else param.a_c
        self.b1_c = b1_c if param is None else param.b1_c
        self.b2_c = b2_c if param is None else param.b2_c
        self.b3_c = b3_c if param is None else param.b3_c
        self.Tc = Tc if param is None else param.Tc
        self.k1_c = k1_c if param is None else param.k1_c

        a1c = (1 - self.a_s * self.k1_s) / (2 - 2 * self.a_s)
        a2c = self.k1_s - a1c
        self.k0_c = gamma(a1c) * gamma(a2c) / gamma(self.k1_c) / (2 - 2 * self.a_c) / self.b2_c ** self.k1_c * (self.b2_c / self.b3_c) ** a1c / self.Tc

        self.k2 = k2 if param is None else param.k2
        self.dt = dt if param is None else param.dt
        self.dim = dim if param is None else param.dim
        self.s = np.zeros(self.dim)
        self.control_in = np.zeros(self.dim)
        self.control_out = np.zeros(self.dim)

    def print_param(self):
        print('==== For sliding surface ====')
        print('alpha_s:', self.a_s)
        print('b1_s:', self.b1_s)
        print('b2_s:', self.b2_s)
        print('b3_s:', self.b3_s)
        print('k1_s:', self.k1_s)
        print('Ts:', self.Ts)
        print('k0_s:', self.k0_s)
        print('==== For sliding surface ====')

        print('==== For controller ====')
        print('alpha_c:', self.a_c)
        print('b1_c:', self.b1_c)
        print('b2_c:', self.b2_c)
        print('b3_c:', self.b3_c)
        print('k1_c:', self.k1_c)
        print('Tc:', self.Tc)
        print('k0_c:', self.k0_c)
        print('==== For controller ====')

    @staticmethod
    def sig(x, a, kt=5):
        return np.fabs(x) ** a * np.tanh(kt * x)

    @staticmethod
    def sig_sign(x, a, th=0.01):
        res = []
        for _x, _a in zip(x, a):
            if np.fabs(_x) <= th:  # 这个数奇异
                # res.append(np.sin(np.pi / 2 / th) * np.fabs(_x) ** (-_a))
                res.append(0.)
            else:
                if _x < 0:
                    res.append(np.fabs(_x) ** _a * np.sign(_x))
                else:
                    res.append(np.fabs(_x) ** _a * np.sign(_x))
        return np.array(res)

    @staticmethod
    def singu(x, a, th=0.01):
        # 专门处理奇异的函数
        x = np.fabs(x)
        res = []
        for _x, _a in zip(x, a):
            if _x > th:
                # res.append(self.sig(_x, _a, kt))
                res.append(1.0)
            else:
                res.append(np.sin(np.pi / 2 / th) * _x ** (-_a))
        return np.array(res)

    @staticmethod
    def sig_sign_singu(x, a, th=0.01):
        res = []
        for _x, _a in zip(x, a):
            if np.fabs(_x) <= th:   # 这个数奇异
                res.append(np.sin(np.pi / 2 / th) * np.fabs(_x) ** np.fabs(_a))
                # res.append(0.)
            else:
                if _x < 0:
                    res.append(np.fabs(_x) ** _a * np.tanh(5 * _x))
                else:
                    res.append(np.fabs(_x) ** _a * np.tanh(5 * _x))
        return np.array(res)

    def control_update_inner(self,
                             e_rho: np.ndarray,
                             dot_e_rho: np.ndarray,
                             dd_ref: np.ndarray,
                             A_rho: np.ndarray,
                             B_rho: np.ndarray,
                             obs: np.ndarray,
                             att_only: bool = False):
        if not att_only:
            dd_ref = np.zeros(self.dim)

        '''calculate sliding mode'''
        _s1 = self.b1_s * self.sig_sign_singu(e_rho, 2 - 1 / self.k1_s)
        _s2 = self.b2_s * self.sig_sign_singu(e_rho, 2 * self.a_s - 1 / self.k1_s) / 2.0
        _s3 = self.b3_s * self.sig_sign_singu(e_rho, 4 - 2 * self.a_s - 1 / self.k1_s) / 2.0
        self.s = dot_e_rho + self.k0_s * self.sig_sign_singu(_s1 + _s2 + _s3, self.k1_s)      # 改过
        '''calculate sliding mode'''

        _m1 = self.b1_s * self.sig_sign_singu(e_rho, 2 - 1 / self.k1_s)
        _m2 = self.b2_s * self.sig_sign_singu(e_rho, 2 * self.a_s - 1 / self.k1_s) / 2.0
        _m3 = self.b3_s * self.sig_sign_singu(e_rho, 4 - 2 * self.a_s - 1 / self.k1_s) / 2.0
        M_rho1 = self.k0_s * self.k1_s * self.sig_sign_singu(_m1 + _m2 + _m3, self.k1_s - 1)      # 改过
        M_rho2 = (self.b1_s * (2 - 1 / self.k1_s) * self.sig_sign_singu(e_rho, 1 - 1 / self.k1_s) +
                  self.b2_s * (2 * self.a_s - 1 / self.k1_s) * self.singu(e_rho, 2 * self.a_s - 1 / self.k1_s - 1) / 2.0 +
                  self.b3_s * (4 - 2 * self.a_s - 1 / self.k1_s) * self.sig_sign_singu(e_rho, 3 - 2 * self.a_s - 1 / self.k1_s) / 2.0) * dot_e_rho
        M_rho = M_rho1 * M_rho2

        tau1 = A_rho - dd_ref + M_rho       # 补偿模型
        tau2 = obs      # 观测器
        _t1 = self.b1_c * self.sig_sign_singu(self.s, 2 - 1 / self.k1_c)
        _t2 = self.b2_c * self.sig_sign_singu(self.s, 2 * self.a_c - 1 / self.k1_c) / 2.0
        _t3 = self.b3_c * self.sig_sign_singu(self.s, 4 - 2 * self.a_c - 1 / self.k1_c) / 2.0
        tau3 = self.k0_c * self.sig_sign_singu(_t1 + _t2 + _t3, self.k1_c) + self.k2 * np.tanh(5 * self.s)
        self.control_in = -np.dot(np.linalg.inv(B_rho), tau1 + tau2 + tau3)

    def control_update_outer(self,
                             e_eta: np.ndarray,
                             dot_e_eta: np.ndarray,
                             dot_eta: np.ndarray,
                             kt: float,
                             m: float,
                             dd_ref: np.ndarray,
                             obs: np.ndarray,
                             e_m: np.ndarray = np.array([2, 2, 2]).astype(float),
                             de_m: np.ndarray = np.array([1.5, 1.5, 1.5]).astype(float)):
        e_eta = np.clip(e_eta, -e_m, e_m)
        dot_e_eta = np.clip(dot_e_eta, -de_m, de_m)

        '''calculate sliding mode'''
        _s1 = self.b1_s * self.sig_sign_singu(e_eta, 2 - 1 / self.k1_s)
        _s2 = self.b2_s * self.sig_sign_singu(e_eta, 2 * self.a_s - 1 / self.k1_s) / 2.0
        _s3 = self.b3_s * self.sig_sign_singu(e_eta, 4 - 2 * self.a_s - 1 / self.k1_s) / 2.0
        self.s = dot_e_eta + self.k0_s * self.sig_sign_singu(_s1 + _s2 + _s3, self.k1_s)
        '''calculate sliding mode'''

        _m1 = self.b1_s * self.sig_sign_singu(e_eta, 2 - 1 / self.k1_s)
        _m2 = self.b2_s * self.sig_sign_singu(e_eta, 2 * self.a_s - 1 / self.k1_s) / 2.0
        _m3 = self.b3_s * self.sig_sign_singu(e_eta, 4 - 2 * self.a_s - 1 / self.k1_s) / 2.0
        M_rho1 = self.k0_s * self.k1_s * self.sig_sign_singu(_m1 + _m2 + _m3, self.k1_s - 1)
        M_rho2 = (self.b1_s * (2 - 1 / self.k1_s) * self.sig_sign_singu(e_eta, 1 - 1 / self.k1_s) +
                  self.b2_s * (2 * self.a_s - 1 / self.k1_s) * self.singu(e_eta, 2 * self.a_s - 1 / self.k1_s - 1) / 2.0 +
                  self.b3_s * (4 - 2 * self.a_s - 1 / self.k1_s) * self.sig_sign_singu(e_eta, 3 - 2 * self.a_s - 1 / self.k1_s) / 2.0) * dot_e_eta
        M_rho = M_rho1 * M_rho2

        u1 = kt / m * dot_eta - dd_ref + M_rho
        u2 = obs
        _t1 = self.b1_c * self.sig_sign_singu(self.s, 2 - 1 / self.k1_c)
        _t2 = self.b2_c * self.sig_sign_singu(self.s, 2 * self.a_c - 1 / self.k1_c) / 2.0
        _t3 = self.b3_c * self.sig_sign_singu(self.s, 4 - 2 * self.a_c - 1 / self.k1_c) / 2.0
        u3 = self.k0_c * self.sig_sign_singu(_t1 + _t2 + _t3, self.k1_c) + self.k2 * np.tanh(5 * self.s)
        self.control_out = -(u1 + u2 + u3)
