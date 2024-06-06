import numpy as np


class recursive_fntsmc_param:
    def __init__(self):
        self.k1: np.ndarray = np.array([1.2, 0.8, 1.5])
        self.k2: np.ndarray = np.array([0.2, 0.6, 1.5])
        self.k3: np.ndarray = np.array([0.05, 0.05, 0.05])
        self.alpha: np.ndarray = np.array([1.2, 1.5, 1.2])
        self.beta: np.ndarray = np.array([0.3, 0.3, 0.3])
        self.gamma: np.ndarray = np.array([0.2, 0.2, 0.2])
        self.lmd: np.ndarray = np.array([2.0, 2.0, 2.0])
        self.dim: int = 3
        self.dt: float = 0.01
        self.ctrl0: np.ndarray = np.array([0., 0., 0.])

    def print_param(self):
        print('==== PARAM ====')
        print('k1:     ', self.k1)
        print('k2:     ', self.k2)
        print('alpha:  ', self.alpha)
        print('beta:   ', self.beta)
        print('gamma:  ', self.gamma)
        print('lambda: ', self.lmd)
        print('dim:    ', self.dim)
        print('dt', self.dt)
        print('ctrl0:', self.ctrl0)
        print('==== PARAM ====')


class recursive_fntsmc:
    def __init__(self,
                 param: recursive_fntsmc_param = None,
                 k1: np.ndarray = np.array([1.2, 0.8, 1.5]),  # 1.2, 1.5
                 k2: np.ndarray = np.array([0.2, 0.6, 1.5]),  # 0.2, 0.5
                 k3: np.ndarray = np.array([0.05, 0.05, 0.05]),
                 alpha: np.ndarray = np.array([1.2, 1.5, 1.2]),
                 beta: np.ndarray = np.array([0.3, 0.3, 0.3]),
                 gamma: np.ndarray = np.array([0.2, 0.2, 0.2]),
                 lmd: np.ndarray = np.array([2.0, 2.0, 2.0]),
                 dim: int = 3,
                 dt: float = 0.001,
                 ctrl0: np.ndarray = np.array([0., 0., 0.])):

        self.k1 = k1 if param is None else param.k1
        self.k2 = k2 if param is None else param.k2
        self.k3 = k3 if param is None else param.k3
        self.alpha = alpha if param is None else param.alpha
        self.beta = beta if param is None else param.beta
        self.gamma = gamma if param is None else param.gamma
        self.lmd = lmd if param is None else param.lmd
        self.dt = dt if param is None else param.dt
        self.dim = dim if param is None else param.dim

        self.sigma_o = np.zeros(self.dim)
        self.dot_sigma_o1 = np.zeros(self.dim)
        self.sigma_o1 = np.zeros(self.dim)
        self.so = self.sigma_o + lmd * self.sigma_o1
        # self.ctrl = np.zeros(self.dim)
        self.control = ctrl0

    def control_update(self, kp: float, m: float, vel: np.ndarray, e: np.ndarray, de: np.ndarray, dd_ref: np.ndarray, obs: np.ndarray):
        """
        :param kp:
        :param m:
        :param vel:
        :param e:
        :param de:
        :param dd_ref:
        :param obs:
        :brief:         输出为 x y z 三轴的虚拟的加速度
        :return:
        """
        k_tanh_e = 5
        k_tanh_sigma0 = 5
        self.sigma_o = de + self.k1 * e + self.gamma * np.fabs(e) ** self.alpha * np.tanh(k_tanh_e * e)
        self.dot_sigma_o1 = np.fabs(self.sigma_o) ** self.beta * np.tanh(k_tanh_sigma0 * self.sigma_o)
        self.sigma_o1 += self.dot_sigma_o1 * self.dt
        self.so = self.sigma_o + self.lmd * self.sigma_o1

        uo1 = kp / m * vel + dd_ref - self.k1 * de - self.gamma * self.alpha * np.fabs(e) ** (self.alpha - 1) * de - self.lmd * self.dot_sigma_o1
        uo2 = -self.k2 * self.so - obs

        self.control = uo1 + uo2

    def control_update2(self,
                        second_order_att_dynamics: np.ndarray,
                        control_mat: np.ndarray,
                        e: np.ndarray,
                        de: np.ndarray,
                        dd_ref: np.ndarray,
                        obs: np.ndarray):
        k_tanh_e = 5
        k_tanh_s = 5
        k_tanh_sigma = 10
        self.sigma_o = 1 * de + self.k1 * e + self.gamma * np.fabs(e) ** self.alpha * np.tanh(k_tanh_e * e)
        self.dot_sigma_o1 = np.fabs(self.sigma_o) ** self.beta * np.tanh(k_tanh_s * self.sigma_o)
        self.sigma_o1 += self.dot_sigma_o1 * self.dt
        self.sigma = self.sigma_o + self.lmd * self.sigma_o1

        u1 = second_order_att_dynamics + dd_ref + self.k1 * de + self.gamma * self.alpha * np.fabs(e) ** (self.alpha - 1) * de + self.lmd * self.dot_sigma_o1
        u2 = -self.k2 * np.tanh(k_tanh_sigma * self.sigma) - obs
        # u2 = -self.k2 * self.sigma

        self.control = -np.dot(np.linalg.inv(control_mat), u1 + u2)
