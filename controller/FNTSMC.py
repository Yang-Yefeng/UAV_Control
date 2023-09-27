import numpy as np


class fntsmc:
    def __init__(self,
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
        self.k1 = k1  # 第一维度控制 y ，第二维度控制 x
        self.k2 = k2
        # self.k2 = np.array([1.0, 1.0])
        self.k3 = k3
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lmd = lmd
        self.dt = dt

        self.dim = dim

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
