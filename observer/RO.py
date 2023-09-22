import numpy as np


class ro:
    def __init__(self,
                 k1: np.ndarray,
                 k2: np.ndarray,
                 dim: int,
                 dt: float):
        self.k1 = k1  # 干扰估计增益
        self.k2 = k2  # 速度估计增益
        self.dt = dt  # 原始系统采样时间
        self.dim = dim
        self.delta_obs = np.zeros(self.dim)
        self.dot_delta_obs = np.zeros(self.dim)
        self.de_obs = np.zeros(self.dim)
        self.dot_de_obs = np.zeros(self.dim)

    def observe(self,
                syst_dynamic: np.ndarray,
                de: np.ndarray):
        self.dot_delta_obs = self.k1 * (de - self.de_obs)
        self.dot_de_obs = self.delta_obs + syst_dynamic + self.k2 * (de - self.de_obs)

        self.delta_obs += self.dt * self.dot_delta_obs
        self.de_obs += self.dt * self.dot_de_obs
        return self.delta_obs, self.dot_delta_obs

    def set_init(self, de: np.ndarray):
        self.de_obs = de
