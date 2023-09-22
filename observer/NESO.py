import numpy as np


class neso:
    def __init__(self,
                 l1: np.ndarray,
                 l2: np.ndarray,
                 l3: np.ndarray,
                 r: np.ndarray,
                 k1: np.ndarray,
                 k2: np.ndarray,
                 dim: int,
                 dt:float):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.r = r
        self.k1 = k1
        self.k2 = k2

        self.dt = dt
        self.dim = dim

        self.z1 = np.zeros(dim)
        self.z2 = np.zeros(dim)
        self.z3 = np.zeros(dim)  # 这个是输出
        self.xi = np.zeros(dim)
        self.dot_z1 = np.zeros(dim)
        self.dot_z2 = np.zeros(dim)
        self.dot_z3 = np.zeros(dim)

    def fal(self, xi: np.ndarray):
        res = []
        for i in range(self.dim):
            if np.fabs(xi[i]) <= self.k2[i]:
                res.append(xi[i] / (self.k2[i] ** (1 - self.k1[i])))
            else:
                res.append(np.fabs(xi[i]) ** self.k1[i] * np.sign(xi[i]))
        return np.array(res)

    def set_init(self, x0: np.ndarray, dx0: np.ndarray, syst_dynamic: np.ndarray):
        self.z1 = x0  # 估计x
        self.z2 = dx0  # 估计dx
        self.z3 = np.zeros(self.dim)  # 估计干扰
        self.xi = x0 - self.z1
        self.dot_z1 = self.z2 + self.l1 / self.r * self.fal(xi=self.r ** 2 * self.xi)
        self.dot_z2 = self.z3 + self.l2 * self.fal(xi=self.r ** 2 * self.xi) + syst_dynamic
        self.dot_z3 = self.r * self.l3 * self.fal(xi=self.r ** 2 * self.xi)

    def observe(self, x: np.ndarray, syst_dynamic: np.ndarray):
        self.xi = x - self.z1
        self.dot_z1 = self.z2 + self.l1 / self.r * self.fal(xi=self.r ** 2 * self.xi)
        self.dot_z2 = self.z3 + self.l2 * self.fal(xi=self.r ** 2 * self.xi) + syst_dynamic
        self.dot_z3 = self.r * self.l3 * self.fal(xi=self.r ** 2 * self.xi)
        self.z1 = self.z1 + self.dot_z1 * self.dt
        self.z2 = self.z2 + self.dot_z2 * self.dt
        self.z3 = self.z3 + self.dot_z3 * self.dt
        delta_obs = self.z3.copy()
        dot_delta_obs = self.dot_z3.copy()
        return delta_obs, dot_delta_obs
