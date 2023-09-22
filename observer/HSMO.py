import numpy as np


def sig(_x: np.ndarray, _a: float):
    return np.fabs(_x) ** _a * np.sign(_x)


class hsmo:
    def __init__(self, alpha1: np.ndarray, alpha2: np.ndarray, alpha3: np.ndarray, p: np.ndarray, dim: int, dt: float):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.p = p
        self.dt = dt
        self.dim = dim
        self.z1 = np.zeros(self.dim)
        self.a1 = np.zeros(self.dim)
        self.z2 = np.zeros(self.dim)
        self.a2 = np.zeros(self.dim)
        self.z3 = np.zeros(self.dim)

        self.dot_z1 = np.zeros(self.dim)
        self.dot_z2 = np.zeros(self.dim)
        self.dot_z3 = np.zeros(self.dim)

    def set_init(self, de0: np.ndarray):
        self.z1 = de0

    def observe(self, de: np.ndarray, syst_dynamic: np.ndarray):
        self.dot_z1 = syst_dynamic + self.a1
        self.a1 = -self.alpha1 * self.p ** (1 / 3) * sig(self.z1 - de, 2 / 3) + self.z2
        self.dot_z2 = self.a2
        self.a2 = -self.alpha2 * self.p ** (1 / 2) * sig(self.z2 - self.a1, 1 / 2) + self.z3
        self.dot_z3 = -self.alpha3 * self.p * np.sign(self.z3 - self.a2)

        self.z1 = self.z1 + self.dot_z1 * self.dt
        self.z2 = self.z2 + self.dot_z2 * self.dt
        self.z3 = self.z3 + self.dot_z3 * self.dt
        return self.z2, self.z3
