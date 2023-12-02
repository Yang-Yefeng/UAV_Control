import numpy as np


class robust_differentiator_3rd:
    def __init__(self,
                 m1: np.ndarray,
                 m2: np.ndarray,
                 m3: np.ndarray,
                 n1: np.ndarray,
                 n2: np.ndarray,
                 n3: np.ndarray,
                 dim: int,
                 dt: float):
        self.a1 = 3. / 4.
        self.a2 = 2. / 4.
        self.a3 = 1. / 4.
        self.b1 = 5. / 4.
        self.b2 = 6. / 4.
        self.b3 = 7. / 4.

        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.z1 = np.zeros(dim)
        self.z2 = np.zeros(dim)
        self.z3 = np.zeros(dim)
        self.dz1 = np.zeros(dim)
        self.dz2 = np.zeros(dim)
        self.dz3 = np.zeros(dim)
        self.dt = dt

    def set_init(self, z0: np.ndarray, dz0: np.ndarray):
        """
        :param z0:
        :return:
        """
        self.z1 = z0[0]
        self.z2 = z0[1]
        self.z3 = z0[2]
        self.dz1 = dz0[0]
        self.dz2 = dz0[1]
        self.dz3 = dz0[2]

    @staticmethod
    def sig(x: np.ndarray, a):
        return np.fabs(x) ** a * np.sign(x)

    def observe(self, e: np.ndarray, syst_dynamic: np.ndarray):
        self.dz1 = self.z2 + self.m1 * self.sig(e, self.a1) + self.n1 * self.sig(e, self.b1)
        self.dz2 = syst_dynamic + self.z3 + self.m2 * self.sig(e, self.a2) + self.n2 * self.sig(e, self.b2)
        self.dz3 = self.m3 * self.sig(e, self.a3) + self.n3 * self.sig(e, self.b3)
        self.z1 = self.z1 + self.dz1 * self.dt
        self.z2 = self.z2 + self.dz2 * self.dt
        self.z3 = self.z3 + self.dz3 * self.dt

        return self.z3.copy()
