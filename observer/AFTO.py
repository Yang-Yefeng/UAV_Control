import numpy
import numpy as np


class afto:
    def __init__(self,
                 K: np.ndarray = np.array([20., 20., 20.]),
                 alpha : np.ndarray = np.array([1, 1, 1]),
                 beta: np.ndarray = np.array([10, 10, 10]),
                 p: np.ndarray = np.array([0.5, 0.5, 0.5]),
                 q: np.ndarray = np.array([2, 2, 2]),
                 dt: float = 0.01,
                 dim: int = 3,
                 init_de: np.ndarray = np.array([0., 0., 0.])):
        self.dt = dt
        self.k = K
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q

        self.ef: np.ndarray = -self.k * init_de
        # self.ef: np.ndarray = np.array([0., 0., 0., 0.])
        self.dot_ef = -self.k * self.ef

        self.ef_obs = self.ef
        self.tilde_ef = self.ef - self.ef_obs
        self.dot_ef_obs: np.ndarray = self.alpha * np.fabs(self.tilde_ef) ** self.p * np.sign(self.tilde_ef) + \
                                      self.beta * np.fabs(self.tilde_ef) ** self.q * np.sign(self.tilde_ef) + \
                                      self.dot_ef
        self.delta_obs: np.ndarray = -(self.dot_ef_obs + self.k * self.ef) / self.k
        self.dot_ef = -self.k * (self.ef + self.delta_obs)
        self.dot_delta_obs = np.zeros(dim)
        self.count = 0

    def observe(self,
                syst_dynamic: np.ndarray,
                dot_e_old: np.ndarray,
                dot_e: np.ndarray):
        if self.count == 0:
            self.count += 1
            # self.update_K()
            return self.delta_obs, self.dot_delta_obs
        else:
            self.count += 1
            ef_old = self.ef.copy()
            self.ef = (self.k * self.dt * syst_dynamic + ef_old + self.k * (dot_e_old - dot_e)) / (1 + self.k * self.dt)
            self.dot_ef = (self.ef - ef_old) / self.dt

            delta_obs_old = self.delta_obs.copy()
            self.delta_obs = -(self.dot_ef + self.k * self.ef) / self.k
            # self.delta_obs = -(self.dot_ef_obs + self.k * self.ef) / self.k
            self.dot_delta_obs = (self.delta_obs - delta_obs_old) / self.dt

            return self.delta_obs, self.dot_delta_obs
