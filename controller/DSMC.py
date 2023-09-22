import os

import numpy as np
import pandas as pd


class dsmc:
    def __init__(self, ctrl0: np.ndarray, dt: float):
        self.c = np.array([5, 5, 10]).astype(float)
        self.lmd = np.array([15, 8, 10]).astype(float)
        self.k0 = np.array([40, 40, 25]).astype(float)
        self.s = np.array([0, 0, 0]).astype(float)
        self.ds = np.array([0, 0, 0]).astype(float)
        self.sigma = np.array([0, 0, 0]).astype(float)
        self.control = ctrl0
        self.du = np.array([0, 0, 0]).astype(float)
        self.dt = dt

    def dot_control(self,
                    e: np.ndarray, de: np.ndarray, dde: np.ndarray,
                    f1: np.ndarray, f2: np.ndarray, rho2: np.ndarray, h: np.ndarray,
                    F: np.ndarray, dot_Frho2_f1f2: np.ndarray, dot_f1h: np.ndarray,
                    delta_obs: np.ndarray):
        syst_dynamic1 = np.dot(F, rho2) + np.dot(f1, f2)
        syst_dynamic2 = dot_Frho2_f1f2

        self.s = self.c * e + de
        self.ds = self.c * de + dde
        self.sigma = self.ds + self.lmd * self.s

        du1 = self.lmd * self.c * de + (self.c + self.lmd) * syst_dynamic1 + syst_dynamic2
        du2 = np.dot((self.c + self.lmd) * np.dot(f1, h) + dot_f1h, self.control)
        V_dis = (self.c + self.lmd) * delta_obs  # + dot_delta_obs
        du3 = (np.fabs(V_dis) + self.k0) * np.tanh(5 * self.sigma)
        self.du = -np.dot(np.linalg.inv(np.dot(f1, h)), du1 + du2 + du3)

    def control_update(self):
        self.control += self.du * self.dt