import numpy as np


class backstepping:
    def __init__(self,
                 k_bs1: np.ndarray = np.array([1., 1., 1.]),
                 k_bs2: np.ndarray = np.array([1., 1., 1.]),
                 dim: int = 3,
                 dt: float = 0.001,
                 ctrl0: np.ndarray = np.array([0., 0., 0.])):
        self.k_bs1 = k_bs1
        self.k_bs2 = k_bs2
        self.dt = dt
        self.dim = dim
        self.control_in = ctrl0
        self.control_out = ctrl0

    def Gamma_omega(self, W: np.ndarray, dot_ref: np.ndarray, e_rho: np.ndarray):
        return np.dot(np.linalg.inv(W), dot_ref - self.k_bs1 * e_rho)

    def Gamma_eta(self, e: np.ndarray):
        return -self.k_bs1 * e

    def control_update_inner(self,
                             A_omega: np.ndarray,
                             B_omega: np.ndarray,
                             W: np.ndarray,
                             dot_ref: np.ndarray,
                             e_rho: np.ndarray,
                             omega: np.ndarray,
                             obs: np.ndarray):
        """
        :param obs:
        :param A_omega:     omega 系统动态矩阵
        :param B_omega:     omega 控制矩阵
        :param W:           无人机建模中的定义
        :param dot_ref:     参考轨迹一阶导数
        :param e_rho:     角度控制误差
        :param omega:       角速度 p q r
        :return:
        """
        Gamma = self.Gamma_omega(W, dot_ref, e_rho)
        tau_1 = -A_omega - obs
        tau_2 = -np.dot(W.T, e_rho)
        tau_3 = -self.k_bs2 * (omega - Gamma)
        self.control_in = np.dot(np.linalg.inv(B_omega), tau_1 + tau_2 + tau_3)

    def control_update_outer(self, A_eta: np.ndarray, e: np.ndarray, dot_e: np.ndarray, obs: np.ndarray):
        """
        :param A_eta:
        :param e:
        :param dot_e:
        :param obs:
        :return:
        """
        u1 = -A_eta - obs
        u2 = -e
        u3 = -self.k_bs2 * (dot_e - self.Gamma_eta(e))
        self.control_out = u1 + u2 + u3
