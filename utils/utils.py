import numpy as np


def deg2rad(deg: float) -> float:
    """
    :brief:         omit
    :param deg:     degree
    :return:        radian
    """
    return deg * np.pi / 180.0


def rad2deg(rad: float) -> float:
    """
    :brief:         omit
    :param rad:     radian
    :return:        degree
    """
    return rad * 180.8 / np.pi


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def uo_2_ref_angle_throttle(uo: np.ndarray, att: np.ndarray, m: float, g: float):
    # print('fuck', uf)
    ux = uo[0]
    uy = uo[1]
    uz = uo[2]
    uf = (uz + g) * m / (C(att[0]) * C(att[1]))
    asin_phi_d = min(max((ux * np.sin(att[2]) - uy * np.cos(att[2])) * m / uf, -1), 1)
    phi_d = np.arcsin(asin_phi_d)
    asin_theta_d = min(max((ux * np.cos(att[2]) + uy * np.sin(att[2])) * m / (uf * np.cos(phi_d)), -1), 1)
    theta_d = np.arcsin(asin_theta_d)
    # print(phi_d * 180 / np.pi, theta_d * 180 / np.pi)
    return phi_d, theta_d, uf
