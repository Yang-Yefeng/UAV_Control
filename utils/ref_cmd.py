import numpy as np


def ref_inner(time, amplitude: np.ndarray, period: np.ndarray, bias_a: np.ndarray, bias_phase: np.ndarray):
    """
    :param time:        time
    :param amplitude:   amplitude
    :param period:      period
    :param bias_a:      amplitude bias
    :param bias_phase:  phase bias
    :return:            reference attitude angles and their 1st - 3rd derivatives
                        [phi_d, theta_d, psi_d]
                        [dot_phi_d, dot_theta_d, dot_psi_d]
                        [dot2_phi_d, dot2_theta_d, dot2_psi_d]
                        [dot3_phi_d, dot3_theta_d, dot3_psi_d]
    """
    w = 2 * np.pi / period
    _r = amplitude * np.sin(w * time + bias_phase) + bias_a
    _dr = amplitude * w * np.cos(w * time + bias_phase)
    _ddr = -amplitude * w ** 2 * np.sin(w * time + bias_phase)
    _dddr = -amplitude * w ** 3 * np.cos(w * time + bias_phase)
    return _r, _dr, _ddr, _dddr


def ref_uav(time: float, amplitude: np.ndarray, period: np.ndarray, bias_a: np.ndarray, bias_phase: np.ndarray):
    """
    :param time:        time
    :param amplitude:   amplitude
    :param period:      period
    :param bias_a:      amplitude bias
    :param bias_phase:  phase bias
    :return:            reference position and yaw angle and their 1st - 3rd derivatives
                        [x_d, y_d, z_d, yaw_d]
                        [dot_x_d, dot_y_d, dot_z_d, dot_yaw_d]
                        [dot2_x_d, dot2_y_d, dot2_z_d, dot2_yaw_d]
                        [dot3_x_d, dot3_y_d, dot3_z_d, dot3_yaw_d]
    """
    # _r = np.zeros_like(amplitude)
    # _dr = np.zeros_like(amplitude)
    # _ddr = np.zeros_like(amplitude)
    # _dddr = np.zeros_like(amplitude)
    # if time <= 5.0:
    w = 2 * np.pi / period
    _r = amplitude * np.sin(w * time + bias_phase) + bias_a
    _dr = amplitude * w * np.cos(w * time + bias_phase)
    _ddr = -amplitude * w ** 2 * np.sin(w * time + bias_phase)
    _dddr = -amplitude * w ** 3 * np.cos(w * time + bias_phase)
    # elif 5.0 < time <= 10.0:
    #     pass
    # elif 10.0<time<=15:
    #     pass
    # else:
    #     _r=np.ones_like(amplitude)
    return _r, _dr, _ddr, _dddr


def generate_uncertainty(time: float, is_ideal: bool = False) -> np.ndarray:
    """
    :param time:        time
    :param is_ideal:    ideal or not
    :return:            Fdx, Fdy, Fdz, dp, dq, dr
    """
    if is_ideal:
        return np.array([0, 0, 0, 0, 0, 0]).astype(float)
    else:
        T = 2
        w = 2 * np.pi / T
        phi0 = 0.
        if time <= 5:
            phi0 = 0.
            Fdx = 0.5 * np.sin(w * time + phi0) + 0.2 * np.cos(3 * w * time + phi0) + 0.2
            Fdy = 0.5 * np.cos(w * time + phi0) + 0.2 * np.sin(3 * w * time + phi0) + 0.4
            Fdz = 0.5 * np.sin(w * time + phi0) + 0.2 * np.cos(3 * w * time + phi0) - 0.5
            dp = 0.5 * np.sin(w * time + phi0) + 0.2 * np.cos(w * time + phi0)
            dq = 0.5 * np.cos(w * time + phi0) + 0.2 * np.sin(w * time + phi0)
            dr = 0.5 * np.cos(w * time + phi0) + 0.2 * np.sin(w * time + phi0)
        elif 5 < time <= 10:
            Fdx = 1.5
            Fdy = 0.4 * (time - 5.0)
            Fdz = -0.6
            dp = 0.5 * np.sin(w * time + phi0) + 0.2 * np.cos(w * time + phi0)
            dq = 0.5 * np.cos(w * time + phi0) + 0.2 * np.sin(w * time + phi0)
            dr = 0.5
        else:
            phi0 = np.pi / 2
            Fdx = 0.5 * np.sin(w * time + phi0) - 1.0 * np.cos(3 * w + time + phi0)
            Fdy = 0.5 * np.cos(w * time + phi0) - 1.0 * np.sin(3 * w + time + phi0) + 1.0
            Fdz = 0.5 * np.sin(w * time + phi0) - 1.0 * np.cos(3 * w + time + phi0) - 0.4
            dp = 0.5 * np.sin(w * time + phi0) + 0.2 * np.cos(w * time + phi0)
            dq = 0.5 * np.cos(w * time + phi0) + 0.2 * np.sin(w * time + phi0)
            dr = 0.5 * np.cos(w * time + phi0) + 0.2 * np.sin(w * time + phi0)

        # Fdx = 0.5 * np.sin(0.8 * np.pi * time) + 0.2 * np.cos(0.4 * np.pi * time)
        # Fdy = 0.5 * np.cos(0.8 * np.pi * time) + 0.2 * np.sin(0.4 * np.pi * time)
        # Fdz = 0.5 * np.sin(0.8 * np.pi * time) + 0.2 * np.cos(0.4 * np.pi * time)
        # dp = 0.5 * np.sin(0.4 * np.pi * time) + 0.2 * np.cos(0.8 * np.pi * time)
        # dq = 0.5 * np.cos(0.4 * np.pi * time) + 0.2 * np.sin(0.8 * np.pi * time)
        # dr = 0.5 * np.cos(0.4 * np.pi * time) + 0.2 * np.sin(0.8 * np.pi * time)
        return np.array([Fdx, Fdy, Fdz, dp, dq, dr])
