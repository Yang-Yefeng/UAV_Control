import numpy as np



class backstepping:
    def __init__(self,
                 k_bc: np.ndarray = np.array([1., 1., 1.]),
                 dim: int = 3,
                 dt: float = 0.001,
                 ctrl0: np.ndarray = np.array([0., 0., 0.])):
        self.k_bc = k_bc
        self.dt = dt
        self.dim = dim
        self.control = ctrl0

    def control_update(self, e1:np.ndarray, de1: np.ndarray, f: np.ndarray, g: np.ndarray, dd_ref: np.ndarray, obs: np.ndarray):
        """
        :param de1:
        :param e2:
        :param f:       system dynamics
        :param g:       control matrix
        :param dd_ref:  dot dot reference
        :param obs:     observer
        :return:        control output
        """
        if dd_ref is not None:
            tau_1 = f + self.k_bc * de1 - dd_ref - obs
        else:
            tau_1 = f + self.k_bc * de1 - obs
        e_bs = de1 + self.k_bc *e1
        tau_2 = -self.k_bc * e_bs
        self.control = -np.dot(np.linalg.inv(g), tau_1 + tau_2)
