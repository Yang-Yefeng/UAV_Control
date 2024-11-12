import numpy as np


class fntsmc2_param:
	def __init__(self):
		self.k1: np.ndarray = np.array([1.2, 0.8, 1.5])
		self.k2: np.ndarray = np.array([0.2, 0.6, 1.5])
		self.k3: np.ndarray = np.array([0.05, 0.05, 0.05])
		self.k4: np.ndarray = np.array([0.05, 0.05, 0.05])
		self.p1: np.ndarray = 7 * np.ones(3)
		self.p2: np.ndarray = 5 * np.ones(3)
		self.p3: np.ndarray = 9 * np.ones(3)
		self.p4: np.ndarray = 7 * np.ones(3)
		self.p5: np.ndarray = 7 * np.ones(3)
		self.p6: np.ndarray = 5 * np.ones(3)
		self.dim: int = 3
		self.dt: float = 0.01


class fntsmc2:
	def __init__(self,
				 param: fntsmc2_param = None,
				 k1: np.ndarray = np.array([1.2, 0.8, 1.5]),
				 k2: np.ndarray = np.array([0.2, 0.6, 1.5]),
				 k3: np.ndarray = np.array([0.05, 0.05, 0.05]),
				 k4: np.ndarray = np.array([0.05, 0.05, 0.05]),
				p1: np.ndarray = 7 * np.ones(3),
				p2: np.ndarray = 5 * np.ones(3),
				p3: np.ndarray = 9 * np.ones(3),
				p4: np.ndarray = 7 * np.ones(3),
				p5: np.ndarray = 7 * np.ones(3),
				p6: np.ndarray = 5 * np.ones(3),
				 dim: int = 3,
				 dt: float = 0.01):
		self.k1 = k1 if param is None else param.k1
		self.k2 = k2 if param is None else param.k2
		self.k3 = k3 if param is None else param.k3
		self.k4 = k4 if param is None else param.k4
		self.p1 = p1 if param is None else param.p1
		self.p2 = p2 if param is None else param.p2
		self.p3 = p3 if param is None else param.p3
		self.p4 = p4 if param is None else param.p4
		self.p5 = p5 if param is None else param.p5
		self.p6 = p6 if param is None else param.p6
		self.dt = dt if param is None else param.dt
		self.dim = dim if param is None else param.dim
		self.s = np.zeros(self.dim)
		self.control_in = np.zeros(self.dim)
		self.control_out = np.zeros(self.dim)

	@staticmethod
	def sig(x, a, kt=5):
		return np.fabs(x) ** a * np.tanh(kt * x)

	def control_update_inner(self,
							 e_rho: np.ndarray,
							 dot_e_rho: np.ndarray,
							 dd_ref: np.ndarray,
							 A_rho: np.ndarray,
							 B_rho: np.ndarray,
							 obs: np.ndarray,
							 att_only: bool = False):
		if not att_only:
			dd_ref = np.zeros(self.dim)
		self.s = e_rho + self.k1 * self.sig(e_rho, self.p1 / self.p2) + self.k2 * self.sig(dot_e_rho, self.p3 / self.p4)
		tau1 = A_rho + dd_ref
		tau2 = self.p4 / (self.k2 * self.p3) * self.sig(dot_e_rho, 2 - self.p3 / self.p4)
		tau3 = self.k1 * self.p1 * self.p4 / (self.k2 * self.p2 * self.p3) * self.sig(dot_e_rho, 2 - self.p3 / self.p4) * self.sig(e_rho, self.p1 / self.p2 - 1)
		tau4 = obs + self.k3 * np.tanh(5 * self.s) + self.k4 * self.sig(self.s, self.p5 / self.p6)
		self.control_in = -np.dot(np.linalg.inv(B_rho), tau1 + tau2 + tau3 + tau4)

	def control_update_outer(self,
							 e_eta: np.ndarray,
							 dot_e_eta: np.ndarray,
							 dot_eta: np.ndarray,
							 kt: float,
							 m: float,
							 dd_ref: np.ndarray,
							 obs: np.ndarray):
		self.s = dot_e_eta + self.k1 * self.sig(e_eta, self.p1 / self.p2) + self.k2 * self.sig(dot_e_eta, self.p3 / self.p4)
		u1 = kt / m * dot_eta + dd_ref
		u2 = self.p4 / (self.k2 * self.p3) * (self.sig(dot_e_eta, 2 - self.p3 / self.p4) + self.k1 * self.p1 / self.p2 * self.sig(e_eta, self.p1 / self.p2 - 1))
		u3 = self.k3 * np.tanh(5 * self.s) + self.k4 * self.sig(self.s, self.p5 / self.p6) + obs
		self.control_out = -(u1 + u2 + u3)
