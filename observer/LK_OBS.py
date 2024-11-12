import numpy as np
from typing import Union
import pandas as pd


class lk_obs:
    def __init__(self,
                 eta: np.ndarray = np.array([0.5, 0.5, 0.5]).astype(float),
                 beta2: np.ndarray = np.zeros(3).astype(float),
                 beta3: np.ndarray = np.zeros(3).astype(float),
                 Td: np.ndarray = np.array([5., 5., 5.]).astype(float),
                 w0: np.ndarray = np.array([1., 1., 1.]).astype(float),
                 dim: int = 3,
                 dt: float = 0.01
                 ):
        self.w0 = w0
        self.dim = dim
        self.dt = dt
        
        self.eta = eta
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta1 = np.sqrt(beta2 * beta3)
        
        self.Td = Td
        
        self.a1 = 0.5 * np.ones(self.dim)
        self.a2 = 1 / (2 ** ((1 + self.eta) / 2))
        self.a3 = 2 ** (self.eta - 2)
        
        self.b1 = 1 * np.ones(self.dim)
        self.b2 = 1 * np.ones(self.dim)
        self.b3 = ((3 - self.eta) / (2 - self.eta)) * (2 ** (self.eta - 2))
        
        self.xi = np.zeros(self.dim)  # position estimation error
        
        self.var_theta = np.zeros(self.dim)  #
        self.d_var_theta = np.zeros(self.dim)
        
        self.d_sigma = np.zeros(self.dim)
        self.sigma = 1 * np.ones(self.dim)
        
        # self.d_delta = np.zeros(self.dim)
        self.delta = np.zeros(self.dim)
        
        '''data record'''
        self.index = 0
        self.rec_t = np.empty(shape=[0, 1], dtype=float)
        self.rec_xi = np.empty(shape=[0, 3], dtype=float)
        self.rec_var_theta = np.empty(shape=[0, 3], dtype=float)
        self.rec_d_var_theta = np.empty(shape=[0, 3], dtype=float)
        self.rec_d_sigma = np.empty(shape=[0, 3], dtype=float)
        self.rec_sigma = np.empty(shape=[0, 3], dtype=float)
        self.rec_delta = np.empty(shape=[0, 3], dtype=float)
        '''data record'''
    
    @staticmethod
    def sig(x: Union[np.ndarray, list], a):
        return np.fabs(x) ** a * np.sign(x)
    
    def set_init(self, e0: np.ndarray, de0: np.ndarray, dy0: np.ndarray):
        self.xi = self.var_theta - e0
        self.var_theta = de0
        self.d_var_theta = dy0
    
    def observe(self, dy: Union[np.ndarray, list], de: Union[np.ndarray, list]):
        self.xi = self.var_theta - de
        
        self.d_var_theta = (-(np.pi / (self.Td * self.beta1 * (1 - self.eta))) *
                            (2 * self.a1 * self.beta1 * self.xi + self.a2 * self.beta2 * self.sig(self.xi, self.eta) + self.a3 * self.beta3 * self.sig(self.xi, 2 - self.eta))
                            - self.sigma * np.tanh(self.xi / self.w0)
                            + dy
                            )
        self.d_sigma = (- (2 * np.pi * self.b1 * self.beta1) / (self.Td * self.beta1 * (1 - self.eta)) * self.sigma
                        - (np.pi * self.b2 * self.beta2) / (self.Td * self.beta1 * (1 - self.eta)) * self.sigma
                        - (np.pi * self.b3 * self.beta3) / (self.Td * self.beta1 * (1 - self.eta)) * self.sig(self.sigma, 2 - self.eta)
                        + self.xi * np.tanh(self.xi / self.w0)
                        )
        self.delta = (-(np.pi / (self.Td * self.beta1 * (1 - self.eta))) *
                      (2 * self.a1 * self.beta1 * self.xi + self.a2 * self.beta2 * self.sig(self.xi, self.eta) + self.a3 * self.beta3 * self.sig(self.xi, 2 - self.eta))
                      - self.sigma * np.tanh(self.xi / self.w0)
                      )
        
        self.var_theta = self.var_theta + self.dt * self.d_var_theta
        self.sigma = self.sigma + self.dt * self.d_sigma
        
        '''datasave'''
        self.rec_t = np.append(self.rec_t, np.array([[self.index * self.dt]]), axis=0)
        self.rec_xi = np.append(self.rec_xi, [self.xi], axis=0)  # 添加整行元素，axis=1添加整列元素
        self.rec_d_var_theta = np.append(self.rec_d_var_theta, [self.d_var_theta], axis=0)
        self.rec_var_theta = np.append(self.rec_var_theta, [self.var_theta], axis=0)
        self.rec_d_sigma = np.append(self.rec_d_sigma, [self.d_sigma], axis=0)
        self.rec_sigma = np.append(self.rec_sigma, [self.sigma], axis=0)
        self.rec_delta = np.append(self.rec_delta, [self.delta], axis=0)
        self.index += 1
        '''datasave'''
        
        return self.delta

    def save_adap_obs_param(self, path: str, flag: str = 'pos'):
        pd.DataFrame(np.hstack((self.rec_t, self.rec_xi, self.rec_d_var_theta, self.rec_var_theta, self.rec_d_sigma, self.rec_sigma, self.rec_delta)),
                     columns=['time',
                              'xi_x', 'xi_y', 'xi_z',
                              'd_var_theta_x', 'd_var_theta_y', 'd_var_theta_z',
                              'var_theta_x', 'var_theta_y', 'var_theta_z',
                              'd_sigma_x', 'd_sigma_y', 'd_sigma_z',
                              'sigma_x', 'sigma_y', 'sigma_z',
                              'delta_x', 'delta_y', 'delta_z']). \
            to_csv(path + flag + '_adap_obs_param.csv', sep=',', index=False)
