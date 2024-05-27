import os
import sys
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from observer.RobustDifferentatior_3rd import robust_differentiator_3rd as rd3
from controller.FNTSMC import fntsmc, fntsmc_param
from uav.uav import UAV, uav_param
from utils.ref_cmd import *
from utils.collector import data_collector


'''Parameter list of the quadrotor'''
DT = 0.001
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 20
'''Parameter list of the quadrotor'''


'''Parameter list of the attitude controller'''
att_ctrl_param = fntsmc_param()
att_ctrl_param.k1 = np.array([25, 25, 40])
att_ctrl_param.k2 = np.array([0.1, 0.1, 0.2])
att_ctrl_param.k3 = np.array([0.05, 0.05, 0.05])
att_ctrl_param.alpha = np.array([2.5, 2.5, 2.5])
att_ctrl_param.beta = np.array([0.99, 0.99, 0.99])
att_ctrl_param.gamma = np.array([1.5, 1.5, 1.2])
att_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
att_ctrl_param.dim = 3
att_ctrl_param.dt = DT
att_ctrl_param.ctrl0 = np.array([0., 0., 0.])
att_ctrl_param.saturation = np.array([0.3, 0.3, 0.3])
'''Parameter list of the attitude controller'''


IS_IDEAL = False
OBSERVER = 'rd3'


if __name__ == '__main__':
    uav = UAV(uav_param)
    ctrl_in = fntsmc(att_ctrl_param)

    ref_amplitude = np.array([np.pi / 3, np.pi / 3, np.pi / 2])
    ref_period = np.array([5, 5, 4])
    ref_bias_a = np.array([0, 0, 0])
    ref_bias_phase = np.array([0, np.pi / 2, 0])

    rhod, dot_rhod, dot2_rhod, dot3_rhod = ref_inner(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
    e0 = uav.rho1() - rhod
    de0 = uav.dot_rho1() - dot_rhod

    if OBSERVER == 'rd3':
        '''
            m 和 n 可以相等，也可以不同。m对应低次，n对应高次。
        '''
        observer = rd3(use_freq=True,
                       omega=np.array([3.5, 3.4, 3.9]),
                       dim=3,
                       dt=uav.dt)
        syst_dynamic0 = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), np.dot(uav.J_inv(), uav.f_rho() + ctrl_in.control))
        observer.set_init(e0=e0, de0=de0, syst_dynamic=syst_dynamic0)
    else:
        observer = None

    de = np.array([0, 0, 0]).astype(float)
    data_record = data_collector(N=int(uav.time_max / uav.dt))

    while uav.time < uav.time_max:
        if uav.n % int(1 / uav.dt) == 0:
            print('time: %.2f s.' % (uav.n / int(1 / uav.dt)))

        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''
        uncertainty = generate_uncertainty(time=uav.time, is_ideal=IS_IDEAL)
        rhod, dot_rhod, dot2_rhod, dot3_rhod = ref_inner(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''

        '''2. 计算 tk 时刻误差信号'''
        e = uav.rho1() - rhod
        de = uav.dot_rho1() - dot_rhod  # 这个时候 de 是新时刻的
        '''2. 计算 tk 时刻误差信号'''

        '''3. 观测器'''
        if OBSERVER == 'rd3':
            syst_dynamic = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control))
            _, _, delta_obs = observer.observe(syst_dynamic=syst_dynamic, e=e)
        else:
            delta_obs = np.zeros(3)
        '''3. 观测器'''

        '''4. 计算控制量'''
        ctrl_in.control_update2(second_order_att_dynamics=uav.A_rho(),
                                control_mat=uav.B_rho(),
                                e=e,
                                de=de,
                                dd_ref=dot2_rhod,
                                obs=delta_obs)
        '''4. 计算控制量'''

        '''5. 状态更新'''
        action_4_uav = np.array([uav.m * uav.g, ctrl_in.control[0], ctrl_in.control[1], ctrl_in.control[2]])
        uav.rk44(action=action_4_uav, dis=uncertainty, n=1, att_only=True)
        '''5. 状态更新'''

        '''6. 数据存储'''
        data_block = {'time': uav.time,
                      'control': action_4_uav,
                      'ref_angle': rhod,
                      'ref_pos': np.array([0., 0., 0.]),
                      'ref_vel': np.array([0., 0., 0.]),
                      'd_in': np.dot(uav.W(), np.array([uncertainty[3], uncertainty[4], uncertainty[5]])) - dot2_rhod,
                      'd_in_obs': delta_obs,
                      'd_out': np.array([uncertainty[0], uncertainty[1], uncertainty[2]]) / uav.m,
                      'd_out_obs': np.array([0., 0., 0.]),
                      'state': np.hstack((np.zeros(6), uav.uav_att_pqr_call_back()))}
        data_record.record(data=data_block)
        '''6. 数据存储'''

    data_record.plot_att()
    data_record.plot_torque()
    data_record.plot_inner_obs()
    plt.show()
