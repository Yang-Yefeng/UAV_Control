import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from observer.RobustDifferentatior_3rd import robust_differentiator_3rd as rd3
from controller.BackStepping import backstepping
from uav.uav import UAV, uav_param
from utils.ref_cmd import *
from utils.collector import data_collector

IS_IDEAL = False
observer_pool = ['rd3', 'none']
# neso: 非线性扩张状态观测器
# hsmo: 高阶滑模观测器
# afto: 固定时间观测器
# ro:   龙贝格 (勒贝格) 观测器
# rd3:  三阶鲁棒微分器
# none: 没观测器
OBSERVER = observer_pool[0]

'''Parameter list of the quadrotor'''
param = uav_param()
param.m = 0.8
param.g = 9.8
param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
param.d = 0.12
param.CT = 2.168e-6
param.CM = 2.136e-8
param.J0 = 1.01e-5
param.kr = 1e-3
param.kt = 1e-3
param.pos0 = np.array([0, 0, 0])
param.vel0 = np.array([0, 0, 0])
param.angle0 = np.array([0, 0, 0])
param.pqr0 = np.array([0, 0, 0])
param.dt = 1e-3
param.time_max = 20
'''Parameter list of the quadrotor'''

if __name__ == '__main__':
    uav = UAV(param)
    ctrl_in = backstepping(k_bs1=np.array([5., 5., 5.]),     # gain for tracking "e_rho"
                           k_bs2=np.array([40., 40., 40.]),        # gain for tracking "omega - virtual omega_d"
                           dim=3,
                           ctrl0=np.array([0, 0, 0]).astype(float),
                           dt=uav.dt)

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
        syst_dynamic0 = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(),
                                                                                                          ctrl_in.control_in))
        observer.set_init(e0=e0, de0=de0, syst_dynamic=syst_dynamic0)
    else:
        observer = np.zeros(3)

    de = np.array([0, 0, 0]).astype(float)
    data_record = data_collector(N=int(uav.time_max / uav.dt))

    while uav.time < uav.time_max:
        if uav.n % int(1 / param.dt) == 0:
            print('time: %.2f s.' % (uav.n / int(1 / param.dt)))
        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''
        uncertainty = generate_uncertainty(time=uav.time, is_ideal=IS_IDEAL)
        rhod, dot_rhod, dot2_rhod, dot3_rhod = ref_inner(uav.time, ref_amplitude, ref_period, ref_bias_a,
                                                         ref_bias_phase)
        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''

        '''2. 计算 tk 时刻误差信号'''
        e = uav.rho1() - rhod
        de = uav.dot_rho1() - dot_rhod  # 这个时候 de 是新时刻的
        '''2. 计算 tk 时刻误差信号'''

        '''3. 观测器'''
        if OBSERVER == 'rd3':
            syst_dynamic = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control_in))
            delta_obs, dot_delta_obs = observer.observe(syst_dynamic=syst_dynamic, e=e)
        else:
            delta_obs, dot_delta_obs = np.zeros(3), np.zeros(3)
        '''3. 观测器'''

        e_rho = uav.rho1() - rhod                           # e1
        de_rho = np.dot(uav.W(), uav.rho2()) - dot_rhod     # de1

        action_4_uav = np.array([uav.m * uav.g, ctrl_in.control_in[0], ctrl_in.control_in[1], ctrl_in.control_in[2]])
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

        uav.rk44(action=action_4_uav, dis=uncertainty, n=1, att_only=True)
        ctrl_in.control_update_inner(A_omega=uav.A_omega(),
                                     B_omega=uav.B_omega(),
                                     W=uav.W(),
                                     dot_ref=dot_rhod,
                                     e_rho=e_rho,
                                     omega=uav.rho2(),
                                     obs=delta_obs)

    SAVE = False
    if SAVE:
        new_path = '../datasave/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '/'
        os.mkdir(new_path)
        data_record.package2file(new_path)

    data_record.plot_att()
    data_record.plot_torque()
    data_record.plot_inner_obs()
    plt.show()
