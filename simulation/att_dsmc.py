import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from observer.NESO import neso
from observer.HSMO import hsmo
from observer.AFTO import afto
from observer.RO import ro
from observer.RobustDifferentatior_3rd import robust_differentiator_3rd as rd3
from controller.DSMC import dsmc
from uav.uav import UAV, uav_param
from utils.ref_cmd import *
from utils.collector import data_collector

IS_IDEAL = False
observer_pool = ['neso', 'hsmo', 'afto', 'ro', 'rd3', 'none']
# neso: 非线性扩张状态观测器
# hsmo: 高阶滑模观测器
# afto: 固定时间观测器
# ro:   龙贝格 (勒贝格) 观测器
# rd3:  三阶鲁棒微分器
# none: 没观测器
OBSERVER = observer_pool[4]

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
    ctrl_in = dsmc(ctrl0=np.array([0, 0, 0]).astype(float), dt=uav.dt)

    ref_amplitude = np.array([np.pi / 3, np.pi / 3, np.pi / 2])
    ref_period = np.array([5, 5, 4])
    ref_bias_a = np.array([0, 0, 0])
    ref_bias_phase = np.array([0, np.pi / 2, 0])

    rhod, dot_rhod, dot2_rhod, dot3_rhod = ref_inner(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
    e0 = uav.rho1() - rhod
    de0 = uav.dot_rho1() - dot_rhod

    if OBSERVER == 'neso':
        observer = neso(l1=np.array([2., 2., 2.]),
                        l2=np.array([2., 2., 2.]),
                        l3=np.array([1., 1., 1.]),
                        r=np.array([20., 20., 20.]),  # 50
                        k1=np.array([0.5, 0.5, 0.5]),
                        k2=np.array([0.001, 0.001, 0.001]),
                        dim=3,
                        dt=uav.dt)
        syst_dynamic0 = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control))
        observer.set_init(x0=e0, dx0=de0, syst_dynamic=syst_dynamic0)
    elif OBSERVER == 'hsmo':
        observer = hsmo(alpha1=np.array([10, 3, 3]),
                        alpha2=np.array([2.0, 1.5, 1.5]),
                        alpha3=np.array([3.0, 1.1, 1.1]),
                        p=np.array([20, 20, 10]),
                        dim=3,
                        dt=uav.dt)
        observer.set_init(de0=de0)
    elif OBSERVER == 'afto':
        observer = afto(K=np.array([5., 5., 5.]),
                        alpha=np.array([1, 1, 1]),
                        beta=np.array([2, 2, 2]),
                        p=np.array([0.5, 0.5, 0.5]),
                        q=np.array([2, 2, 2]),
                        dt=uav.dt,
                        dim=3,
                        init_de=de0)
    elif OBSERVER == 'ro':
        observer = ro(k1=np.array([15., 15., 15.]),
                      k2=np.array([5., 5., 5.]),
                      dim=3,
                      dt=uav.dt)
        observer.set_init(de=de0)
    elif OBSERVER == 'rd3':
        '''
            m 和 n 可以相等，也可以不同。m对应低次，n对应高次。
        '''
        observer = rd3(m1=11 * np.ones(3),
                       m2=35 * np.ones(3),
                       m3=25 * np.ones(3),
                       n1=3 * np.ones(3),
                       n2=3 * np.ones(3),
                       n3=3 * np.ones(3),
                       dim=3,
                       dt=uav.dt)
        syst_dynamic0 = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control))
        observer.set_init(e0=e0, de0=de0, syst_dynamic=syst_dynamic0)
    else:
        observer = None

    de = np.array([0, 0, 0]).astype(float)
    data_record = data_collector(N=int(uav.time_max / uav.dt))

    while uav.time < uav.time_max:
        if uav.n % int(1 / param.dt) == 0:
            print('time: %.2f s.' % (uav.n / int(1 / param.dt)))
        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''
        uncertainty = generate_uncertainty(time=uav.time, is_ideal=False)
        rhod, dot_rhod, dot2_rhod, dot3_rhod = ref_inner(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''

        '''2. 计算 tk 时刻误差信号'''
        if uav.n == 0:
            e = uav.rho1() - rhod
            de = uav.dot_rho1() - dot_rhod  # 这个时候 de 是新时刻的
            de_old = de.copy()  # 这个时候 de 是上一时刻的
        else:
            de_old = de.copy()  # 这个时候 de 是上一时刻的
            e = uav.rho1() - rhod
            de = uav.dot_rho1() - dot_rhod  # 这个时候 de 是新时刻的

        if OBSERVER == 'neso':
            syst_dynamic = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control))
            delta_obs, dot_delta_obs = observer.observe(x=e, syst_dynamic=syst_dynamic)
        elif OBSERVER == 'hsmo':
            syst_dynamic = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control))
            delta_obs, dot_delta_obs = observer.observe(syst_dynamic=syst_dynamic, de=de)
        elif OBSERVER == 'afto':
            syst_dynamic = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control))
            delta_obs, dot_delta_obs = observer.observe(syst_dynamic=syst_dynamic,
                                                        dot_e_old=de_old,
                                                        dot_e=de)
        elif OBSERVER == 'ro':
            syst_dynamic = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control))
            delta_obs, dot_delta_obs = observer.observe(syst_dynamic=syst_dynamic, de=de)
        elif OBSERVER == 'rd3':
            syst_dynamic = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control))
            delta_obs, dot_delta_obs = observer.observe(syst_dynamic=syst_dynamic, e=e)
        else:
            delta_obs, dot_delta_obs = np.zeros(3), np.zeros(3)

        if IS_IDEAL:
            dde = np.dot(uav.dW(), uav.rho2()) + \
                  np.dot(uav.W(), uav.f2() + np.dot(uav.J_inv(), ctrl_in.control))  # - dot2_rhod
        else:
            dde = np.dot(uav.dW(), uav.rho2()) + \
                  np.dot(uav.W(), uav.f2() + np.dot(uav.J_inv(), ctrl_in.control)) + \
                  delta_obs
        '''2. 计算 tk 时刻误差信号'''

        ctrl_in.dot_control(e, de, dde,
                            uav.W(), uav.f2(), uav.rho2(), uav.J_inv(), uav.dW(), uav.dot_Frho2_f1f2(), uav.dot_f1h(),
                            delta_obs)
        action_4_uav = np.array([uav.m * uav.g, ctrl_in.control[0], ctrl_in.control[1], ctrl_in.control[2]])
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
        ctrl_in.control_update()

    SAVE = False
    if SAVE:
        new_path = '../datasave/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '/'
        os.mkdir(new_path)
        data_record.package2file(new_path)

    data_record.plot_att()
    data_record.plot_torque()
    data_record.plot_inner_obs()
    plt.show()
