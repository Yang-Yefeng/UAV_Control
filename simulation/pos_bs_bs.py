import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

from controller.BackStepping import backstepping
from observer.RobustDifferentatior_3rd import robust_differentiator_3rd as rd3
from uav.uav import UAV, uav_param
from utils.ref_cmd import *
from utils.utils import *
from utils.collector import data_collector

IS_IDEAL = True
observer_pool = ['rd3', 'none']
# neso: 非线性扩张状态观测器
# hsmo: 高阶滑模观测器
# afto: 固定时间观测器
# ro:   龙贝格 (勒贝格) 观测器
# rd3:  三阶鲁棒微分器
# none: 没观测器
OBSERVER_IN = observer_pool[1]
OBSERVER_OUT = observer_pool[1]

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
param.dt = 0.001
param.time_max = 20
'''Parameter list of the quadrotor'''


if __name__ == '__main__':
    uav = UAV(param)

    ctrl_out = backstepping(k_bs1=np.array([0.7, 0.7, 0.7]),  # gain for tracking e_eta
                            k_bs2=np.array([5., 5., 5.]),  # gain for tracking "de - virtual de_eta_d"
                            dim=3,
                            dt=uav.dt,
                            ctrl0=np.array([0., 0., uav.m * uav.g]))
    ctrl_in = backstepping(k_bs1=np.array([5., 5., 5.]),  # gain for tracking "e_rho"
                           k_bs2=np.array([40., 40., 40.]),  # gain for tracking "omega - virtual omega_d"
                           dim=3,
                           ctrl0=np.array([0, 0, 0]).astype(float),
                           dt=uav.dt)

    '''reference signal initialization'''
    ref_amplitude = np.array([2, 2, 1, np.pi / 2])  # x y z psi
    # ref_amplitude = np.array([0, 0, 0, 0])  # x y z psi
    ref_period = np.array([5, 5, 4, 5])
    ref_bias_a = np.array([2, 2, 1, 0])
    ref_bias_phase = np.array([np.pi / 2, 0, 0, 0])

    '''data storage initialization'''
    data_record = data_collector(N=int(uav.time_max / uav.dt))

    '''reference signal'''
    phi_d = theta_d = phi_d_old = theta_d_old = 0.
    dot_phi_d = (phi_d - phi_d_old) / uav.dt
    dot_theta_d = (theta_d - theta_d_old) / uav.dt
    throttle = uav.m * uav.g

    ref, dot_ref, _, _ = ref_uav(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)  # 整体参考信号 xd yd zd psid
    rhod = np.array([phi_d, theta_d, ref[3]]).astype(float)  # 内环参考信号 phi_d theta_d psi_d
    dot_rhod = np.array([dot_phi_d, dot_theta_d, dot_ref[3]]).astype(float)  # 内环参考信号导数

    '''initial error'''
    e_rho = uav.rho1() - rhod
    de_rho = uav.dot_rho1() - dot_rhod
    e_eta = uav.eta() - ref[0: 3]
    de_eta = uav.dot_eta() - dot_ref[0: 3]

    '''observer'''
    if OBSERVER_IN == 'rd3':
        observer = rd3(use_freq=True,
                       omega=np.array([3.5, 3.4, 3.9]),
                       dim=3,
                       dt=uav.dt)
        syst_dynamic0 = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(),
                                                                                                          ctrl_in.control_in))
        observer.set_init(e0=e_rho, de0=de_rho, syst_dynamic=syst_dynamic0)
    else:
        observer = np.zeros(3)

    if OBSERVER_OUT == 'rd3':
        obs_out = rd3(use_freq=True,
                      omega=np.array([0.9, 0.9, 0.9]),
                      dim=3,
                      dt=uav.dt)

        syst_dynamic_out = -uav.kt / uav.m * uav.dot_eta() + uav.A()
        obs_out.set_init(e0=uav.eta(), de0=uav.dot_eta(), syst_dynamic=syst_dynamic_out)
    else:
        obs_out = None

    '''control'''
    while uav.time < uav.time_max - uav.dt / 2:
        if uav.n % int(1 / param.dt) == 0:
            print('time: %.2f s.' % (uav.n / int(1 / param.dt)))

        '''1. generate reference command and uncertainty'''
        ref, dot_ref, dot2_ref, dot3_ref = ref_uav(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)  # 整体参考信号 xd yd zd psid
        uncertainty = generate_uncertainty(time=uav.time, is_ideal=IS_IDEAL)

        '''2. generate outer-loop reference signal 'eta_d' and its 1st derivatives'''
        eta_d = ref[0: 3]
        e_eta = uav.eta() - eta_d
        de_eta = uav.dot_eta() - dot_ref[0: 3]

        '''3. generate outer-loop virtual control command'''
        if OBSERVER_OUT == 'rd3':
            syst_dynamic = -uav.kt / uav.m * uav.dot_eta() + uav.A()
            _, _, obs_eta = obs_out.observe(e=uav.eta(), syst_dynamic=syst_dynamic)
        else:
            obs_eta = np.zeros(3)

        A_eta = -uav.kt / uav.m * uav.dot_eta() - dot2_ref[0:3]
        ctrl_out.control_update_outer(A_eta=A_eta, e=e_eta, dot_e=de_eta, obs=obs_eta)

        '''4. transfer virtual control command to actual throttle, phi_d, and theta_d'''
        phi_d_old = phi_d
        theta_d_old = theta_d
        phi_d, theta_d, throttle = uo_2_ref_angle_throttle(uo=ctrl_out.control_out, att=uav.uav_att(), m=uav.m, g=uav.g)

        dot_phi_d = (phi_d - phi_d_old) / uav.dt
        dot_theta_d = (theta_d - theta_d_old) / uav.dt
        rhod = np.array([phi_d, theta_d, ref[3]])  # phi_d theta_d psi_d
        dot_rhod = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])  # phi_d theta_d psi_d 的一阶导数

        '''5. inner-loop control'''
        e_rho = uav.rho1() - rhod
        de_rho = np.dot(uav.W(), uav.rho2()) - dot_rhod

        if OBSERVER_IN == 'rd3':
            syst_dynamic = np.dot(uav.dW(), uav.rho2()) + np.dot(uav.W(), uav.f2()) + np.dot(uav.W(), np.dot(uav.J_inv(), ctrl_in.control_in))
            _, _, obs_rho = observer.observe(syst_dynamic=syst_dynamic, e=e_rho)
        else:
            obs_rho = np.zeros(3)

        # 将观测器输出前两维度置为 0 即可
        obs_rho[0] = 0.
        obs_rho[1] = 0.
        # 将观测器输出前两维度置为 0 即可

        ctrl_in.control_update_inner(A_omega=uav.A_omega(),
                                     B_omega=uav.B_omega(),
                                     W=uav.W(),
                                     dot_ref=dot_rhod,
                                     e_rho=e_rho,
                                     omega=uav.rho2(),
                                     obs=obs_rho)
        action_4_uav = np.array([throttle, ctrl_in.control_in[0], ctrl_in.control_in[1], ctrl_in.control_in[2]])
        uav.rk44(action=action_4_uav, dis=uncertainty, n=1, att_only=False)

        '''6. '''
        data_block = {'time': uav.time,
                      'control': action_4_uav,
                      'ref_angle': rhod,
                      'ref_pos': ref[0: 3],
                      'ref_vel': dot_ref[0: 3],
                      'd_in': np.array([0., 0., np.dot(uav.W(), np.array([uncertainty[3], uncertainty[4], uncertainty[5]]))[2] - dot2_ref[3]]),
                      'd_in_obs': obs_rho,
                      'd_out': np.array([uncertainty[0], uncertainty[1], uncertainty[2]]) / uav.m,
                      'd_out_obs': obs_eta,
                      'state': uav.uav_state_call_back()}
        data_record.record(data=data_block)

    data_record.plot_att()
    data_record.plot_vel()
    data_record.plot_pos()
    data_record.plot_torque()
    data_record.plot_throttle()
    data_record.plot_outer_obs()
    # data_record.plot_inner_obs()
    plt.show()
