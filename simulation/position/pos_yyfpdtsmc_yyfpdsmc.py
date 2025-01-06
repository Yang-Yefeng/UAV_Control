import os
import sys
import datetime
import matplotlib.pyplot as plt
import platform
import pandas as pd

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from observer.YYF_pdt_obs import yyf_pdt_obs
from controller.PDT_yyf_SMC import pdt_yyf_smc_param, pdt_yyf_smc
from uav.uav import UAV, uav_param
from utils.ref_cmd import *
from utils.utils import *
from utils.collector import data_collector

'''Parameter list of the quadrotor'''
DT = 0.01
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
att_ctrl_param = pdt_yyf_smc_param(
    a_s=np.array([5./6., 5./6., 5./6.]).astype(float),  # 滑模里面的 alpha
# a_s=np.array([0.5, 0.5, 0.5]).astype(float),  # 滑模里面的 alpha
    b1_s=np.array([1., 1., 1.]).astype(float),  # 滑模里面的 beta1
    b2_s=np.array([1., 1., 1.]).astype(float),  # 滑模里面的 beta2
    b3_s=np.array([1., 1., 1.]).astype(float),  # 滑模里面的 beta3
    Ts=np.array([1., 1., 1.]).astype(float),  # 滑模里面的预设时间
    k1_s=np.array([1., 1., 1.]).astype(float),  # 滑模里面的 kappa1
    
    a_c=np.array([5./6., 5./6., 5./6.]).astype(float),  # 控制器里面的 alpha
    # a_c=np.array([0.5, 0.5, 0.5]).astype(float),  # 控制器里面的 alpha
    b1_c=np.array([1., 1., 1.]).astype(float),  # 控制器里面的 beta1
    b2_c=np.array([1., 1., 1.]).astype(float),  # 控制器里面的 beta2
    b3_c=np.array([1., 1., 1.]).astype(float),  # 控制器里面的 beta3
    Tc=np.array([5., 5., 5.]).astype(float),  # 控制器里面的预设时间
    k1_c=np.array([1., 1., 1.]).astype(float),  # 控制器里面的 kappa1
    k2=np.array([2, 2, 2]).astype(float),  # 补偿观测器
    dim=3,
    dt=DT
)
'''Parameter list of the attitude controller'''

'''Parameter list of the position controller'''
pos_ctrl_param = pdt_yyf_smc_param(
    a_s=np.array([11./13., 11./13., 11./13.]).astype(float),  # 滑模里面的 alpha
    b1_s=np.array([1., 1., 1.]).astype(float),  # 滑模里面的 beta1
    b2_s=np.array([1., 1., 1.]).astype(float),  # 滑模里面的 beta2
    b3_s=np.array([1., 1., 1.]).astype(float),  # 滑模里面的 beta3
    Ts=np.array([5., 5., 5.]).astype(float),  # 滑模里面的预设时间
    k1_s=np.array([1., 1., 1.]).astype(float),  # 滑模里面的 kappa1
    a_c=np.array([11./13., 11./13., 11./13.]).astype(float),  # 控制器里面的 alpha
    b1_c=np.array([1., 1., 1.]).astype(float),  # 控制器里面的 beta1
    b2_c=np.array([1., 1., 1.]).astype(float),  # 控制器里面的 beta2
    b3_c=np.array([1., 1., 1.]).astype(float),  # 控制器里面的 beta3
    Tc=np.array([10, 10, 10]).astype(float),  # 控制器里面的预设时间
    k1_c=np.array([1., 1., 1.]).astype(float),  # 控制器里面的 kappa1
    k2=np.array([2, 2, 2]).astype(float),
    dim=3,
    dt=DT
)
'''Parameter list of the position controller'''

IS_IDEAL = False
USE_OBS_IN = False
USE_OBS_OUT = True
SAVE = False
cur_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
cur_path = os.path.dirname(os.path.abspath(__file__))
windows = platform.system().lower() == 'windows'
if windows:
    new_path = cur_path + '\\..\\..\\datasave\\pos_yyfpdtsmc-' + cur_time + '/'
else:
    new_path = cur_path + '/../../datasave/pos_yyfpdtsmc-' + cur_time + '/'

if __name__ == '__main__':
    uav = UAV(uav_param)
    ctrl_in = pdt_yyf_smc(att_ctrl_param)
    ctrl_out = pdt_yyf_smc(pos_ctrl_param)

    '''reference signal initialization'''
    # ref_amplitude = np.array([2, 2, 1, np.pi / 2])  # x y z psi
    # # ref_amplitude = np.array([0, 0, 0, 0])  # x y z psi
    # ref_period = np.array([5, 5, 5, 5])
    # ref_bias_a = np.array([0., 0., 0., 0.])
    # ref_bias_phase = np.array([0, np.pi / 2, 0, 0])
    
    ref_amplitude = np.array([5, 5, 1, np.pi / 2])  # x y z psi
    ref_period = np.array([10, 10, 5, 10])
    ref_bias_a = np.array([2, 3, 2.0, 0])
    ref_bias_phase = np.array([0, np.pi / 2, 0, 0])
    
    rv = 2.0
    t0 = np.pi / 3
    offset_amplitude = np.array([2., 2., 0., 0.])
    offset_period = np.array([5, 5, 10, 4.])
    offset_bias_a = np.array([0., 0., 2., 0])
    offset_bias_phase = np.array([np.pi / 2 + (1 - 1) * t0, (1 - 1) * t0, 0., 0])

    '''data storage initialization'''
    data_record = data_collector(N=int(uav.time_max / uav.dt))

    '''reference signal'''
    phi_d = theta_d = phi_d_old = theta_d_old = 0.
    dot_phi_d = (phi_d - phi_d_old) / uav.dt
    dot_theta_d = (theta_d - theta_d_old) / uav.dt
    throttle = uav.m * uav.g

    # ref, dot_ref, dot2_ref = ref_uav(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)  # 整体参考信号 xd yd zd psid
    # nu, dot_nu, dot2_nu = ref_uav(uav.time, offset_amplitude, offset_period, offset_bias_a, offset_bias_phase)
    #
    # ref[:] = ref[:] + nu[:]
    # dot_ref[:] = dot_ref[:] + dot_nu[:]
    # dot2_ref[:] = dot2_ref[:] + dot2_nu[:]
    
    ref, dot_ref, dot2_ref = ref_uav_Bernoulli(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
    
    rhod = np.array([phi_d, theta_d, ref[3]]).astype(float)  # 内环参考信号 phi_d theta_d psi_d
    dot_rhod = np.array([dot_phi_d, dot_theta_d, dot_ref[3]]).astype(float)  # 内环参考信号导数

    '''initial error'''
    e_rho = uav.rho1() - rhod
    de_rho = uav.dot_rho1() - dot_rhod
    e_eta = uav.eta() - ref[0: 3]
    de_eta = uav.dot_eta() - dot_ref[0: 3]

    '''observer'''
    obs_in = yyf_pdt_obs(T0=np.array([1, 1, 1]).astype(float),
                         k1=np.array([1, 1, 1]).astype(float),
                         alpha=np.array([0.5, 0.5, 0.5]).astype(float),
                         beta1=np.array([2, 2, 2]).astype(float),
                         beta2=np.array([1, 1, 1]).astype(float),
                         beta3=np.array([1, 1, 1]).astype(float),
                         dim=3,
                         dt=uav.dt)
    obs_out = yyf_pdt_obs(T0=np.array([1, 1, 1]).astype(float),
                          k1=np.array([1, 1, 1]).astype(float),
                          alpha=np.array([0.5, 0.5, 0.5]).astype(float),
                          beta1=np.array([2, 2, 2]).astype(float),
                          beta2=np.array([1, 1, 1]).astype(float),
                          beta3=np.array([1, 1, 1]).astype(float),
                          dim=3,
                          dt=uav.dt)

    '''control'''
    while uav.time < uav.time_max - uav.dt / 2:
        if uav.n % int(1 / uav.dt) == 0:
            print('time: %.2f s.' % (uav.n / int(1 / uav.dt)))

        '''1. generate reference command and uncertainty'''
        # ref, dot_ref, dot2_ref = ref_uav(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
        # nu, dot_nu, dot2_nu = ref_uav(uav.time, offset_amplitude, offset_period, offset_bias_a, offset_bias_phase)
        #
        # ref[:] = ref[:] + nu[:]
        # dot_ref[:] = dot_ref[:] + dot_nu[:]
        # dot2_ref[:] = dot2_ref[:] + dot2_nu[:]
        
        ref, dot_ref, dot2_ref = ref_uav_Bernoulli(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
        
        uncertainty = generate_uncertainty(time=uav.time, is_ideal=IS_IDEAL, att=False)

        '''2. generate outer-loop reference signal 'eta_d' and its 1st derivatives'''
        eta_d = ref[0: 3]
        e_eta = uav.eta() - eta_d
        de_eta = uav.dot_eta() - dot_ref[0: 3]

        '''3. generate outer-loop virtual control command'''
        if USE_OBS_OUT:
            syst_dynamic = -uav.kt / uav.m * uav.dot_eta() + uav.A()
            # print('外环')
            obs_eta = obs_out.observe(de=uav.dot_eta(), dy=syst_dynamic)
        else:
            obs_eta = np.zeros(3)

        '''4. outer loop control'''
        ctrl_out.control_update_outer(e_eta=e_eta,
                                      dot_e_eta=de_eta,
                                      dot_eta=uav.dot_eta(),
                                      kt=uav.kt,
                                      m=uav.m,
                                      dd_ref=dot2_ref[0:3],
                                      obs=obs_eta,
                                      e_m=np.inf,
                                      de_m=np.inf)

        '''5. transfer virtual control command to actual throttle, phi_d, and theta_d'''
        phi_d_old = phi_d
        theta_d_old = theta_d
        phi_d, theta_d, throttle = uo_2_ref_angle_throttle(uo=ctrl_out.control_out, att=uav.uav_att(), m=uav.m, g=uav.g)

        dot_phi_d = (phi_d - phi_d_old) / uav.dt
        dot_theta_d = (theta_d - theta_d_old) / uav.dt
        rhod = np.array([phi_d, theta_d, ref[3]])  # phi_d theta_d psi_d
        dot_rhod = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])  # phi_d theta_d psi_d 的一阶导数

        '''6. inner-loop control'''
        e_rho = uav.rho1() - rhod
        de_rho = np.dot(uav.W(), uav.rho2()) - dot_rhod

        if USE_OBS_IN:
            syst_dynamic = np.dot(uav.dW(), uav.omega()) + np.dot(uav.W(), uav.A_omega() + np.dot(uav.B_omega(), ctrl_in.control_in))
            # print('内环')
            obs_rho = obs_in.observe(dy=syst_dynamic, de=de_rho)
        else:
            obs_rho = np.zeros(3)

        # 将观测器输出前两维度置为 0 即可
        obs_rho[0] = 0.
        obs_rho[1] = 0.
        # 将观测器输出前两维度置为 0 即可

        ctrl_in.control_update_inner(e_rho=e_rho,
                                     dot_e_rho=de_rho,
                                     dd_ref=np.zeros(3),
                                     A_rho=uav.A_rho(),
                                     B_rho=uav.B_rho(),
                                     obs=obs_rho,
                                     att_only=False)

        '''7. rk44 update'''
        action_4_uav = np.array([throttle, ctrl_in.control_in[0], ctrl_in.control_in[1], ctrl_in.control_in[2]])
        uav.rk44(action=action_4_uav, dis=uncertainty, n=1, att_only=False)

        '''8. data record'''
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

    '''datasave'''
    if SAVE:
        os.mkdir(new_path)
        data_record.package2file(new_path)
        obs_out.save_adap_obs_param(new_path, flag='pos')
    '''datasave'''

    data_record.plot_att()
    # data_record.plot_vel()
    data_record.plot_pos()
    # data_record.plot_torque()
    # data_record.plot_throttle()
    data_record.plot_outer_obs()
    # data_record.plot_inner_obs()
    plt.show()
