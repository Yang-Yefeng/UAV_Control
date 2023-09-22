import os
import datetime
import matplotlib.pyplot as plt

from observer.NESO import neso
from observer.HSMO import hsmo
from observer.AFTO import afto
from observer.RO import ro
from controller.DSMC import dsmc
from controller.FNTSMC import fntsmc
from uav.uav import UAV
from utils.ref_cmd import *
from utils.utils import *
from utils.collector import data_collector

observer_pool = ['neso', 'hsmo', 'afto', 'ro', 'none']
# neso: 非线性扩张状态观测器
# hsmo: 高阶滑模观测器
# afto: 固定时间观测器
# ro:   龙贝格 (勒贝格) 观测器
# none: 没观测器
OBSERVER_IN = observer_pool[2]
OBSERVER_OUT = observer_pool[2]

if __name__ == '__main__':
    '''inner controller initialization'''
    uav = UAV()

    '''controller initialization'''
    ctrl_out = fntsmc(k1=np.array([1.2, 0.8, 0.5]),         # 0.4
                      k2=np.array([0.2, 0.6, 0.5]),     # 0.4
                      alpha=np.array([1.2, 1.5, 1.2]),  # 1.2
                      beta=np.array([0.3, 0.3, 0.5]),       # 0.7
                      gamma=np.array([0.2, 0.2, 0.2]),  # 0.2
                      lmd=np.array([2.0, 2.0, 2.0]),    # 2.0
                      dim=3,
                      dt=uav.dt,
                      ctrl0=np.array([0., 0., uav.m * uav.g]))  #
    ctrl_in = dsmc(ctrl0=np.array([0, 0, 0]).astype(float), dt=uav.dt)

    '''reference signal initialization'''
    ref_amplitude = np.array([2, 2, 1, np.pi / 2])      # x y z psi
    # ref_amplitude = np.array([0, 0, 0, 0])  # x y z psi
    ref_period = np.array([5, 5, 4, 5])
    ref_bias_a = np.array([2, 2, 1, 0])
    ref_bias_phase = np.array([np.pi / 2, 0, 0, 0])

    '''data storage initialization'''
    data_record = data_collector(N=int(uav.time_max / uav.dt))

    '''reference signal initialization'''
    phi_d = theta_d = phi_d_old = theta_d_old = phi_d_old2 = theta_d_old2 = 0.
    dot_phi_d = (phi_d - phi_d_old) / uav.dt
    dot_theta_d = (theta_d - theta_d_old) / uav.dt
    dot2_phi_d = (phi_d - 2 * phi_d_old + phi_d_old2) / uav.dt ** 2
    dot2_theta_d = (theta_d - 2 * theta_d_old + theta_d_old2) / uav.dt ** 2
    throttle = uav.m * uav.g

    ref, dot_ref, _, _ = ref_uav(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)  # 整体参考信号 xd yd zd psid
    rhod = np.array([phi_d, theta_d, ref[3]]).astype(float)  # 内环参考信号 z_d phi_d theta_d psi_d
    dot_rhod = np.array([dot_phi_d, dot_theta_d, dot_ref[3]]).astype(float)  # 内环参考信号导数

    e_I = uav.rho1() - rhod
    dot_e_I = uav.dot_rho1() - dot_rhod
    e_o = uav.eta() - ref[0: 3]
    dot_e_o = uav.dot_eta() - dot_ref[0: 3]

    if OBSERVER_IN == 'neso':
        obs_in = neso(l1=np.array([2., 2., 2.]),
                      l2=np.array([2., 2., 2.]),
                      l3=np.array([1., 1., 1.]),
                      r=np.array([20., 20., 20.]),
                      k1=np.array([0.5, 0.5, 0.5]),
                      k2=np.array([0.001, 0.001, 0.001]),
                      dim=3,
                      dt=uav.dt)
        syst_dynamic_in = np.dot(uav.dot_f1(), uav.rho2()) + np.dot(uav.f1(), uav.f2()) + np.dot(uav.f1(), np.dot(uav.h(), ctrl_in.control))
        obs_in.set_init(x0=e_I, dx0=dot_e_I, syst_dynamic=syst_dynamic_in)
    elif OBSERVER_IN == 'hsmo':
        obs_in = hsmo(alpha1=np.array([10, 3, 3]),
                      alpha2=np.array([2.0, 1.5, 1.5]),
                      alpha3=np.array([3.0, 1.1, 1.1]),
                      p=np.array([20, 20, 10]),
                      dim=3,
                      dt=uav.dt)
        obs_in.set_init(de0=dot_e_I)
    elif OBSERVER_IN == 'afto':
        obs_in = afto(K=np.array([5., 5., 5.]),
                      alpha=np.array([1, 1, 1]),
                      beta=np.array([2, 2, 2]),
                      p=np.array([0.5, 0.5, 0.5]),
                      q=np.array([2, 2, 2]),
                      dt=uav.dt,
                      dim=3,
                      init_de=dot_e_I)
    elif OBSERVER_IN == 'ro':
        obs_in = ro(k1=np.array([7., 7., 7.]),
                    k2=np.array([5., 5., 5.]),
                    dim=3,
                    dt=uav.dt)
    else:
        obs_in = None

    if OBSERVER_OUT == 'neso':
        obs_out = neso(l1=np.array([3., 3., 3.]),
                       l2=np.array([3., 3., 3.]),
                       l3=np.array([1., 1., 3.]),
                       r=np.array([20., 20., 20.]),
                       k1=np.array([0.7, 0.7, 0.7]),
                       k2=np.array([0.001, 0.001, 0.001]),
                       dim=3,
                       dt=uav.dt)
        syst_dynamic_out = -uav.kt / uav.m * uav.dot_eta() + uav.A()
        obs_out.set_init(x0=uav.eta(), dx0=uav.dot_eta(), syst_dynamic=syst_dynamic_out)
    elif OBSERVER_OUT == 'hsmo':
        obs_out = hsmo(alpha1=np.array([3, 3, 6]),
                       alpha2=np.array([2.0, 1.5, 1.5]),
                       alpha3=np.array([3.0, 1.1, 1.1]),
                       p=np.array([10, 10, 15]),
                       dim=3,
                       dt=uav.dt)
        obs_out.set_init(de0=dot_e_I)
    elif OBSERVER_OUT == 'afto':
        obs_out = afto(K=np.array([5., 5., 5.]),
                       alpha=np.array([1, 1, 1]),
                       beta=np.array([2, 2, 2]),
                       p=np.array([0.5, 0.5, 0.5]),
                       q=np.array([2, 2, 2]),
                       dt=uav.dt,
                       dim=3,
                       init_de=uav.dot_eta() - dot_ref[0: 3])
    elif OBSERVER_OUT == 'ro':
        obs_out = ro(k1=np.array([2.0, 2.0, 10.]),
                     k2=np.array([10, 20, 5.]),
                     dim=3,
                     dt=uav.dt, )
    else:
        obs_out = None

    '''嗨嗨嗨，开始控制了'''
    while uav.time < uav.time_max - uav.dt / 2:
        if uav.n % 1000 == 0:
            print('time: ', uav.n * uav.dt)

        '''1. generate reference command and uncertainty'''
        ref, dot_ref, dot2_ref, dot3_ref = ref_uav(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)  # 整体参考信号 xd yd zd psid
        uncertainty = generate_uncertainty(time=uav.time, is_ideal=False)

        '''2. generate outer-loop reference signal 'eta_d' and its 1st, 2nd, and 3rd-order derivatives'''
        e_o_old = e_o.copy()
        dot_e_o_old = dot_e_o.copy()
        eta_d = ref[0: 3]
        dot_eta_d = dot_ref[0: 3]
        dot2_eta_d = dot2_ref[0: 3]
        e_o = uav.eta() - eta_d
        dot_e_o = uav.dot_eta() - dot_eta_d

        '''3. generate outer-loop virtual control command'''
        if OBSERVER_OUT == 'neso':
            syst_dynamic = -uav.kt / uav.m * uav.dot_eta() + uav.A()
            obs_o, _ = obs_out.observe(x=uav.eta(), syst_dynamic=syst_dynamic)
        elif OBSERVER_OUT == 'hsmo':
            syst_dynamic = -uav.kt / uav.m * uav.dot_eta() + uav.A()
            obs_o, _ = obs_out.observe(de=dot_e_o, syst_dynamic=syst_dynamic)
        elif OBSERVER_OUT == 'afto':
            syst_dynamic = -uav.kt / uav.m * uav.dot_eta() + uav.A()
            obs_o, _ = obs_out.observe(syst_dynamic=syst_dynamic,
                                       dot_e_old=dot_e_o_old,
                                       dot_e=dot_e_o)
        elif OBSERVER_OUT == 'ro':
            syst_dynamic = -uav.kt / uav.m * uav.dot_eta() + uav.A()
            obs_o, _ = obs_out.observe(syst_dynamic=syst_dynamic, de=uav.eta())
        else:
            obs_o = np.zeros(3)
        # obs_o = np.array([0., 0., 0.])
        dot_eta = uav.dot_eta()
        e_o = uav.eta() - eta_d
        dot_e_o = dot_eta - dot_eta_d
        ctrl_out.control_update(kp=uav.kt, m=uav.m, vel=uav.uav_vel(), e=e_o, de=dot_e_o, dd_ref=dot2_eta_d, obs=obs_o)

        '''4. transfer virtual control command to actual throttle, phi_d, and theta_d'''
        phi_d_old = phi_d
        theta_d_old = theta_d
        phi_d, theta_d, throttle = uo_2_ref_angle_throttle(uo=ctrl_out.control, att=uav.uav_att(), m=uav.m, g=uav.g)

        # if uav.n % 100 == 0:
        #     print('t:', uav.time)
        #     print('  phi_d: ', rad2deg(phi_d))
        #     print('  theta_d: ', rad2deg(theta_d))
        #     print('  throttle: ', throttle)

        '''5. inner-loop control'''
        dot_phi_d = (phi_d - phi_d_old) / uav.dt
        dot_theta_d = (theta_d - theta_d_old) / uav.dt
        rhod = np.array([phi_d, theta_d, ref[3]])                   # phi_d theta_d psi_d
        dot_rhod = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])   # phi_d theta_d psi_d 的一阶导数

        e_I = uav.rho1() - rhod
        dot_e_I_old = dot_e_I.copy()
        dot_e_I = uav.dot_rho1() - dot_rhod

        if OBSERVER_IN == 'neso':
            syst_dynamic = np.dot(uav.dot_f1(), uav.rho2()) + np.dot(uav.f1(), uav.f2()) + np.dot(uav.f1(), np.dot(uav.h(), ctrl_in.control))
            obs_I, _ = obs_in.observe(x=e_I, syst_dynamic=syst_dynamic)
        elif OBSERVER_IN == 'hsmo':
            syst_dynamic = np.dot(uav.dot_f1(), uav.rho2()) + np.dot(uav.f1(), uav.f2()) + np.dot(uav.f1(), np.dot(uav.h(), ctrl_in.control))
            obs_I, _ = obs_in.observe(syst_dynamic=syst_dynamic, de=dot_e_I)
        elif OBSERVER_IN == 'afto':
            syst_dynamic = np.dot(uav.dot_f1(), uav.rho2()) + np.dot(uav.f1(), uav.f2()) + np.dot(uav.f1(), np.dot(uav.h(), ctrl_in.control))
            obs_I, _ = obs_in.observe(syst_dynamic=syst_dynamic,
                                      dot_e_old=dot_e_I_old,
                                      dot_e=dot_e_I)
        elif OBSERVER_IN == 'ro':
            syst_dynamic = np.dot(uav.dot_f1(), uav.rho2()) + np.dot(uav.f1(), uav.f2()) + np.dot(uav.f1(), np.dot(uav.h(), ctrl_in.control))
            obs_I, _ = obs_in.observe(syst_dynamic=syst_dynamic, de=dot_e_I)
        else:
            obs_I = np.zeros(3)

        # TODO 双闭环时，内环的模型不确定性反映在整个系统中是一个高频信号，这种信号已经不再适合用观测器，故而直接将观测器输出前两维度置为 0 即可
        obs_I[0] = 0.
        obs_I[1] = 0.
        # TODO 双闭环时，内环的模型不确定性反映在整个系统中是一个高频信号，这种信号已经不再适合用观测器，故而直接将观测器输出前两维度置为 0 即可

        dot2_e_I = np.dot(uav.dot_f1(), uav.rho2()) + np.dot(uav.f1(), uav.f2() + np.dot(uav.h(), ctrl_in.control)) + obs_I
        ctrl_in.dot_control(e_I, dot_e_I, dot2_e_I,
                            uav.f1(), uav.f2(), uav.rho2(), uav.h(), uav.dot_f1(), uav.dot_Frho2_f1f2(), uav.dot_f1h(),
                            obs_I)

        ctrl_in.control_update()

        '''6. get the actual control command for UAV'''
        action_4_uav = np.array([throttle, ctrl_in.control[0], ctrl_in.control[1], ctrl_in.control[2]])
        uav.rk44(action=action_4_uav, dis=uncertainty, n=1)

        '''7. '''
        data_block = {'time': uav.time,
                      'control': action_4_uav,
                      'ref_angle': rhod,
                      'ref_pos': ref[0: 3],
                      'ref_vel': dot_ref[0: 3],
                      'd_in': np.array([0., 0., np.dot(uav.f1(), np.array([uncertainty[3], uncertainty[4], uncertainty[5]]))[2] - dot2_ref[3]]),
                      'd_in_obs': obs_I,
                      'd_out': np.array([uncertainty[0], uncertainty[1], uncertainty[2]]) / uav.m,
                      'd_out_obs': obs_o,
                      'state': uav.uav_state_call_back()}
        data_record.record(data=data_block)

    new_path = '../datasave/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '/'
    os.mkdir(new_path)
    data_record.package2file(path=new_path)

    data_record.plot_att()
    data_record.plot_vel()
    data_record.plot_pos()
    data_record.plot_torque()
    data_record.plot_throttle()
    data_record.plot_outer_obs()
    # data_record.plot_inner_obs()
    plt.show()
