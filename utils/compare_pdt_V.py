import numpy as np
import pandas as pd
from scipy.special import gamma
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


def f1(t, v):
    b1 = 1
    b2 = 1
    b3 = 1
    T0 = 5
    a = 0.5
    k1 = 1
    a1s = (1 - a * k1) / (2 - 2 * a)
    a2s = k1 - a1s
    k0 = gamma(a1s) * gamma(a2s) / gamma(k1) / (2 - 2 * a) / b2 ** k1 * (b2 / b3) ** a1s / T0
    if v < 1e-5:
        v = 0
    return -np.pi / ((2 - 2 * a) * T0 * np.sqrt(b2 * b3)) * (b2 * v ** a + b3 * v ** (2 - a))

def f2(t, v):
    b1 = 1
    b2 = 1
    b3 = 1
    T0 = 10
    a = 0.5
    k1 = 1
    a1s = (1 - a * k1) / (2 - 2 * a)
    a2s = k1 - a1s
    k0 = gamma(a1s) * gamma(a2s) / gamma(k1) / (2 - 2 * a) / b2 ** k1 * (b2 / b3) ** a1s / T0
    if v < 1e-5:
        v = 0
    return -k0 * (2 * b1 * v + b2 * v ** a + b3 * v ** (2 - a)) ** k1


def f3(t, v):
    b1 = 1
    b2 = 1
    b3 = 1
    T0 = 8
    a = 0.5
    k1 = 1
    a1s = (1 - a * k1) / (2 - 2 * a)
    a2s = k1 - a1s
    k0 = gamma(a1s) * gamma(a2s) / gamma(k1) / (2 - 2 * a) / b2 ** k1 * (b2 / b3) ** a1s / T0
    if v < 1e-5:
        v = 0
    return -k0 * (2 * b1 * v + b2 * v ** a + b3 * v ** (2 - a)) ** k1


def f4(t, v):
    b1 = 1
    b2 = 1
    b3 = 1
    T0 = 6
    a = 0.5
    k1 = 1
    a1s = (1 - a * k1) / (2 - 2 * a)
    a2s = k1 - a1s
    k0 = gamma(a1s) * gamma(a2s) / gamma(k1) / (2 - 2 * a) / b2 ** k1 * (b2 / b3) ** a1s / T0
    if v < 1e-5:
        v = 0
    return -k0 * (2 * b1 * v + b2 * v ** a + b3 * v ** (2 - a)) ** k1


def f5(t, v):
    b1 = 1
    b2 = 1
    b3 = 1
    T0 = 4
    a = 0.5
    k1 = 1
    a1s = (1 - a * k1) / (2 - 2 * a)
    a2s = k1 - a1s
    k0 = gamma(a1s) * gamma(a2s) / gamma(k1) / (2 - 2 * a) / b2 ** k1 * (b2 / b3) ** a1s / T0
    if v < 1e-5:
        v = 0
    return -k0 * (2 * b1 * v + b2 * v ** a + b3 * v ** (2 - a)) ** k1



if __name__ == '__main__':
    t0=0
    tf=5
    v0=[5]

    solution1 = solve_ivp(f1, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000))    # 提出的
    solution2 = solve_ivp(f2, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000))
    solution3 = solve_ivp(f3, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000))
    solution4 = solve_ivp(f4, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000))
    solution5 = solve_ivp(f5, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000))

    data = {
        "time": solution1.t,
        # "beta1_1": solution1.y[0],
        "beta1_2": solution2.y[0],
        "beta1_3": solution3.y[0],
        "beta1_4": solution4.y[0],
        "beta1_5": solution5.y[0],
    }
    pd.DataFrame(data).to_csv('v_different_pdt.csv', sep=',', index=False)

    plt.figure()
    # plt.plot(solution1.t, solution1.y[0], color='red')
    plt.plot(solution2.t, solution2.y[0], color='blue')
    plt.plot(solution3.t, solution3.y[0], color='green')
    plt.plot(solution4.t, solution4.y[0], color='orange')
    plt.plot(solution5.t, solution5.y[0], color='purple')

    plt.show()
