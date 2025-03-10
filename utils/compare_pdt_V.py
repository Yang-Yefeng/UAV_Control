import numpy as np
import pandas as pd
from scipy.special import gamma
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


def f1(t, v, b2, b3, T0, a):
    if v < 1e-5:
        v = 0
    return -np.pi / ((2 - 2 * a) * T0 * np.sqrt(b2 * b3)) * (b2 * v ** a + b3 * v ** (2 - a))

def pdt_f(t, v, b1, b2, b3, T0, a, k1, ep):
    a1s = (1 - a * k1) / (2 - 2 * a)
    a2s = k1 - a1s
    k0 = gamma(a1s) * gamma(a2s) / gamma(k1) / (2 - 2 * a) / b2 ** k1 * (b2 / b3) ** a1s / T0
    if v < 1e-5:
        v = 0
    return -k0 * (2 * b1 * v + b2 * v ** a + b3 * v ** (2 - a)) ** k1 + ep


if __name__ == '__main__':
    t0=0
    tf=8
    v0=[5]

    solution1 = solve_ivp(f1, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000), args=(1, 1, 10, 0.5))
    solution2 = solve_ivp(pdt_f, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000), args=(1, 1, 1, 10, 0.5, 1, 0.))
    solution3 = solve_ivp(pdt_f, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000), args=(2, 1, 1, 10, 0.5, 1, 0.))
    solution4 = solve_ivp(pdt_f, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000), args=(3, 1, 1, 10, 0.5, 1, 0.))
    solution5 = solve_ivp(pdt_f, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000), args=(4, 1, 1, 10, 0.5, 1, 0.))
    # solution6 = solve_ivp(pdt_f, [t0, tf], v0, method='RK45', t_eval=np.linspace(t0, tf, 1000), args=(1, 1, 1, 10, 0.5, 1, 0.))

    data = {
        "time": solution1.t,
        "b1=0": solution1.y[0],
        "b1=1": solution2.y[0],
        "b1=2": solution3.y[0],
        "b1=3": solution4.y[0],
        "b1=4": solution5.y[0],
    }
    pd.DataFrame(data).to_csv('different_b1.csv', sep=',', index=False)

    plt.figure()
    plt.plot(solution1.t, solution1.y[0], color='red')
    plt.plot(solution2.t, solution2.y[0], color='blue')
    plt.plot(solution3.t, solution3.y[0], color='green')
    plt.plot(solution4.t, solution4.y[0], color='orange')
    plt.plot(solution5.t, solution5.y[0], color='purple')

    plt.show()
