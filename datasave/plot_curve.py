import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_pos():
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(time, ref_pos[:, 0], 'red')
    plt.plot(time, pos[:, 0], 'blue')
    plt.grid(True)
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xlabel('time(s)')
    plt.title('X')

    plt.subplot(1, 3, 2)
    plt.plot(time, ref_pos[:, 1], 'red')
    plt.plot(time, pos[:, 1], 'blue')
    plt.grid(True)
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xlabel('time(s)')
    plt.title('Y')

    plt.subplot(1, 3, 3)
    plt.plot(time, ref_pos[:, 2], 'red')
    plt.plot(time, pos[:, 2], 'blue')
    plt.grid(True)
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xlabel('time(s)')
    plt.title('Z')


def plot_vel():
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(time, ref_vel[:, 0], 'red')
    plt.plot(time, vel[:, 0], 'blue')
    plt.grid(True)
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xlabel('time(s)')
    plt.title('vx')

    plt.subplot(1, 3, 2)
    plt.plot(time, ref_vel[:, 1], 'red')
    plt.plot(time, vel[:, 1], 'blue')
    plt.grid(True)
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xlabel('time(s)')
    plt.title('vy')

    plt.subplot(1, 3, 3)
    plt.plot(time, ref_vel[:, 2], 'red')
    plt.plot(time, vel[:, 2], 'blue')
    plt.grid(True)
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xlabel('time(s)')
    plt.title('vz')


def plot_att():
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(time, ref_att[:, 0], 'red')
    plt.plot(time, att[:, 0], 'blue')
    plt.grid(True)
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xlabel('time(s)')
    plt.title('phi-roll')

    plt.subplot(1, 3, 2)
    plt.plot(time, ref_att[:, 1], 'red')
    plt.plot(time, att[:, 1], 'blue')
    plt.grid(True)
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xlabel('time(s)')
    plt.title('theta-pitch')

    plt.subplot(1, 3, 3)
    plt.plot(time, ref_att[:, 2], 'red')
    plt.plot(time, att[:, 2], 'blue')
    plt.grid(True)
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xlabel('time(s)')
    plt.title('psi-yaw')


def plot_thrust():
    plt.figure()
    plt.plot(time, thrust, 'red')
    plt.grid(True)
    plt.ylim((0, 1))
    plt.xlabel('time(s)')
    plt.title('thrust')


if __name__ == '__main__':
    data = pd.read_csv('datasave.csv', header=0).to_numpy()
    time = data[:, 0]
    ref_pos = data[:, [1, 3, 5]]
    pos = data[:, [2, 4, 6]]

    ref_vel = data[:, [7, 9, 11]]
    vel = data[:, [8, 10, 12]]

    ref_att = data[:, [13, 15, 17]]
    att = data[:, [14, 16, 18]]

    thrust = data[:, 19]

    plot_pos()
    plot_vel()
    plot_att()
    plot_thrust()
    plt.show()
