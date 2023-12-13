import numpy as np
from matplotlib import pyplot as plt


def sig(x, a):
    return np.fabs(x)**a* np.sign(x)

if __name__ == '__main__':
    x = np.linspace(1, 5, 100)
    a = 3 / 4
    b = 5 / 4
    y1 = sig(x, a)
    y2 = sig(x, b)
    plt.figure()
    plt.plot(x, np.fabs(y1), 'red')
    plt.plot(x, np.fabs(y2), 'blue')
    plt.show()
