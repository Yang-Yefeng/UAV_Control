import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma

def f(x, k):
    return np.sign(1 - x) * np.fabs(1 - x) ** k - 1 + x ** k

if __name__ == '__main__':
    # k = 3. / 5.
    # x = np.linspace(0, 1, 500)
    # y = f(x, k)
    # plt.figure()
    # plt.plot(x,y)
    # plt.show()
    a = {'aa': 1, 'bb':2, 'cc': 344}
    print(a.keys())
    for key in a.keys():
        print(key)
        print(a[key])
