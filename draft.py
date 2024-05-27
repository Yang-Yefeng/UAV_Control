import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    a = np.array([-2, 0, 5])
    b = np.array([1, 1, 1])
    print(np.clip(np.sign(a - b), 0, 1))
