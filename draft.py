import numpy as np
import scipy as sp
# from datetime import datetime
from time import sleep


if __name__ == '__main__':
    mass_of_earth = 5.9736 * 10 ** 24
    r_of_earth = 6371000
    distance_of_iss = 400000

    v = 0
    for i in range(10):
        v += np.sqrt(mass_of_earth * sp.constants.G/(distance_of_iss+r_of_earth))
        # sleep(60)
    v /= 10.0
    print(v)