import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    rmb = 10962.8 + 4533.34 + 3116.35 + 4651.35 + 4886.64 + 4445.76
    hkd = 12693.44
    total_hkd = rmb / 0.92 + 12693.44
    print(total_hkd)
    print(80000 - total_hkd)
