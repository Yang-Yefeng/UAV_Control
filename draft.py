import numpy as np


a = np.zeros(6)
b = np.array([2,3,4,5,6,8])

c = np.hstack((a,b))
print(c)