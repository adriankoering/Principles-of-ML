import numpy as np
import numpy.linalg as la
matX = np.array([[ 1.00000, 0.00000, 0.00000], [-1.00000, 0.00001, 0.00000]])
vecY = np.array( [ 0.00000, 0.00001, 0.00000])
vecW = la.inv(matX @ matX.T) @ matX @ vecY
print (vecW)
