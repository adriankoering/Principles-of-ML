import numpy as np
import numpy.linalg as la

import numpy as np
import imageio.v3 as iio
import numpy.linalg as la
import scipy.ndimage as img

matX = np.array([[ 1.00000, 0.00000, 0.00000], [-1.00000, 0.00001, 0.00000]])
vecY = np.array( [ 0.00000, 0.00001, 0.00000])

def marvelous():
    vecW = la.inv(matX @ matX.T) @ matX @ vecY
    print("Marvelous: ", vecW)

def task112():
    Q, R = la.qr(matX)
    print(R.T.shape, Q.shape)
    vecW = R.T @ Q.T @ vecY
    print("QR: ", vecW)


def task113():
    # TODO: matX.T ?
    vecW, residuals, rank, s = la.lstsq(matX.T, vecY)
    print("LeastSquares: ", vecW)

def task114():
    # * real numbers aint floats - floats are only an inexact approximation
    # machine precision is the lower limit of significant digits of any number
    # large and small floating point numbers push the boundaries of representable numbers
    # and hence can exceed deviate from the exact numerical result

    from sympy import Matrix
    eXXT = Matrix(matX) @ Matrix(matX).T
    print("Excat: X X.T", eXXT)

    XXT = matX @ matX.T # [[1, -1], [-1, 1]]
    print("Numerical X X.T: " , XXT)

    # you probably cant trust any software fully
    # can also not trust any data

def task120():
    # rule110 = {
    #     (0, 0, 0): 0,
    #     (0, 0, 1): 1,
    #     (0, 1, 0): 1,
    #     (0, 1, 1): 1,
    #     (1, 0, 0): 0,
    #     (1, 0, 1): 1,
    #     (1, 1, 0): 1,
    #     (1, 1, 1): 0
    # }


    # rule126 = {
    #     (0, 0, 0): 0,
    #     (0, 0, 1): 1,
    #     (0, 1, 0): 1,
    #     (0, 1, 1): 1,
    #     (1, 0, 0): 1,
    #     (1, 0, 1): 1,
    #     (1, 1, 0): 1,
    #     (1, 1, 1): 0
    # }

    X = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=float)
    Xp = np.power(-1, X)
    
    rule110 = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=float)
    rule126 = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=float)
    
    rule110p = np.power(-1, rule110)
    rule126p = np.power(-1, rule126)

    def solve_and_compare(X, rule):
        w, residual, rank, s = la.lstsq(X, rule)
        print(f"Rule: {rule} solved to be {w}")


    solve_and_compare(Xp, rule110p)
    solve_and_compare(Xp, rule126p)


def task122():
    # TODO: generalize phi to n-element vectors
    # [combinations of x**n for n in range(len(x))] ? 
    pass

    
def task123():
    def phi(x):
        x1, x2, x3 = x
        return np.array([1, x1, x2, x3, x1*x2, x1*x3, x2*x3, x1*x2*x3])

    X = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=float)
    Xp = np.power(-1, X)

    Phi = np.array([phi(x) for x in Xp]).T
    print(Phi)

    rule110 = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=float)
    rule126 = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=float)
    
    rule110p = np.power(-1, rule110)
    rule126p = np.power(-1, rule126)

    w, residual, rank, s = la.lstsq(Phi, rule110p)
    print(w, rule110p)

    # TODO: repeat for 126
    # TODO: result should differ? not seeing that yet?

def task130():
    imgF = iio.imread("tree.png", mode="L").astype(float)

    
    def binarize(imgF):
        blur_med = img.filters.gaussian_filter(imgF, sigma=0.50)
        blur_high = img.filters.gaussian_filter(imgF, sigma=1.00)
        imgD = np.abs(blur_med - blur_high)
        return img.morphology.binary_closing(np.where(imgD < 0.1 * imgD.max(), 0, 1))
    
    imgB = binarize(imgF)



def main():
    marvelous()
    # task112()
    # task113()
    # task114()
    # task120()
    task123()

if __name__ == "__main__":
    main()