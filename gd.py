"""
Generic implementation of gradient descent.
"""

from numpy import *
import util


def gd(func, grad, x0, numIter, stepSize):
    """
    Perform gradient descent on some function func, where grad(x)
    computes its gradient at position x. Begin at position x0 and run
    for exactly numIter iterations. Use stepSize/sqrt(t+1) as a
    step-size, where t is the iteration number.

    We return the final solution as well as the trajectory of function
    values.
    """

    x = array(x0, copy=True) if not isinstance(x0, (int, float)) else x0

    trajectory = zeros(numIter + 1)
    trajectory[0] = func(x)

    for iter in range(numIter):
        g = grad(x)
        eta = stepSize / sqrt(iter + 1)
        x = x - eta * g
        trajectory[iter + 1] = func(x)

    return (x, trajectory)
