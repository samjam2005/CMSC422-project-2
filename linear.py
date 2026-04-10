"""
Implementation of *regularized* linear classification/regression by
plug-and-play loss functions
"""

from numpy import *
from pylab import *

from binary import *
from gd import *
import util


class LossFunction:
    def loss(self, Y, Yhat):
        util.raiseNotDefined()

    def lossGradient(self, X, Y, Yhat):
        util.raiseNotDefined()


class SquaredLoss(LossFunction):
    """
    Squared loss is (1/2) * sum_n (y_n - y'_n)^2
    """

    def loss(self, Y, Yhat):
        return 0.5 * dot(Y - Yhat, Y - Yhat)

    def lossGradient(self, X, Y, Yhat):
        return -sum((Y - Yhat) * X.T, axis=1)


class LogisticLoss(LossFunction):
    """
    Logistic loss is sum_n log(1 + exp(- y_n * y'_n))
    """

    def loss(self, Y, Yhat):
        margin = Y * Yhat
        return sum(logaddexp(0.0, -margin))

    def lossGradient(self, X, Y, Yhat):
        margin = Y * Yhat
        sig = 1.0 / (1.0 + exp(margin))
        return -sum((Y * sig) * X.T, axis=1)


class HingeLoss(LossFunction):
    """
    Hinge loss is sum_n max{ 0, 1 - y_n * y'_n }
    """

    def loss(self, Y, Yhat):
        margin = Y * Yhat
        return sum(maximum(0.0, 1.0 - margin))

    def lossGradient(self, X, Y, Yhat):
        margin = Y * Yhat
        active = (margin < 1).astype(float)
        return -sum((Y * active) * X.T, axis=1)


class LinearClassifier(BinaryClassifier):
    """
    This class defines an arbitrary linear classifier parameterized by
    a loss function and a ||w||^2 regularizer.
    """

    def __init__(self, opts):
        self.opts = opts
        self.reset()

    def reset(self):
        self.weights = 0

    def online(self):
        return False

    def __repr__(self):
        return "w=" + repr(self.weights)

    def predict(self, X):
        if isinstance(self.weights, int):
            return 0
        return dot(X, self.weights)

    def getRepresentation(self):
        return self.weights

    def train(self, X, Y):
        if isinstance(self.weights, int):
            self.weights = zeros(X.shape[1])

        lossFn = self.opts['lossFunction']
        lambd = self.opts['lambda']
        numIter = self.opts['numIter']
        stepSize = self.opts['stepSize']

        def func(w):
            Yhat = dot(X, w)
            return lossFn.loss(Y, Yhat) + (lambd / 2.0) * dot(w, w)

        def grad(w):
            Yhat = dot(X, w)
            return lossFn.lossGradient(X, Y, Yhat) + lambd * w

        w, trajectory = gd(func, grad, self.weights, numIter, stepSize)
        self.weights = w
        self.trajectory = trajectory
