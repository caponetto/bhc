# -*- coding: utf-8 -*-

# License: GPL 3.0

import numpy as np

from numpy.linalg import linalg
from scipy.special.spfun_stats import multigammaln

from core.api import AbstractPrior

LOG2PI = np.log(2 * np.pi)
LOG2 = np.log(2)


class NormalInverseWishart(AbstractPrior):
    """
    Reference: MURPHY, Kevin P. 
               Conjugate Bayesian analysis of the Gaussian distribution. 
               def, v. 1, n. 2σ2, p. 16, 2007.
               https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/readings/bayesGauss.pdf
    """

    def __init__(self, S, r, v, m):
        self.S = S
        self.r = r
        self.v = v
        self.m = m
        self.log_prior0 = NormalInverseWishart.__calc_log_prior(S, r, v)

    def calc_log_mlh(self, X):
        Xl = X.copy()
        Xl = Xl[np.newaxis] if Xl.ndim == 1 else Xl
        n, d = Xl.shape
        Sp, rp, vp = NormalInverseWishart.__calc_posterior(
            Xl, self.S, self.r, self.v, self.m)
        log_prior = NormalInverseWishart.__calc_log_prior(Sp, rp, vp)
        return log_prior - self.log_prior0 - LOG2PI * (n * d / 2.0)

    @staticmethod
    def __calc_log_prior(S, r, v):
        d = S.shape[0]
        log_prior = LOG2 * (v * d / 2.0) + (d / 2.0) * np.log(2.0 * np.pi / r)
        log_prior += multigammaln(v / 2.0, d) - \
            (v / 2.0) * np.log(linalg.det(S))
        return log_prior

    @staticmethod
    def __calc_posterior(X, S, r, v, m):
        n = X.shape[0]
        x_bar = np.mean(X, axis=0)
        rp = r + n
        vp = v + n
        St = np.zeros(S.shape) if n == 1 else (n - 1) * np.cov(X.T)
        dt = (x_bar - m)[np.newaxis]
        Sp = S + St + (r * n / rp) * np.dot(dt.T, dt)
        return Sp, rp, vp
