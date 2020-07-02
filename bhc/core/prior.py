# -*- coding: utf-8 -*-

# License: GPL 3.0

import numpy as np
from numpy.linalg import linalg
from scipy.special.spfun_stats import multigammaln

from bhc.api import AbstractPrior

LOG2PI = np.log(2 * np.pi)
LOG2 = np.log(2)


class NormalInverseWishart(AbstractPrior):
    """
    Reference: MURPHY, Kevin P.
               Conjugate Bayesian analysis of the Gaussian distribution.
               def, v. 1, n. 2σ2, p. 16, 2007.
               https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/readings/bayesGauss.pdf
    """

    def __init__(self, s_mat, r, v, m):
        self.s_mat = s_mat
        self.r = r
        self.v = v
        self.m = m
        self.log_prior0 = NormalInverseWishart.__calc_log_prior(s_mat, r, v)

    def calc_log_mlh(self, x_mat):
        x_mat_l = x_mat.copy()
        x_mat_l = x_mat_l[np.newaxis] if x_mat_l.ndim == 1 else x_mat_l
        n, d = x_mat_l.shape
        s_mat_p, rp, vp = NormalInverseWishart.__calc_posterior(
            x_mat_l, self.s_mat, self.r, self.v, self.m)
        log_prior = NormalInverseWishart.__calc_log_prior(s_mat_p, rp, vp)
        return log_prior - self.log_prior0 - LOG2PI * (n * d / 2.0)

    @staticmethod
    def __calc_log_prior(s_mat, r, v):
        d = s_mat.shape[0]
        log_prior = LOG2 * (v * d / 2.0) + (d / 2.0) * np.log(2.0 * np.pi / r)
        log_prior += multigammaln(v / 2.0, d) - \
            (v / 2.0) * np.log(linalg.det(s_mat))
        return log_prior

    @staticmethod
    def __calc_posterior(x_mat, s_mat, r, v, m):
        n = x_mat.shape[0]
        x_bar = np.mean(x_mat, axis=0)
        rp = r + n
        vp = v + n
        s_mat_t = np.zeros(s_mat.shape) if n == 1 else (
            n - 1) * np.cov(x_mat.T)
        dt = (x_bar - m)[np.newaxis]
        s_mat_p = s_mat + s_mat_t + (r * n / rp) * np.dot(dt.T, dt)
        return s_mat_p, rp, vp

    @staticmethod
    def create(data, g, scale_factor):
        degrees_of_freedom = data.shape[1] + 1
        data_mean = np.mean(data, axis=0)
        data_matrix_cov = np.cov(data.T)
        scatter_matrix = (data_matrix_cov / g).T

        return NormalInverseWishart(scatter_matrix,
                                    scale_factor,
                                    degrees_of_freedom,
                                    data_mean)
