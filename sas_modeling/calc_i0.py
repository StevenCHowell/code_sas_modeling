#!/usr/bin/env python
#coding:utf-8
'''
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: calculating the Guinier fit
    Created: 12/21/2016

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
'''

import logging
import numpy as np
from scipy import optimize

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def fit_line_v0(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b
    essentially the same results as fit_line_v2

    error estimate seems reasonable compared to input data
    '''

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] * x + p[1]
    errfunc = lambda p, x, y, yerr: (y - fitfunc(p, x)) / yerr

    # use the last two points to guess the initial values
    a_guess = (y[-2] - y[-1]) / (x[-2] - x[-1])
    b_guess = y[-1] - a_guess * x[-1]
    p_guess = [a_guess, b_guess]
    out = optimize.leastsq(errfunc, p_guess, args=(x, y, dy), full_output=1)

    p_final = out[0]
    a = p_final[0]
    b = p_final[1]

    # from the docs page:
    # cov_x : ndarray
    #   Uses the fjac and ipvt optional outputs to construct an estimate
    #   of the jacobian around the solution. None if a singular matrix
    #   encountered (indicates very flat curvature in some direction).
    #   This matrix must be multiplied by the residual variance to get the
    #   covariance of the parameter estimates – see curve_fit.
    #
    # curve_fit documentation says:
    #   The diagonals provide the variance of the parameter estimate.
    #   To compute one standard deviation errors on the parameters use
    #   perr = np.sqrt(np.diag(pcov)).
    #
    #   How the sigma parameter affects the estimated covariance depends
    #   on absolute_sigma argument, as described above.
    #
    #   If the Jacobian matrix at the solution doesn’t have a full rank,
    #   then ‘lm’ method returns a matrix filled with np.inf, on the other
    #   hand ‘trf’ and ‘dogbox’ methods use Moore-Penrose pseudoinverse to
    #   compute the covariance matrix.
    cov = out[1]
    a_err = np.sqrt( cov[0, 0] )
    b_err = np.sqrt( cov[1, 1] )

    return a, b, a_err, b_err


def fit_line_v1(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b

    no error estimate
    '''

    A = np.vstack([x / dy, 1.0 / dy]).T  # weight each row by 1 / dy
    p, residuals, _, _ = np.linalg.lstsq(A, y / dy)

    a = p[0]
    b = p[1]

    # from the docs page:
    # residuals : {(), (1,), (K,)} ndarray
    #   Sums of residuals; squared Euclidean 2-norm for each column in b - a*x.
    #   If the rank of a is < N or M <= N, this is an empty array. If b is
    #   1-dimensional, this is a (1,) shape array. Otherwise the shape is (K,).
    # rank : int
    #   Rank of matrix a.
    # s : (min(M, N),) ndarray
    #   Singular values of a.

    a_err = 0.0
    b_err = 0.0

    return a, b, a_err, b_err


def fit_line_v2(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b
    essentially the same results as fit_line_v0

    no error estimate
    '''

    out = np.polynomial.polynomial.polyfit(x, y, 1, w=1/dy, full=True)
    # does not provide the covariance matrix, not sure how to extract error

    p_final = out[0]
    a = p_final[1]
    b = p_final[0]

    # from the docs page:
    # [residuals, rank, singular_values, rcond] : list
    # These values are only returned if full = True
    #   resid – sum of squared residuals of the least squares fit
    #   rank  – the numerical rank of the scaled Vandermonde matrix
    #   sv    – singular values of the scaled Vandermonde matrix
    #   rcond – value of rcond.
    # For more details, see linalg.lstsq.

    b_err = 0.0
    a_err = 0.0

    return a, b, a_err, b_err


def fit_line_v3(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b
    method taken from SasView:
    github.com/SasView/sasview/blob/master/src/sas/sascalc/invariant/invariant.py

    error estimate seems much too small
    '''
    A = np.vstack([x / dy, 1.0 / dy]).T
    p, residuals, _, _ = np.linalg.lstsq(A, y / dy)

    a = p[0]
    b = p[1]

    # Get the covariance matrix, defined as inv_cov = a_transposed * a
    inv_cov = np.dot(A.transpose(), A)
    cov = np.linalg.pinv(inv_cov)
    err_matrix = np.abs(residuals) * cov
    a_err = np.sqrt(err_matrix[0][0])
    b_err = np.sqrt(err_matrix[1][1])

    return a, b, a_err, b_err


def fit_line_v4(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b

    error estimate seems much too small
    '''

    p, cov = np.polyfit(x, y, 1, w=1/dy, cov=True)

    a, b = p

    # From docs page:
    # The diagonal of this matrix (cov) are the
    # variance estimates for each coefficient.
    a_err, b_err = np.sqrt(np.diag(cov))  # standard devaitions

    return a, b, a_err, b_err


def fit_line_v5(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b
    method taken from wikipedia:
    https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)#Python

    error estimate seems reasonable comared to input data
    '''

    m = len(x)
    X = np.array([x, np.ones(m)]).T
    Y = np.array(y).reshape(-1, 1)
    W = np.eye(m) / dy  #** 2

    # calculate the parameters
    xtwx_inv = np.linalg.inv(X.T.dot(W).dot(X))
    a, b = xtwx_inv.dot(X.T).dot(W).dot(Y)

    # calculate the error of the parameters:
    # (X.T * W * X)^-1 * X.T * W * M * W.T * X * (X.T * W.T * X)^-1
    # cov_xy = covariance(x, y)
    # var_x = covariance(x, x)
    # var_y = covariance(y, y)
    # M = np.eye(m) * dy ** 2

    # xtwtx_inv = np.linalg.inv(X.T.dot(W.T).dot(X))
    # M_beta = xtwx_inv.dot(X.T).dot(W).dot(M).dot(W.T).dot(X).dot(xtwtx_inv)
    # M_beta = xtwx_inv  # because M = W^-1

    cov = xtwx_inv
    a_err, b_err = np.sqrt(np.diag(cov))

    return a, b, a_err, b_err


def fit_line_v6(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b
    method taken from Experimentation by Baird: pg 138-140
    The dy's in the derivation are not the same as the error of the y values
    '''
    wx = x / dy
    wy = y / dy

    sum_xy = np.sum(wx * wy)
    sum_x = np.sum(wx)
    sum_y = np.sum(wy)
    sum_x_dy_inv = np.sum(wx / dy)
    sum_dy_inv = np.sum(1 / dy)
    sum_x2 = np.sum(wx ** 2)

    den = sum_dy_inv * sum_x2 - sum_x * sum_x_dy_inv

    a_num = sum_dy_inv * sum_xy - sum_x_dy_inv * sum_y
    a = a_num / den

    b_num = sum_x2 * sum_y - sum_x * sum_xy
    b = b_num / den

    n = len(x)
    y_fit = a * x + b
    delta_y = y - y_fit
    y_err = np.sqrt(np.sum(delta_y ** 2) / (n - 2))
    a_err = y_err * np.sqrt(n / den)
    b_err = y_err * np.sqrt(sum_x2 / den)

    return a, b, a_err, b_err


def guinier_fit(q, iq, diq, dq=None, q_min=0.0, q_max=0.1, view_fit=False,
                fit_method=fit_line_v5, save_fname='guiner_fit.html'):
    '''
    perform Guinier fit
    return I(0) and Rg
    '''

    # Identify the range for the fit
    id_x = (q >= q_min) & (q <= q_max)

    q2 = q[id_x] ** 2
    log_iq = np.log(iq[id_x])
    dlog_iq = iq[id_x] / diq[id_x]
    if dq is not None:
        dq2 = 2 * q[id_x] * dq[id_x]

    a, b, a_err, b_err = fit_method(q2, log_iq, dlog_iq)

    rg = np.sqrt(-3 * a)
    rg_err = 3 / (2 * rg) * a_err

    i0 = np.exp(b)
    i0_err = i0 * b_err

    if view_fit:
        import make_figures
        q2 = np.insert(q2, 0, 0.0)
        log_iq = np.insert(log_iq, 0, b)
        dlog_iq = np.insert(dlog_iq, 0, b_err)

        y_fit = a * q2 + b
        make_figures.plot_guinier_fit(q2, log_iq, y_fit, i0, i0_err, rg, rg_err,
                                      yerr=dlog_iq, save_fname=save_fname)

    return i0, rg, i0_err, rg_err


def compare_guinier_fit(q, iq, diq, **args):
    '''
    perform Guinier fit
    return I(0) and Rg
    '''

    fit_methods = [
        fit_line_v0,
        fit_line_v1,
        fit_line_v2,
        fit_line_v3,
        fit_line_v4,
        fit_line_v5,
        fit_line_v6,
    ]

    for fit_method in fit_methods:
        save_fname = 'fit_{}_comparison.html'.format(fit_method.func_name[-2:])
        i0, rg, i0_err, rg_err = guinier_fit(q, iq, diq, fit_method=fit_method,
                                             save_fname=save_fname,
                                             view_fit=True, **args)


def covariance(x, y):
    assert len(x) == len(y)

    cov = ((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1)

    return cov


def bayesian():
    NotImplemented


if __name__ == '__main__':
    import os
    import make_figures

    data_fname = 'data/1mgml_LysoSANS.sub'; skiprows = 1
    # data_fname = 'exp_data_lysozyme.dat'; skiprows = 0
    assert os.path.exists(data_fname)
    data = np.asfortranarray(np.loadtxt(data_fname, skiprows=skiprows))

    compare_guinier_fit(data[:, 0], data[:, 1], data[:, 2])

    # i0, rg, i0_err, rg_err = guinier_fit(data[:, 0], data[:, 1], data[:,2],
                                         # dq=data[:, 3], q_max=0.07,
                                         # view_fit=True, fit_method=fit_line_v6)

    logging.debug('\m/ >.< \m/')
