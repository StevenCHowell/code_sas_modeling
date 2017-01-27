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
    Fit data for y = mx + b
    return m and b

    http://scipy-cookbook.readthedocs.io/items/FittingData.html#id2
    error estimate seems reasonable compared to input data
    '''
    w = 1 / dy

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] * x + p[1]
    errfunc = lambda p, x, y, w: (y - fitfunc(p, x)) * w

    # use the last two points to guess the initial values
    m_guess = (y[-2] - y[-1]) / (x[-2] - x[-1])  # guess the slope from 2 points
    b_guess = y[-1] - m_guess * x[-1]  # gues the y-intercept from 2 points
    p_guess = [m_guess, b_guess]

    out = optimize.leastsq(errfunc, p_guess, args=(x, y, w), full_output=1)

    p_final = out[0]
    m = p_final[0]
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
    m_err = np.sqrt( cov[0, 0] )
    b_err = np.sqrt( cov[1, 1] )

    return m, b, m_err, b_err


def fit_line_v1(x, y, dy):
    '''
    Fit data for y = mx + b
    return m and b

    no error estimates
    '''
    w = 1/ dy ** 2

    A = np.vstack([x * w, 1.0 * w]).T
    p, residuals, _, _ = np.linalg.lstsq(A, y * w)

    m = p[0]
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

    m_err = 0.0
    b_err = 0.0

    return m, b, m_err, b_err


def fit_line_v2(x, y, dy):
    '''
    Fit data for y = mx + b
    return m and b
    essentially the same results as fit_line_v0

    no error estimates
    '''
    w = 1 / dy ** 2

    out = np.polynomial.polynomial.polyfit(x, y, 1, w=w, full=True)
    # does not provide the covariance matrix, not sure how to extract error

    p_final = out[0]
    m = p_final[1]
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
    m_err = 0.0

    return m, b, m_err, b_err


def fit_line_v3(x, y, dy):
    '''
    Fit data for y = mx + b
    return m and b
    method taken from SasView:
    github.com/SasView/sasview/blob/master/src/sas/sascalc/invariant/invariant.py

    error estimate seems reasonable
    '''
    A = np.vstack([x / dy, 1.0 / dy]).T
    p, residuals, _, _ = np.linalg.lstsq(A, y / dy)

    m = p[0]
    b = p[1]

    # Get the covariance matrix, defined as inv_cov = a_transposed * a
    inv_cov = np.dot(A.transpose(), A)
    cov = np.linalg.pinv(inv_cov)
    err_matrix = np.abs(residuals) * cov
    m_err, b_err = np.sqrt(np.diag(err_matrix))

    return m, b, m_err, b_err


def fit_line_v4(x, y, dy):
    '''
    Fit data for y = mx + b
    return m and b

    error estimate seems much too small
    '''
    w = 1 / dy ** 2

    p, cov = np.polyfit(x, y, 1, w=w, cov=True)

    m, b = p

    # From docs page:
    # The diagonal of this matrix (cov) are the
    # variance estimates for each coefficient.
    m_err, b_err = np.sqrt(np.diag(cov))  # standard devaitions
    # m_err, b_err = np.diag(cov)

    return m, b, m_err, b_err


def fit_line_v5(x, y, dy):
    '''
    Fit data for y = mx + b
    return m and b
    method taken from wikipedia:
    https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)#Python

    error estimate seems reasonable comared to input data
    This result is identical to v0 and v7
    '''
    w = 1 / dy ** 2

    n = len(x)
    X = np.array([x, np.ones(n)]).T
    Y = np.array(y).reshape(-1, 1)
    W = np.eye(n) * w  # weight using the inverse of the variance

    # calculate the parameters
    xtwx_inv = np.linalg.inv(X.T.dot(W).dot(X))
    m, b = xtwx_inv.dot(X.T).dot(W).dot(Y).reshape(2)

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
    m_err, b_err = np.sqrt(np.diag(cov))

    return m, b, m_err, b_err


def fit_line_v6(x, y, dy):
    '''
    Fit data for y = mx + b
    return m and b
    method taken from Baird's "Experimentation": pg 138-140
    The dy's in the derivation are not the same as the error of the y values
    This method does not propagate the error
    '''
    var = dy ** 2  # variance, when dy is the standard deviation

    wx = x / var
    wy = y / var

    sum_xy = np.sum(wx * wy)
    sum_x = np.sum(wx)
    sum_y = np.sum(wy)
    sum_x_dy_inv = np.sum(wx / var)
    sum_dy_inv = np.sum(1 / var)
    sum_x2 = np.sum(wx ** 2)

    den = sum_dy_inv * sum_x2 - sum_x * sum_x_dy_inv

    m_num = sum_dy_inv * sum_xy - sum_x_dy_inv * sum_y
    m = m_num / den

    b_num = sum_x2 * sum_y - sum_x * sum_xy
    b = b_num / den

    n = len(x)
    y_fit = m * x + b
    delta_y = y - y_fit
    y_err = np.sqrt(np.sum(delta_y ** 2) / (n - 2))
    m_err = y_err * np.sqrt(n / den)
    b_err = y_err * np.sqrt(sum_x2 / den)

    return m, b, m_err, b_err


def fit_line_v7(x, y, dy):
    '''
    Fit data for y = mx + b
    return m and b
    from Huges & Hase "Measurements and their Uncertainties", pg 69-70
    and Press et al. "Numerical Recipes 3rd Edition", pg 781-783
    '''
    w = 1 / dy ** 2  # weight is the inverse square of the uncertainty

    s = np.sum(w)
    sx = np.sum(w * x)
    sy = np.sum(w * y)
    sxx = np.sum(w * x **2)
    sxy = np.sum(w * x * y)

    den = s * sxx - sx ** 2

    m_num = s * sxy - sx * sy
    m = m_num / den

    b_num = sxx * sy - sx * sxy
    b = b_num / den

    m_err = np.sqrt(s / den)
    b_err = np.sqrt(sxx / den)

    return m, b, m_err, b_err


def fit_line_v8(x, y, dy):
    '''
    Fit data for y = mx + b
    return m and b
    from Press et al. "Numerical Recipes 3rd Edition", pg 781-783
    using numerically robust forulism
    '''
    w = 1 / dy ** 2  # weight is the inverse square of the uncertainty

    s = np.sum(w)
    sx = np.sum(w * x)
    sy = np.sum(w * y)
    sxx = np.sum(w * x **2)
    sxy = np.sum(w * x * y)

    t = 1 / dy * (x - sx / s)
    stt = np.sum(t ** 2)

    m = np.sum(t * y / dy) / stt
    b = (sy - sx * m) / s

    m_err = np.sqrt(1 / stt)
    b_err = np.sqrt((1 + sx ** 2 / (s * stt)) / s)

    return m, b, m_err, b_err


def guinier_fit(q, iq, diq, dq=None, q_min=0.0, q_max=0.1, view_fit=False,
                fit_method=fit_line_v5, save_fname='guiner_fit.html',
                refine=False):
    '''
    perform Guinier fit
    return I(0) and Rg
    '''

    # Identify the range for the fit
    id_x = (q >= q_min) & (q <= q_max)

    q2 = q[id_x] ** 2
    log_iq = np.log(iq[id_x])
    dlog_iq = diq[id_x] / iq[id_x]
    if dq is not None:
        dq2 = 2 * q[id_x] * dq[id_x]

    m, b, m_err, b_err = fit_method(q2, log_iq, dlog_iq)

    rg = np.sqrt(-3 * m)
    rg_err = 3 / (2 * rg) * m_err
    rg, rg_err = round_error(rg, rg_err)

    i0 = np.exp(b)
    i0_err = i0 * b_err
    i0, i0_err = round_error(i0, i0_err)

    rg_q_max = 1.3 / rg
    if rg_q_max < q[id_x][-1]:
        logging.warning('initial q-max too high, 1.3/Rg={} < {}'.format(
            rg_q_max, q[id_x][-1]))
        if refine:
            logging.warning('repeating fit with q-max={}'.format(rg_q_max))
            return guinier_fit(q, iq, diq, dq=dq, q_min=q_min, q_max=rg_q_max,
                               view_fit=view_fit, fit_method=fit_method,
                               save_fname=save_fname)

    if view_fit:
        import make_figures
        q2 = np.insert(q2, 0, 0.0)
        log_iq = np.insert(log_iq, 0, b)
        dlog_iq = np.insert(dlog_iq, 0, b_err)

        fit_line = m * q2 + b
        q_range = q[id_x][[0, -1]]
        make_figures.plot_guinier_fit(q2, log_iq, fit_line, i0, i0_err, rg,
                                      rg_err, dlog_iq, q_range,
                                      save_fname=save_fname)

    return i0, rg, i0_err, rg_err


def round_error(val, val_err, sig_figs=2):
    '''
    Round a value and its error estimate to a certain number
    of significant figures (on the error estimate).  By default 2
    significant figures are used.
    '''
    # round number to a certain number of significant figures

    n = int(np.log10(val_err))  # displacement from ones place
    if val_err >= 1:
        n += 1

    scale = 10 ** (sig_figs - n)
    val = round(val * scale) / scale
    val_err = round(val_err * scale) / scale

    return val, val_err


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
        fit_line_v7,
        fit_line_v8,
    ]

    for fit_method in fit_methods:
        save_fname = 'fit_{}_comparison.html'.format(fit_method.__name__[-2:])
        # save_fname = 'fit_{}_comparison.html'.format(fit_method.func_name[-2:])
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

    # data_fname = 'data/1mgml_LysoSANS.sub'; skiprows = 1
    skiprows = 0
    data_fname = 'data/1mgml_lys_sans.dat'; q_max = 0.091  # lys
    # data_fname = 'data/5mgml_nist_mab_sans.dat'; q_max = 0.0296  # mab
    assert os.path.exists(data_fname)
    data = np.asfortranarray(np.loadtxt(data_fname, skiprows=skiprows))

    # data[:, 1:3] *= 1 / data[0, 1]

    # column 4 is the effective q-values, accounting for the beam spread

    if True:
        make_figures.plot_iq_and_guinier(data[:, 0], data[:, 1], data[:, 2],
                                        save_fname='I(q)_and_guinier-no_scale.html')


    # scale the data
    # data[:, 1:3] *= 1 / data[0, 1]  # set the first measured point to 1
    # data[:, 1:3] *= 10 / data[0, 1]  # set the first measured point to 10
    # data[:, 1:3] *= 100 / data[0, 1]  # set the first measured point to 100
    # data[:, 1:3] *= 1000 / data[0, 1]  # set the first measured point to 1000

    # compare_guinier_fit(data[:, 0], data[:, 1], data[:, 2], q_max=q_max,
                        # refine=True)
    save_fname = data_fname.replace('.dat', '.html')
    i0, rg, i0_err, rg_err = guinier_fit(data[:, 0], data[:, 1], data[:,2],
                                         dq=data[:, 3], q_max=q_max,
                                         view_fit=True, fit_method=fit_line_v8,
                                         refine=True, save_fname=save_fname)

    logging.debug('\m/ >.< \m/')
