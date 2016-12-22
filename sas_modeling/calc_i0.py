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


def guinier_fit(q, iq, diq=None, dq=None, q_min=0.0, q_max=0.1, view_fit=False):
    '''
    perform Guinier fit
    return I(0) and Rg
    '''

    # Identify the range for the fit
    id_x = (q >= q_min) & (q <= q_max)

    q2 = q[id_x] ** 2
    log_iq = np.log(iq[id_x])
    dlog_iq = iq[id_x] / diq[id_x]
    dq2 = 2 * q[id_x] * dq[id_x]

    # a, b, a_err, b_err = fit_line(q2, log_iq, dlog_iq)
    vals0 = fit_line_v0(q2, log_iq, dlog_iq)
    vals4 = fit_line_v4(q2, log_iq, dlog_iq)
    vals3 = fit_line_v3(q2, log_iq, dlog_iq)
    vals2 = fit_line_v2(q2, log_iq, dlog_iq)
    vals1 = fit_line_v1(q2, log_iq, dlog_iq)

    a, b, a_err, b_err = vals3; save_fname = 'fit_v3_comparison.html'
    # a, b, a_err, b_err = vals2; save_fname = 'fit_v2_comparison.html'
    # a, b, a_err, b_err = vals1; save_fname = 'fit_v1_comparison.html'
    a, b, a_err, b_err = vals0; save_fname = 'fit_v0_comparison.html'




    rg = np.sqrt(-3 * a)
    rg_err = -3 / (2 * rg) * a_err

    i0 = np.exp(b)
    i0_err = i0 * b_err

    if view_fit:
        import make_figures
        q2 = np.insert(q2, 0, 0.0)
        log_iq = np.insert(log_iq, 0, b)
        dlog_iq = np.insert(dlog_iq, 0, b_err)

        y_fit = a * q2 + b
        make_figures.plot_fit(q2, log_iq, y_fit, yerr=dlog_iq,
                              save_fname=save_fname)

    return i0, rg, i0_err, rg_err


def fit_line_v1(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b
    '''

    A = np.vstack([x / dy, 1.0 / dy]).T  # weight each row by 1 / dy
    p, residuals, _, _ = np.linalg.lstsq(A, y / dy)

    a = p[0]
    b = p[1]

    covar = out[1]
    a_err = np.sqrt( covar[0][0] )
    b_err = np.sqrt( covar[1][1] ) * b

    return a, b, a_err, b_err


def fit_line_v2(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b
    essentially the same results as fit_line_v0
    '''

    out = np.polynomial.polynomial.polyfit(x, y, 1, w=1/dy, full=True)

    p_final = out[0]
    a = p_final[1]
    b = p_final[0]

    residuals = out[1]
    b_err = np.sqrt( residuals[0] ) * b
    a_err = np.sqrt( residuals[1] )

    return a, b, a_err, b_err


def fit_line_v3(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b
    method taken from SasView:
    github.com/SasView/sasview/blob/master/src/sas/sascalc/invariant/invariant.py
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
    '''

    p, cov = np.polyfit(x, y, 1, w=1/dy, cov=True)

    a = p[0]
    b = p[1]

    a_err = np.sqrt( cov[0, 0] )
    b_err = np.sqrt( cov[1, 1] ) * b

    return a, b, a_err, b_err


def fit_line_v0(x, y, dy):
    '''
    Fit data for y = ax + b
    return a and b
    essentially the same results as fit_line_v2
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

    cov = out[1]
    a_err = np.sqrt( cov[0, 0] )
    b_err = np.sqrt( cov[1, 1] ) * b

    return a, b, a_err, b_err


def bayesian():
    NotImplemented


if __name__ == '__main__':
    import os
    import make_figures

    data_fname = 'dev/1mgml_LysoSANS.sub'; skiprows = 1
    # data_fname = 'exp_data_lysozyme.dat'; skiprows = 0
    assert os.path.exists(data_fname)
    data = np.asfortranarray(np.loadtxt(data_fname, skiprows=skiprows))

    # x = data[:, 0] ** 2
    # y = np.log(data[:, 1])
    # dy = data[:, 2] / data[:, 1]  # d(log(I(q))) = dI(q) / I(q)
    # dx = 2 * data[:, 0] * data[:, 3]  # d(q**2) = 2 q dq

    i0, rg, i0_err, rg_err = guinier_fit(data[:, 0], data[:, 1],
                                         diq=data[:,2], dq=data[:, 3],
                                         q_max=0.07, view_fit=True)

    logging.debug('\m/ >.< \m/')
