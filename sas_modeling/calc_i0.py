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

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def guinier_fit(q, iq, diq=None, dq=None, q_min=0.0, q_max=0.1):
    '''
    Fit data for y = ax + b  return a and b
    '''

    # Identify the bin range for the fit
    id_x = (q >= q_min) & (q <= q_max)

    q2 = q[id_x] ** 2
    logiq = np.log(iq[id_x])
    dlogiq = iq[id_x] / diq[id_x]
    dq2 = 2 * q[id_x] * dq[id_x]

    fit_x = np.zeros(len(q))


    ############## edit this #################
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy, logyerr), full_output=1)

    pfinal = out[0]
    covar = out[1]
    print pfinal
    print covar

    index = pfinal[1]
    amp = 10.0**pfinal[0]

    indexErr = np.sqrt( covar[0][0] )
    ampErr = np.sqrt( covar[1][1] ) * amp

    ############### edit this ################

    # A = np.vstack([q2 / dlogiq, 1.0 / dlogiq]).T
    # p, residuals, _, _ = np.linalg.lstsq(A, iq / diq)

    # # Get the covariance matrix, defined as inv_cov = a_transposed * a
    # err = np.zeros(2)
    # try:
        # inv_cov = np.dot(A.transpose(), A)
        # cov = np.linalg.pinv(inv_cov)
        # err_matrix = math.fabs(residuals) * cov
        # err = [math.sqrt(err_matrix[0][0]), math.sqrt(err_matrix[1][1])]
    # except:
        # err = [-1.0, -1.0]

    # return p, err


def bayesian():
    NotImplemented


def dummy():
    print('\m/ >.< \m/')



if __name__ == '__main__':
    import os

    data_fname = 'dev/1mgml_LysoSANS.sub'; skiprows = 1
    # data_fname = 'exp_data_lysozyme.dat'; skiprows = 0
    assert os.path.exists(data_fname)
    data = np.asfortranarray(np.loadtxt(data_fname, skiprows=skiprows))

    # x = data[:, 0] ** 2
    # y = np.log(data[:, 1])
    # dy = data[:, 2] / data[:, 1]  # d(log(I(q))) = dI(q) / I(q)
    # dx = 2 * data[:, 0] * data[:, 3]  # d(q**2) = 2 q dq

    guinier_fit(data[:, 0], data[:, 1], diq=data[:, 2], dq=data[:, 3],
                q_max=0.07)


    logging.debug('\m/ >.< \m/')
