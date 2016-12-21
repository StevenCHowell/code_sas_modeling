#!/usr/bin/env python
#coding:utf-8
"""
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: calculating the R-factor from a SasCalc run
    Created: 12/09/2016

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
"""

import logging
import numpy as np

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def fit(self, power=None, qmin=None, qmax=None):
    """
    Fit data for y = ax + b  return a and b
    :param power: a fixed, otherwise None
    :param qmin: Minimum Q-value
    :param qmax: Maximum Q-value
    """
    if qmin is None:
        qmin = self.qmin
    if qmax is None:
        qmax = self.qmax

    # Identify the bin range for the fit
    idx = (self.data.x >= qmin) & (self.data.x <= qmax)

    fx = numpy.zeros(len(self.data.x))

    # Uncertainty
    if type(self.data.dy) == numpy.ndarray and \
        len(self.data.dy) == len(self.data.x) and \
        numpy.all(self.data.dy > 0):
        sigma = self.data.dy
    else:
        sigma = numpy.ones(len(self.data.x))

    # Compute theory data f(x)
    fx[idx] = self.data.y[idx]

    # Linearize the data
    if self.model is not None:
        linearized_data = self.model.linearize_data(\
                                        LoaderData1D(self.data.x[idx],
                                                     fx[idx],
                                                     dy=sigma[idx]))
    else:
        linearized_data = LoaderData1D(self.data.x[idx],
                                       fx[idx],
                                       dy=sigma[idx])

    ##power is given only for function = power_law
    if power != None:
        sigma2 = linearized_data.dy * linearized_data.dy
        a = -(power)
        b = (numpy.sum(linearized_data.y / sigma2) \
             - a * numpy.sum(linearized_data.x / sigma2)) / numpy.sum(1.0 / sigma2)


        deltas = linearized_data.x * a + \
                numpy.ones(len(linearized_data.x)) * b - linearized_data.y
        residuals = numpy.sum(deltas * deltas / sigma2)

        err = math.fabs(residuals) / numpy.sum(1.0 / sigma2)
        return [a, b], [0, math.sqrt(err)]
    else:
        A = numpy.vstack([linearized_data.x / linearized_data.dy, 1.0 / linearized_data.dy]).T
        (p, residuals, _, _) = numpy.linalg.lstsq(A, linearized_data.y / linearized_data.dy)

        # Get the covariance matrix, defined as inv_cov = a_transposed * a
        err = numpy.zeros(2)
        try:
            inv_cov = numpy.dot(A.transpose(), A)
            cov = numpy.linalg.pinv(inv_cov)
            err_matrix = math.fabs(residuals) * cov
            err = [math.sqrt(err_matrix[0][0]), math.sqrt(err_matrix[1][1])]
        except:
            err = [-1.0, -1.0]

        return p, err


def bayesian():
    NotImplemented


def dummy():
    print('\m/ >.< \m/')



if __name__ == '__main__':

    logging.debug('\m/ >.< \m/')