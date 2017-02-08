#!/usr/bin/env python
#coding:utf-8
'''
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: functions useful for comparing model data to experimental data
    Created: 02/03/2017
    Origin:  https://github.com/StevenCHowell/sassie_0_dna/
                     blob/master/util/sassie_fits.py

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
'''

import numpy as np
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def scale(in_data, rf_data):
    """
    determine the scale to match the input data to the reference
    data by minimizing the x2 calculation (critical that the
    error estimates are reasonable for all Q values)

    Parameters
    ----------
    in_data:
        input data to match to the rf_data (should be Nx2 np.array)
    rf_data:
        reference data for matching the in_data (should be Nx3 np.array)

    Returns
    -------
    mt_data: version of in_data matched to the reference data
    scale:   scale factor applied to the input data
    x2:      X^2 comparison between the reference data and matched input data

    See also
    --------
    match_poly, match_lstsq, scale_offset

    """
    assert (in_data[:, 0] - rf_data[:, 0]).sum() == 0, (
        'mismatch between input and reference x-grid')

    sigma2 = rf_data[:, 2] * rf_data[:, 2]
    scale = ((rf_data[:, 1] * in_data[:, 1] / sigma2).sum() /
             (in_data[:, 1] * in_data[:, 1] / sigma2).sum())

    mt_data = np.vstack([in_data[:, 0], scale * in_data[:, 1]]).T

    x2 = get_x2(rf_data, mt_data)

    return mt_data, scale, x2


def offset(in_data, rf_data):
    """
    determine the offset to match the input data to the reference
    data by minimizing the x2 calculation

    Parameters
    ----------
    in_data:
        input data to match to the rf_data (should be Nx2 np.array)
    rf_data:
        reference data for matching the in_data (should be Nx3 np.array)

    Returns
    -------
    mt_data: version of in_data matched to the reference data
    offset:   offset applied to the input data
    x2:      X^2 comparison between the reference data and matched input data

    See also
    --------
    match_poly, match_lstsq, scale_offset, scale

    """
    assert (in_data[:, 0] - rf_data[:, 0]).sum() == 0, (
        'mismatch between input and reference x-grid')

    sigma2 = rf_data[:, 2] * rf_data[:, 2]
    a = (rf_data[:, 1] / sigma2).sum()
    b = (in_data[:, 1] / sigma2).sum()
    c = (1 / sigma2).sum()
    offset = (a - b) / c

    mt_data = np.vstack([in_data[:, 0], in_data[:, 1] + offset]).T

    x2 = get_x2(rf_data, mt_data)

    return mt_data, offset, x2


def scale_offset(in_data, rf_data):
    """
    determine the scale and offset to match the input data to the reference
    data by minimizing the reduced chi-square statistic
    \chi^2 = 1/(N_q-1) \sum_{i}^{N_q} [cI_{s_i} + I_c - I_{e_i}]^2 / \sigma_i^2

    Parameters
    ----------
    in_data:
        input data to match to the rf_data (should be Nx2 np.array)
    rf_data:
        reference data for matching the in_data (should be Nx3 np.array)

    Returns
    -------
    mt_data: version of in_data matched to the reference data
    scale:   scale factor applied to the input data
    offset:  offset applied to the input data
    x2:      X^2 comparison between the reference data and matched input data

    See also
    --------
    match_poly, match_lstsq, scale

    """
    small = 1E-4  # small parameter
    assert np.allclose((in_data[:, 0] - rf_data[:, 0]).sum(), 0, atol=small), (
        'mismatch between input and reference x-grid')

    sigma2 = rf_data[:, 2] * rf_data[:, 2]
    a = (rf_data[:, 1] / sigma2).sum()
    b = (in_data[:, 1] / sigma2).sum()
    c = (1 / sigma2).sum()
    d = (rf_data[:, 1] * in_data[:, 1] / sigma2).sum()
    e = (in_data[:, 1] * in_data[:, 1] / sigma2).sum()

    offset = (a * e - b * d) / (c * e - b * b)
    scale = (c * d - b * a) / (c * e - b * b)

    mt_data = np.vstack([in_data[:, 0], scale * in_data[:, 1] + offset]).T

    x2 = get_x2(rf_data, mt_data)

    return mt_data, scale, offset, x2


def match_lstsq(in_data, rf_data):
    """
    determine the scale and offset to match the input data to the reference
    data using a lstsq fit

    Parameters
    ----------
    in_data:
        input data to match to the rf_data (should be Nx2 np.array)
    rf_data:
        reference data for matching the in_data (should be Nx2 np.array)

    Returns
    -------
    mt_data: version of in_data matched to the reference data
    scale:   scale factor applied to the input data
    offset:  offset applied to the input data

    Notes
    --------
    does not use error bars to weight the data

    See also
    --------
    match_poly, scale_offset, scale

    """
    assert (in_data[:, 0] - rf_data[:, 0]).sum() == 0, (
        'mismatch between input and reference x-grid')

    # could implement weights by changeing the second column to be 1/error
    A = np.vstack([in_data[:, 1], np.ones(len(in_data))]).T
    scale, offset = np.linalg.lstsq(A, rf_data[:, 1])[0]
    mt_data = np.vstack([in_data[:, 0], scale * in_data[:, 1] + offset]).T

    x2 = get_x2(rf_data, mt_data)

    return mt_data, scale, offset, x2
