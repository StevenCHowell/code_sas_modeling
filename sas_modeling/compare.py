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
from scipy import interpolate
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def interp_data(data, x_grid):
    """
    interpolate data to a new x-grid

    Parameters
    ----------
    data:
        input data to interpolate (should be NxM np.array, where
        data[:, 0] is the original x-grid, and every subsequent
        column is data-set that should be interpolated)
    x_grid:
        new x-grid to interpolate to (should be np.array of length L)

    Returns
    -------
    new_data: interpolated data (np.array with dimensions LxM)
    """

    n_data = data.shape[1] - 1
    n_x = len(x_grid)
    new_data = np.empty([n_x, n_data+1], order='f')
    new_data[:, 0] = x_grid

    for i in range(n_data):
        interp_data = interpolate.splrep(data[:, 0], data[:, i])
        new_data[:, i] = interpolate.splev(x_grid, interp_data)

    return new_data


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


def get_x2(rf_data, mt_data, dof=None):
    x2, _ = get_x2_components(rf_data, mt_data, dof=dof)

    return x2


def get_x2_components(rf_data, mt_data, dof=None):
    diff = mt_data[:, 1] - rf_data[:, 1]
    diff2 = diff * diff
    er2 = rf_data[:, 2] * rf_data[:, 2]
    if not dof:
        dof = len(rf_data)
    components = (diff2 / er2) / dof
    x2 = components.sum()

    return x2, components


def get_r(rf_data, mt_data):
    r, _ = get_r_components(rf_data, mt_data)

    return r


def get_r_components(rf_data, mt_data):
    # R value as defined by doi: 10.1042/bj2670203
    diff = np.abs(mt_data[:, 1] - rf_data[:, 1])
    norm = np.abs(rf_data[:, 1]).sum()
    components = diff / norm
    r = components.sum()

    return r, components
