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


def interp_data(x_data, y_data, new_x_data):
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

    n_y_data = y_data.shape[1]
    n_x = len(new_x_data)
    new_y_data = np.empty([n_x, n_y_data], order='f')

    for i in range(n_y_data):
        interp_data = interpolate.splrep(x_data, y_data[:, i])
        new_y_data[:, i] = interpolate.splev(new_x_data, interp_data)

    return new_x_data, new_y_data


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

    sigma2 = rf_data[:, 2] * rf_data[:, 2]
    scale = ((rf_data[:, 1] * in_data / sigma2).sum() /
             (in_data * in_data / sigma2).sum())

    mt_data = scale * in_data

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

    sigma2 = rf_data[:, 2] * rf_data[:, 2]
    a = (rf_data[:, 1] / sigma2).sum()
    b = (in_data / sigma2).sum()
    c = (1 / sigma2).sum()
    offset = (a - b) / c

    mt_data = in_data + offset

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

    sigma2 = rf_data[:, 2] * rf_data[:, 2]
    a = (rf_data[:, 1] / sigma2).sum()
    b = (in_data / sigma2).sum()
    c = (1 / sigma2).sum()
    d = (rf_data[:, 1] * in_data / sigma2).sum()
    e = (in_data * in_data / sigma2).sum()

    offset = (a * e - b * d) / (c * e - b * b)
    scale = (c * d - b * a) / (c * e - b * b)

    mt_data = scale * in_data + offset

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

    # could implement weights by changeing the second column to be 1/error
    A = np.vstack([in_data, np.ones(len(in_data))]).T
    scale, offset = np.linalg.lstsq(A, rf_data[:, 1])[0]
    mt_data = scale * in_data + offset

    x2 = get_x2(rf_data, mt_data)

    return mt_data, scale, offset, x2


def get_x2(rf_data, md_data, dof=None):
    """
    determine the chi-square value between reference and model data

    Parameters
    ----------
    rf_data:
        reference data for comparing the model data (should be Nx3 np.array)
    md_data:
        model data to compare to the reference data (should be Nx2 np.array)

    Returns
    -------
    x2: chi-square value

    See also
    --------
    get_x2_components, get_r
    """

    x2, _ = get_x2_components(rf_data, md_data, dof=dof)

    return x2


def get_x2_components(rf_data, md_data, dof=None):
    """
    determine the chi-square value between reference and model data

    Parameters
    ----------
    rf_data:
        reference data for comparing the model data (should be Nx3 np.array)
    md_data:
        model data to compare to the reference data (should be a
        length N np.array)

    Returns
    -------
    x2:         chi-square value
    components: chi-square value components from each data point

    See also
    --------
    get_x2_components, get_r
    """

    diff = md_data - rf_data[:, 1]
    diff2 = diff * diff
    er2 = rf_data[:, 2] ** 2
    if not dof:
        dof = len(rf_data)
    components = (diff2 / er2) / dof
    x2 = components.sum()

    assert np.isfinite(x2), 'x2 not finite'

    return x2, components


def get_r(rf_data, md_data):
    """
    determine the chi-square value between reference and model data

    Parameters
    ----------
    rf_data:
        reference data for comparing the model data (should be Nx3 np.array)
    md_data:
        model data to compare to the reference data (should be Nx2 np.array)

    Returns
    -------
    r:          R value

    See also
    --------
    get_x2_components, get_r
    """
    r, _ = get_r_components(rf_data, md_data)

    return r


def get_r_components(rf_data, md_data):
    """
    determine the R value between a referenc and model data set,
    as defined by doi: 10.1042/bj2670203

    Parameters
    ----------
    rf_data:
        reference data for comparing the model data (should be Nx3 np.array)
    md_data:
        model data to compare to the reference data (should be a
        length N np.array)

    Returns
    -------
    r:          R value
    components: R value components from each data point

    See also
    --------
    get_x2_components, get_r
    """

    diff = np.abs(md_data - rf_data[:, 1])
    norm = np.abs(rf_data[:, 1]).sum()
    components = diff / norm
    r = components.sum()

    return r, components
