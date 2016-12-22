#!/usr/bin/env python
#coding:utf-8
"""
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: calculating the R-factor from a SasCalc run
    Created: 12/09/2016

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
"""

import glob
import logging
import os
import pandas as pd
import numpy as np

from scipy.optimize import minimize

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def get_r(iq_data, iq_goal):
    r, _ = get_r_components(iq_data, iq_goal)

    return r


def get_r_components(iq_data, iq_goal):
    # R value as defined by doi: 10.1042/bj2670203
    diff = np.abs(iq_goal[:, 1] - iq_data[:, 1])
    norm = np.abs(iq_data[:, 1]).sum()
    components = diff / norm
    r = components.sum()

    return r, components


def sascalc_r_factors(run_dir, iq_goal, ext='*.iq', run_name=''):
    r_fname = 'r_factors.txt'
    if run_name:
        r_fname = '{}_{}'.format(run_name, r_fname)
    r_full_fname = os.path.join(run_dir, r_fname)

    file_search = os.path.join(run_dir, ext)
    iq_fnames = glob.glob(file_search)
    assert iq_fnames, 'ERROR: no files found using {}'.format(file_search)
    iq_fnames.sort()

    # verify the q-grids match
    q_grid = np.loadtxt(iq_fnames[0])[:, 0]
    if len(iq_goal) != len(q_grid):
        logging.info('rebinning experimental data to scattering q-grid')
        assert np.isclose(iq_goal[0, 0], 0), 'I(0) required'
        iq_goal = rebin(iq_goal, q_grid, i0=iq_goal[0, 1],
                        er_i0=iq_goal[0, 2])
        good_rows = iq_goal[:, 1] > 0
        if sum(good_rows) < len(good_rows):
            logging.warning('some data is missing, removing empty rows')
        iq_goal = iq_goal[good_rows]
        q_grid = q_grid[good_rows]
    if not np.allclose(iq_goal[:, 0], q_grid):
        raise ValueError('mismatched q-grid')

    if not os.path.exists(r_full_fname):
        index = []
        r_factors = []
        scale_values = []
        for fname in iq_fnames:
            iq_data = np.asfortranarray(np.loadtxt(fname))
            iq_data = iq_data[good_rows]
            if False:
                r, scale = min_r(iq_data, iq_goal)
            else:
                scale = iq_goal[1, 1] / iq_data[1, 1]
                iq_data *= scale
                r = get_r(iq_data, iq_goal)

            index.append(int(fname.replace(ext[1:], '')[-5:]))
            r_factors.append(r)
            scale_values.append(scale)

        if False:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots()
            ax1.plot(index, r_factors, 'b')
            ax1.set_ylabel('R-factor')
            for tick in ax1.get_yticklabels():
                tick.set_color('b')
            ax2 = ax1.twinx()
            ax2.plot(index, scale_values, 'g')
            ax2.set_ylabel('scale')
            for tick in ax2.get_yticklabels():
                tick.set_color('g')

        data = {'r': r_factors, 'scale': scale_values, 'fnames': iq_fnames}
        r_results = pd.DataFrame(data=data, index=index)
        r_results.to_csv(r_full_fname, sep='\t', float_format='%5.10f',
                         columns=['r', 'scale', 'fnames'])
    else:
        logging.info('output already exists, remove using:\nrm {0}'.format(
            r_full_fname))
        r_results = pd.DataFrame.from_csv(r_full_fname, sep='\t')

    return r_results, iq_goal


def get_p(iq_data, iq_goal):
    p, _ = get_p_components(iq_data, iq_goal)

    return p


def get_p_components(iq_data, iq_goal):
    # percent error
    diff = iq_goal[:, 1] - iq_data[:, 1]
    norm = iq_data[:, 1]
    components = np.abs(diff / norm) * 100 / len(norm)
    p = components.sum()

    return p, components


def sascalc_percent_error(run_dir, iq_goal, ext='*.iq', run_name=''):
    p_fname = 'percent_error.txt'
    if run_name:
        p_fname = '{}_{}'.format(run_name, p_fname)
    p_full_fname = os.path.join(run_dir, p_fname)

    if not os.path.exists(p_full_fname):

        file_search = os.path.join(run_dir, ext)
        iq_fnames = glob.glob(file_search)
        assert iq_fnames, 'ERROR: no files found using {}'.format(file_search)
        iq_fnames.sort()

        index = []
        per_error = []
        for fname in iq_fnames:
            iq_data = np.asfortranarray(np.loadtxt(fname))  # load data
            iq_data[:, 1] *= iq_goal[0, 1] / iq_data[0, 1]  # rescale data
            p = get_p(iq_goal, iq_data)

            per_error.append(p)
            index.append(int(fname.replace(ext[1:], '')[-5:]))

        data = {'p': per_error, 'fnames': iq_fnames}
        p_error = pd.DataFrame(data=data, index=index)
        p_error.to_csv(p_full_fname, sep='\t', float_format='%5.10f',
                         columns=['p', 'fnames'])
    else:
        logging.info('output already exists, remove using:\nrm {0}'.format(
            p_full_fname))
        p_error = pd.DataFrame.from_csv(p_full_fname, sep='\t')

    return p_error


def scale_r(scale, iq_data, iq_goal):
    iq_tmp = np.copy(iq_data)
    iq_tmp[:, 1] *= scale
    r = get_r(iq_tmp, iq_goal)
    logging.debug('scale: {}\tr: {}'.format(scale, r))
    return r


def offset_r(iq_data, iq_goal, offset):
    iq_tmp = np.copy(iq_data)
    iq_tmp[:, 1] += offset
    return get_r(iq_data, iq_goal)


def scale_offset_r(iq_data, iq_goal, offset):
    iq_tmp = np.copy(iq_data)
    iq_data[:, 1] *= scale
    iq_data[:, 1] += offset
    return get_r(iq_goal, iq_data)


def min_r(iq_data, iq_goal, scale=True, offset=False):
    assert np.allclose(iq_data[:, 0], iq_goal[:, 0]), 'data not on same q-grid'

    if offset:
        if scale:
            NotImplemented
            return scale, offset

        else:
            NotImplemented
            return offset

    else:
        scale_range = iq_goal[:, 1] / iq_data[:, 1]
        if False:
            import matplotlib.pyplot as plt
            ax1 = plt.subplot(121)
            ax1.plot(iq_goal[:, 0], iq_goal[:, 1], 'r--', label='exp')
            ax2 = plt.subplot(122)
            ax2.loglog(iq_goal[:, 0], iq_goal[:, 1], 'r--', label='exp')
            for scale in scale_range:
                ax1.plot(iq_data[:, 0], scale*iq_data[:, 1],
                         label='s: {}'.format(scale))
                ax2.loglog(iq_data[:, 0], scale*iq_data[:, 1],
                         label='s: {}'.format(scale))

            # manual scan
            scale_vals = np.linspace(min(scale_range), max(scale_range), 1000)
            scale_vals = np.linspace(0.95, 1.03, 1000)
            r_vals = []
            for scale in scale_vals:
                r_vals.append(scale_r(scale, iq_data, iq_goal))
            plt.figure()
            plt.semilogy(scale_vals, r_vals, 'o')
            plt.xlabel('scale')
            plt.ylabel('R-factor')

        initial_guess = iq_goal[0, 1] / iq_data[0, 1]
        res = minimize(scale_r, initial_guess, args=(iq_data, iq_goal),
                       bounds=[(min(scale_range), max(scale_range))])

        return res.fun, res.x[0]


def rebin(data, q_grid, i0=None, er_i0=None):
    '''
    rebin data with error to a specified q-grid
    follows white paper hosted here:
    http://isi.ssl.berkeley.edu/~tatebe/whitepapers/Combining%20Errors.pdf

    this is based on the following 2 assumptions regarding the number
    of pixels binned into the input data points, N
        1. N is roughly equal between neighboring point
        2. N is very large
    '''

    # verify the new q_grid is evenly spaced
    dq = q_grid[1:] - q_grid[:-1]
    if np.allclose(dq, dq[0], atol=1e-5):
        dq = dq[0]
        logging.info('rebinning to evenly spaced grid, dq = {}'.format(dq))
    else:
        raise ValueError('unevenly spaced grid not supported, dq-values: {}'.format(dq))

    if dq < data[0, 0]:
        logging.warning('some q bins will not have any data (q < {}), proceed with caution'.format(data[0, 0]))
        print('some q bins will not have any data (q < {}), proceed with caution'.format(data[0, 0]))

    # create array for rebinned data
    new_data = np.zeros((len(q_grid), 3))
    new_data[:, 0] = q_grid

    # rebin the data to the new q_grid
    bins = np.append(q_grid - dq/2, q_grid[-1] + dq/2)
    inds = np.digitize(data[:, 0], bins)

    for ind in set(inds):
        # get the average scattering intensity
        new_data[ind-1, 1] = np.mean(data[inds==ind, 1])

        # calculate the error in quadrature, based on unknown bin counts
        new_data[ind-1, 2] = np.sqrt(np.mean(data[inds==ind, 2] ** 2))

    # populate the q=0 scattering intensty
    if i0:
        new_data[0, 1] = i0
        if er_i0:
            new_data[0, 2] = er_i0
        else:
            logging.warning('I(0) error unknown, using 0.01 * I(0)')
            new_data[0, 2] = i0 / 100.0
    else:
        logging.warning('I(0) unknown, proceed with caution')
        # new_data = new_data[1:, :]

    return new_data


def rebin_edges(data, bins, i0=None, er_i0=None):
    '''
    ### Still need to test the value of setting the bin edges vs q-grid
    rebin data with error using specific bin edges
    follows white paper hosted here:
    http://isi.ssl.berkeley.edu/~tatebe/whitepapers/Combining%20Errors.pdf

    this is based on the following 2 assumptions regarding the number
    of pixels binned into the input data points, N
        1. N is roughly equal between neighboring point
        2. N is very large
    '''

    # verify the new q_grid is evenly spaced
    dq = bins[1:] - bins[:-1]
    if np.alltrue(np.isclose(dq - dq[0], 0)):
        dq = dq[0]
        logging.info('rebinning to evenly spaced grid, dq = {}'.format(dq))
    else:
        raise ValueError('unevenly spaced grid not yet supported')
    q_grid = bins[:-1] + dq / 2

    if q_grid[0] < data[0, 0]:
        logging.warning('some q bins will not have any data (q < {}), proceed with caution'.format(data[0, 0]))
        print('some q bins will not have any data (q < {}), proceed with caution'.format(data[0, 0]))

    # create array for rebinned data
    new_data = np.empty((len(q_grid), 3))
    new_data[:, 0] = q_grid

    # rebin the data to the new q_grid
    inds = np.digitize(data[:, 0], bins)

    for ind in set(inds):
        # get the average scattering intensity
        new_data[ind-1, 1] = np.mean(data[inds==ind, 1])

        # calculate the error in quadrature, based on unknown bin counts
        new_data[ind-1, 2] = np.sqrt(np.mean(data[inds==ind, 2] ** 2))

    # populate the q=0 scattering intensty
    if i0:
        new_data[0, 1] = i0
        if er_i0:
            new_data[0, 2] = er_i0
        else:
            logging.warning('I(0) error unknown, using 0.01 * I(0)')
            new_data[0, 2] = i0 / 100.0
    else:
        logging.warning('I(0) unknown, proceed with caution')
        # new_data = new_data[1:, :]

    return new_data


if __name__ == '__main__':

    # goal_fname = 'data/lysozyme_00001.iq'
    # run_name = 'match1_6'

    # goal_fname = 'data/exp_lys/rebinned_12_exp_data_lysozyme.dat'
    # run_name = 'match1_12'
    # run_dir = 'lys_ellipsoid_scan1/sascalc_12/neutron_D2Op_100/'

    run_dir = 'lys_ellipsoid_scan1/sascalc_61/neutron_D2Op_100/'
    goal_fname = 'data/exp_lys/norm_1mgml.dat'
    run_name = 'debug61'

    assert os.path.exists(run_dir), 'ERROR, bad path: {}'.format(run_dir)
    assert os.path.exists(goal_fname), 'ERROR, bad fname: {}'.format(goal_fname)
    goal_data = np.asfortranarray(np.loadtxt(goal_fname)[:, :3])
    i0 = np.array([0, 0.0112, 0.0112 / 100])
    goal_data = np.vstack((i0, goal_data))

    r_factors = sascalc_r_factors(run_dir, goal_data, run_name=run_name)
    # p_errors = sascalc_percent_error(run_dir, goal_data, run_name='min_12')

    logging.debug('\m/ >.< \m/')