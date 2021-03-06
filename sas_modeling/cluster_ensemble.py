#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Steven C. Howell --<steven.howell@nist.gov>
  Purpose: clustering structure ensembles
  Created: 02/13/2017

0000000001111111111222222222233333333334444444444555555555566666666667777777777
1234567890123456789012345678901234567890123456789012345678901234567890123456789
"""
from __future__ import absolute_import, division, print_function

import glob
import hdbscan
import logging
import os

import sklearn.cluster
import sklearn.preprocessing
import scipy.spatial
import scipy.interpolate

import numpy as np
import pandas as pd

import sasmol.system as system
import sas_modeling

logging.basicConfig(format=':', level=logging.DEBUG)
np.set_printoptions(suppress=True)

try:
    range = xrange
except NameError:
    pass


def find_data_files(run_dir, file_ext):

    file_search = os.path.join(run_dir, file_ext)
    run_files = glob.glob(file_search)
    run_files.sort()
    logging.info('found {} data files'.format(len(run_files)))

    return run_files


def main(run_dir, file_ext, pdb_fname, dcd_fname, rescale=False, dbscan=False):

    run_files = find_data_files(run_dir, file_ext)
    n_samples = len(run_files)

    if rescale:
        scale_string = 'scale'
    else:
        scale_string = 'raw'
    if dbscan:
        type_string = 'dbscan'
    else:
        type_string = 'hdbscan'
    output_dir = os.path.join(run_dir, '{}_{}_{}'.format(
        file_ext[-2:], scale_string, type_string), '')
    sas_modeling.file_io.mkdir_p(output_dir)

    if file_ext[-2:] == 'iq':
        data = load_iq_data(run_files)

    elif file_ext[-2:] == 'pr':
        data = load_pr_data(run_files)

    else:
        logging.error('unknown file type (not iq or pr): {}'.format(file_ext))

    if rescale:
        data = rescale_data(data, output_dir)

    min_cluster_dist = calc_k_dist(data, output_dir)

    labels = cluster_data(data, output_dir, dbscan, min_cluster_dist)

    create_cluster_dcds(labels, pdb_fname, dcd_fname, output_dir)

    logging.info('completed run, results save to {}'.format(output_dir))


def cluster_data(data, output_dir, dbscan, min_cluster_dist):
    if dbscan:
        # DBSCAN
        distance = 1
        scan = sklearn.cluster.DBSCAN(eps=min_cluster_dist, min_samples=2)
        db = scan.fit(data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_ + 1 # 0's are independent groups

    else:
        # HDBSCAN
        scan = hdbscan.HDBSCAN(min_cluster_size=2)
        db = scan.fit_predict(data)
        labels = db + 1

    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    np.savetxt('{}clusters.dat'.format(output_dir), labels, fmt='%d')

    # info
    unique = set(labels)
    unique.remove(0)
    logging.debug('cluster labels: {}'.format(unique))
    logging.debug('unique clusters: {}'.format(
        len(unique) + list(labels).count(0)))
    for c in set(labels):
        logging.debug('{}: {}'.format(c+1, list(labels).count(c)))
    return labels


def create_cluster_dcds(labels, pdb_fname, dcd_fname, output_dir):
    assert os.path.exists(pdb_fname), 'no such file: {}'.format(pdb_fname)
    assert os.path.exists(dcd_fname), 'no such file: {}'.format(dcd_fname)

    # create a dcd for every cluster with >1 frame
    mol = system.Molecule(pdb_fname)

    dcd_basename = os.path.basename(dcd_fname)[:-4]

    # create a dcd with one structure from each cluster
    unique_out_fname = '{}{}_unique.dcd'.format(output_dir, dcd_basename)
    dcd_out_file = mol.open_dcd_write(unique_out_fname)
    dcd_in_file = mol.open_dcd_read(dcd_fname)
    visited_cluster = set()
    dcd_out_frame = 0
    for (i, label) in enumerate(labels):
        mol.read_dcd_step(dcd_in_file, i)
        if label == 0:
            dcd_out_frame += 1
            mol.write_dcd_step(dcd_out_file, 0, dcd_out_frame)
        elif label not in visited_cluster:
            visited_cluster.add(label)
            dcd_out_frame += 1
            mol.write_dcd_step(dcd_out_file, 0, dcd_out_frame)
    mol.close_dcd_write(dcd_out_file)
    mol.close_dcd_read(dcd_in_file[0])

    # create a dcd for each cluster
    n_max = 500
    i_lists = []
    j = 0
    unique = set(labels)
    remove_0 = False  # consider using this option
    if remove_0:
        unique.remove(0)
        for i in range(np.floor(len(unique)/n_max).astype(int)):
            i_lists.append(list(range(n_max * j + 1, n_max * (j + 1) + 1)))
            j += 1  # required for situation when len(unique) < n_max
        i_lists.append(list(range(n_max * j + 1, len(unique) + 1)))
    else:
        for i in range(np.floor(len(unique)/n_max).astype(int)):
            i_lists.append(list(range(n_max * j, n_max * (j + 1))))
            j += 1  # required for situation when len(unique) < n_max
        i_lists.append(list(range(n_max * j, len(unique))))

    n_written = 0
    for i_list in i_lists:
        dcd_fnames = []
        cluster_out_files = [] # dcds for clusters
        cluster_out_frame = np.zeros(n_max, dtype=int)
        for i in i_list:
            dcd_fnames.append('{}{}_c{:06d}.dcd'.format(output_dir,
                                                        dcd_basename, i))
            cluster_out_files.append(mol.open_dcd_write(dcd_fnames[i%n_max]))

        dcd_in_file = mol.open_dcd_read(dcd_fname)
        for (i, label) in enumerate(labels):
            mol.read_dcd_step(dcd_in_file, i)
            if label in i_list:
                mol.write_dcd_step(cluster_out_files[label%n_max], 0,
                                   cluster_out_frame[label%n_max])
                # cluster_out_frame[label%n_max] += 1  # index for writing output
                n_written += 1

        mol.close_dcd_read(dcd_in_file[0])

        for cluster_out_file in cluster_out_files:
            mol.close_dcd_write(cluster_out_file)


def calc_k_dist(data, output_dir):
    # get the k-dist data
    n_samples = len(data)
    dist = np.zeros([n_samples, 2])
    dist[:, 0] = np.arange(n_samples)
    for i in dist[:, 0].astype(np.int):
        # iterating to save memory
        i_dist = scipy.spatial.distance.cdist(data[i].reshape([1, -1]),
                                              data[dist[:, 0]!=i])
        dist[i, 1] = i_dist.min()

    dist[:, 1].sort()
    np.savetxt('{}k_dist.dat'.format(output_dir), dist)


    # smoothing spline
    spline = scipy.interpolate.UnivariateSpline(dist[:, 0], dist[:, 1], k=4)
    dist_smooth = np.copy(dist)
    dist_smooth[:, 1] = spline(dist_smooth[:, 0])

    # normalize the points to the unit square (0, 0) to (1, 1)
    dist_norm = dist_smooth - dist_smooth.min(axis=0)
    dist_norm /= dist_norm.max(axis=0) - dist_norm.min(axis=0)

    # calculate dd = (x, y-x)
    dist_d = np.copy(dist_norm)
    dist_d[:, 1] -= dist_d[:, 0]

    # find the extremum (minimum in this case)
    i_min = dist_d[:, 1].argmin()

    min_cluster_dist = dist[i_min, 1]

    debug = np.vstack([dist[:, 0], dist[:, 1], dist_smooth[:, 1],
                       dist_norm[:, 1], dist_d[:, 1]])
    k_dist_out =  '{}k_dist.dat'.format(output_dir)
    np.savetxt(k_dist_out,  debug.T, fmt='%f')

    if True:
        import bokeh.plotting
        from bokeh.palettes import Colorblind7 as palette
        import bokeh.models

        dat = np.loadtxt(k_dist_out)
        i_min = dat[:, 4].argmin()

        bokeh.plotting.output_file('{}smooth.html'.format(output_dir))
        p = bokeh.plotting.figure(y_axis_label='distance')
        p.line(dat[:, 0], dat[:, 1], legend='raw', color=palette[0])
        p.line(dat[:, 0], dat[:, 2], legend='smooth', line_color=palette[1])

        # v2 = bokeh.models.Span(location=dat[i2, 0], dimension='height',
                               # line_dash='dashed', line_color='orange')
        # p.add_layout(v2)
        # v3 = bokeh.models.Span(location=dat[i3, 0], dimension='height',
                               # line_dash='dashed', line_color='red')
        # p.add_layout(v3)
        # v4 = bokeh.models.Span(location=dat[i4, 0], dimension='height',
                               # line_dash='dashed', line_color='violet')
        # p.add_layout(v4)
        # v5 = bokeh.models.Span(location=dat[i5, 0], dimension='height',
                               # line_dash='dashed', line_color='blue')
        # p.add_layout(v5)
        vline = bokeh.models.Span(location=dat[i_min, 0], dimension='height',
                                  line_dash='dashed', line_color=palette[-1])
        p.add_layout(vline)
        bokeh.plotting.show(p)

        bokeh.plotting.output_file('{}norm.html'.format(output_dir))
        p = bokeh.plotting.figure(y_axis_label='distance',
                                  title='max dist = {}'.format(dat[i_min, 1]))
        p.line(dat[:, 0]/dat[-1, 0], dat[:, 3], legend='norm', color=palette[0])
        p.line(dat[:, 0]/dat[-1, 0], dat[:, 0]/dat[-1, 0], color='black')
        p.line(dat[:, 0]/dat[-1, 0], dat[:, 4], legend='Dd', color=palette[1])
        p.line(dat[:, 0]/dat[-1, 0], np.zeros_like(dat[:, 0]), color='red')
        p.legend.location = 'top_left'
        vline = bokeh.models.Span(location=dat[i_min, 0]/dat[-1, 0],
                                  dimension='height', line_dash='dashed',
                                  line_color=palette[-1])

        p.add_layout(vline)
        bokeh.plotting.show(p)

    return min_cluster_dist


def rescale_data(data, output_dir):
    # rescale the data so each point is weighted equally

    range_before = data.max(axis=0) - data.min(axis=0)
    np.savetxt('{}before_scaling.dat'.format(output_dir), range_before,
               delimiter=',', fmt='%f')

    data_scaler = sklearn.preprocessing.RobustScaler()
    data = data_scaler.fit_transform(data)

    range_after = data.max(axis=0) - data.min(axis=0)
    np.savetxt('{}after_scaling.dat'.format(output_dir), range_after,
               delimiter=',', fmt='%f')

    return data


def load_pr_data(pr_files):

    pr_data_l = []
    n_pr = len(pr_files)
    n_r = np.empty(n_pr, dtype=int)

    # load in all the data set
    for i, pr_file in enumerate(pr_files):
        n_data = np.loadtxt(pr_file, delimiter=',', dtype=int)
        pr_data_l.append(n_data[:, 1])
        n_r[i] = len(n_data)

    # pad the data with zeros so each data is the same length
    r_max = n_r.max()
    pr_data = np.zeros([n_pr, r_max], dtype=int)
    for i, n_data in enumerate(pr_data_l):
        pr_data[i, :len(n_data)] = n_data

    return pr_data


def load_iq_data(saxs_files):
    iq_data = []

    # load in the first data set to setup the q-mask
    first_data = np.loadtxt(saxs_files[0])
    q_mask = first_data[:, 0] <= 0.18  # only use data up to 0.18 1/A
    first_data[:, 1] /= first_data[0, 1]  # normalize I(0), to 1
    first_data = first_data[q_mask]
    iq_data.append(first_data[1:, 1])  # skip I(0), same for every dataset

    # load in the rest of the data
    for saxs_file in saxs_files[1:]:
        x_data = np.loadtxt(saxs_file)
        x_data[:, 1] /= x_data[0, 1]
        x_data = x_data[q_mask]
        assert np.allclose(x_data[:, 0], first_data[:, 0]
                           ), 'inconsistent Q-grid'
        iq_data.append(x_data[1:, 1])

    iq_data = np.array(iq_data)
    return iq_data


def analyze_clustering(cluster_dir, file_dir, file_ext, pdb_fname):
    '''
    evaluate the percent difference between clustered data
    '''
    run_files = find_data_files(file_dir, file_ext)
    # n_samples = len(run_files)

    cluster_fname = os.path.join(cluster_dir, 'clusters.dat')
    clusters = np.loadtxt(cluster_fname, dtype=np.int)

    df = pd.DataFrame({'fname': run_files, 'cluster': clusters})

    if file_ext[-2:] == 'iq':
        data = load_iq_data(run_files)

    elif file_ext[-2:] == 'pr':
        data = load_pr_data(run_files)

    unique_clusters = set(clusters)
    unique_clusters.remove(0)
    n_clusters = len(unique_clusters)

    max_perc_diff = np.empty((n_clusters, 2))
    cluster_perc_diff = []
    for k, cluster_id in enumerate(unique_clusters):
        c_data = data[df[df['cluster'] == cluster_id].index]
        n = len(c_data)
        frac_diff = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                frac_diff[i, j] = get_frac_diff(c_data[i], c_data[j])
        cluster_perc_diff.append(frac_diff)
        max_perc_diff[k] = [cluster_id, frac_diff.max()]

    out_fname = os.path.join(cluster_dir, 'max_perc_diff.dat')
    np.savetxt(out_fname, max_perc_diff, fmt=['%d', '%f'], header='id, max percent difference')


def get_frac_diff(data1, data2):
    r'''
    calculate the fractional difference between two data sets

    Parameters
    ----------
    data1: array_like
        first data set to use in comparison
        can be a 1D array of length N or a 2D array of size MxN
    data2: array_like
        second data set to use in comparison
        can be a 1D array of length N or a 2D array of size MxN

    Returns
    -------
    frac_diff
        the fractional difference between data1 and data2

    Examples
    --------
    Written to handle 3 possible combinations of data size
    1. data1 and data2 are both 1D arrays
    >>> data1 = np.arange(10)
    >>> data2 = np.arange(-9, 1)[::-1]
    >>> get_frac_diff(data1, data2)
    array([ 9.])

    # 2. one of data1 or data2 is 1D and the other is 2D
    >>> data2 = np.eye(10)[:3]
    >>> get_frac_diff(data1, data2)
    array([ 2.        ,  1.6       ,  1.66666667])

    # 3. both of data1 and data2 are 2D
    >>> data1 = np.ones((3, 10))
    >>> get_frac_diff(data1, data2)
    array([ 1.8,  1.8,  1.8])

    '''
    if len(data1.shape) == 1:
        data1 = data1.reshape(1, -1)
    if len(data2.shape) == 1:
        data2 = data2.reshape(1, -1)

    data_mean = (data1 + data2) / 2.0
    zero_mask = np.abs(data_mean) == 0
    data_mean[zero_mask] = 1  # iff d1+d2=0, diff=d1-d2=0, so diff/1=0
    axis = len(data_mean.shape) - 1  # this accomodates 1D or 2D arrays
    norm_diff = np.abs(data1 - data2) / data_mean
    frac_diff = np.mean(norm_diff, axis=axis)

    return frac_diff


def get_euclid_dist(data1, data2):
    r'''
    calculate the euclidiean distance between two data sets

    Parameters
    ----------
    data1: array_like
        first data set to use in comparison
        can be a 1D array of length N or a 2D array of size MxN
    data2: array_like
        second data set to use in comparison
        can be a 1D array of length N or a 2D array of size MxN

    Returns
    -------
    euclid_dist
        the Euclidean distance between data1 and data2

    Examples
    --------
    Written to handle 3 possible combinations of data size
    1. data1 and data2 are both 1D arrays
    >>> data1 = np.arange(10)
    >>> data2 = np.arange(-9, 1)[::-1]
    >>> get_euclid_dist(data1, data2)
    array([ 33.76388603])

    # 2. one of data1 or data2 is 1D and the other is 2D
    >>> data2 = np.eye(10)[:3]
    >>> get_euclid_dist(data1, data2)
    array([ 16.91153453,  16.85229955,  16.79285562])

    # 3. both of data1 and data2 are 2D
    >>> data1 = np.ones((3, 10))
    >>> get_euclid_dist(data1, data2)
    array([ 3.,  3.,  3.])

    '''

    one_d = False
    if len(data1.shape) == 1:
        data1 = data1.reshape(1, -1)
        one_d = True
    if len(data2.shape) == 1:
        data2 = data2.reshape(1, -1)
        one_d = True

    if one_d:
        euclid_dist = scipy.spatial.distance.cdist(data1, data2)

    else:
        n1 = len(data1)
        n2 = len(data2)
        assert n1 == n2, 'Inconsistent input size'
        euclid_dist = np.empty((1, n1))
        for i in range(n1):
            dist = scipy.spatial.distance.cdist(data1[i:i+1], data2[i:i+1])
            euclid_dist[0, i] = dist

    return euclid_dist[0]


def test(d1, d2):
    return np.mean(np.abs(d1 - d2) / d1)


if __name__ == '__main__':
    import doctest
    doctest.testmod()


    # home_dir = os.path.expanduser("~")
    # run_dir = 'data/scratch/sas_clustering'
    # pdb_fname = os.path.join(home_dir, run_dir, 'centered_mab.pdb')

    # analyze = True
    # if analyze:

        # iq_dir = 'sascalc/xray'
        # pr_dir = 'pr'
        # file_ext = '*.pr'

        # cluster_dirs = [
            # 'pr_raw_dbscan',
            # 'pr_raw_hdbscan',
            # 'pr_scale_dbscan',
            # 'pr_scale_hdbscan',
        # ]

        # for cluster_dir in cluster_dirs:
            # cluster_dir = os.path.join(home_dir, run_dir, pr_dir, cluster_dir)
            # file_dir = os.path.join(home_dir, run_dir, pr_dir)

            # analyze_clustering(cluster_dir, file_dir, file_ext, pdb_fname)

    # test = False
    # if test:
        # dcd_fname = os.path.join(home_dir, run_dir, 'to_test2.dcd')

        # iq_dir = 'sascalc/xray'
        # iq_dir = os.path.join(home_dir, run_dir, iq_dir)
        # iq_ext = '*.iq'
        # main(iq_dir, iq_ext, pdb_fname, dcd_fname)
        # main(iq_dir, iq_ext, pdb_fname, dcd_fname, rescale=True)
        # main(iq_dir, iq_ext, pdb_fname, dcd_fname, dbscan=True)
        # main(iq_dir, iq_ext, pdb_fname, dcd_fname, rescale=True, dbscan=True)

        # pr_dir = 'pr'
        # pr_dir = os.path.join(home_dir, run_dir, pr_dir)
        # pr_ext = '*.pr'
        # main(pr_dir, pr_ext, pdb_fname, dcd_fname)
        # main(pr_dir, pr_ext, pdb_fname, dcd_fname, rescale=True)
        # main(pr_dir, pr_ext, pdb_fname, dcd_fname, dbscan=True)
        # main(pr_dir, pr_ext, pdb_fname, dcd_fname, rescale=True, dbscan=True)

    # logging.info('\m/ >.< \m/')