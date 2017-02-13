#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Steven C. Howell --<steven.howell@nist.gov>
  Purpose: clustering structure ensembles
  Created: 02/13/2017

0000000001111111111222222222233333333334444444444555555555566666666667777777777
1234567890123456789012345678901234567890123456789012345678901234567890123456789
"""
import glob
import hdbscan
import logging
import os

import sklearn.cluster
import sklearn.preprocessing
import scipy.spatial

import numpy as np
import pandas as pd

import sasmol.sasmol as sasmol
import sas_modeling

from builtins import range

logging.basicConfig(format=':', level=logging.DEBUG)


def find_data_files(run_dir, file_ext):

    file_search = os.path.join(run_dir, file_ext)
    run_files = glob.glob(file_search)
    run_files.sort()
    logging.info('found {} data files'.format(n_samples))

    return run_files


def main(run_dir, file_ext, pdb_fname, dcd_fname, rescale=False, dbscan=False):

    run_files = find_data_files(run_dir, file_ext)
    n_samples = len(run_files)

    if rescale:
        scale_string = 'scale'
    else:
        scale_string = 'raw'
    output_dir = os.path.join(run_dir, '{}_{}'.format(file_ext[-2:],
                                                      scale_string), '')
    sas_clustering.file_io.mkdir_p(output_dir)

    if file_ext[-2:] == 'iq':
        data = load_iq_data(run_files)

    elif file_ext[-2:] == 'pr':
        data = load_pr_data(run_files)

    else:
        logging.error('unknown file type (not iq or pr): {}'.format(file_ext))

    if rescale:
        data = rescale_data(data, output_dir)

    min_cluster_dist = calc_k_dist(data, output_dir)

    if dbscan:
        # DBSCAN
        distance = 1
        db = sklearn.cluster.DBSCAN(eps=min_cluster_dist, min_samples=2).fit(data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_ + 1 # 0's are independent groups

    else:
        # HDBSCAN
        db = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(data)
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


    # create a dcd for every cluster with >1 frame
    mol = sasmol.SasMol(0)
    mol.read_pdb(pdb_fname)

    dcd_fnames = []
    cluster_out_files = [] # dcds for clusters
    unique_out_fname = '{}{}_uniue.dcd'.format(out_dir, dcd_fname[:-4])
    dcd_out_file = mol.open_dcd_write(unique_out_fname) # dcd file for unique structures
    dcd_in_file = mol.open_dcd_read(dcd_fname)

    for i in range(len(unique)):
        dcd_fnames.append('{}_c{:02d}.dcd'.format(dcd_fname[:-4], i))
        cluster_out_files.append(mol.open_dcd_write(dcd_fnames[i]))

    visited_cluster = set()
    dcd_out_frame = 0
    cluster_out_frame = np.zeros(len(unique), dtype=int)

    for (i, label) in enumerate(labels):
        mol.read_dcd_step(dcd_in_file, i)
        if label == 0:
            dcd_out_frame += 1
            mol.write_dcd_step(dcd_out_file, 0, dcd_out_frame)
        else:
            cluster_out_frame[label-1] += 1
            mol.write_dcd_step(cluster_out_files[label-1], 0,
                               cluster_out_frame[label-1])
            if label not in visited_cluster:
                visited.add(label)
                dcd_out_frame += 1
                mol.write_dcd_step(dcd_out_file, 0, dcd_out_frame)

    for cluster_out_file in cluster_out_files:
        mol.close_dcd_write(cluster_out_file)

    mol.close_dcd_write(dcd_out_file)
    mol.close_dcd_read(dcd_in_file[0])

    return data


def calc_k_dist(data, output_dir):
    # get the k-dist data
    n_sampels = len(data)
    dist = np.zeros([n_samples, 2])
    dist[:, 0] = np.arange(n_samples)
    for i in dist[:, 0]:
        # iterating to save memory
        i_dist = cdist(data[i].reshape([1, -1]), data[dist[:, 0]!=i])
        dist[i, 1] = i_dist.min()

    np.savetxt('{}k_dist.dat'.format(output_dir), dist)

    min_cluster_dist = 1  # need to add calculation here

    return min_cluster_dist


def rescale_data(data, output_dir):
    # rescale the data so each point is weighted equally

    range_before = data.max(axis=0) - data.min(axis=0)
    np.savetxt('{}before_scaling.dat'.format(output_dir), range_before,
               delimiter=',', fmt='%f')

    data_scaler = sklearn.preprocessing.RobustScaler()
    data_scaler.fit(data)
    data = data_scaler(data)

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
    iq_data.append(first_data[1:, 1])  # do not use I(0), it is the same for every dataset

    # load in the rest of the data
    for saxs_file in saxs_files[1:]:
        x_data = np.loadtxt(saxs_file)
        x_data[:, 1] /= x_data[0, 1]
        x_data = x_data[q_mask]
        assert np.allclose(x_data[0, 1], first_data[0, 1]), 'ERROR: data not normalized to I(0)'
        assert np.allclose(x_data[:, 0], first_data[:, 0]), 'ERROR: data not on same Q-grid'
        iq_data.append(x_data[1:, 1])

    iq_data = np.array(iq_data)
    return iq_data


if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    run_dir = 'data/scratch/sas_clustering'

    iq_dir = 'sascalc/xray'
    iq_dir = os.path.join(home_dir, run_dir, iq_dir)
    iq_ext = '*.iq'
    main(iq_dir, iq_ext)

    pr_dir = 'pr'
    pr_dir = os.path.join(home_dir, run_dir, pr_dir)
    pr_ext = '*.pr'
    main(iq_dir, iq_ext, run_type='pr')

    logging.info('\m/ >.< \m/')