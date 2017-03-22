#!/usr/bin/env python
# coding:utf-8
'''
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: calculate the pair distance distribution, P(r)
    Created: 02/07/2017

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
'''
from __future__ import absolute_import, division, print_function

import errno
import glob
import logging
import os
import time

import numpy as np
from scipy.spatial.distance import pdist

from sasmol import system  # https://github.com/madscatt/sasmol

import numba

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def main(pdb_fname, dcd_fname='', in_dir='', out_dir='', restart=False):
    '''
    calculate the pair-distance distribution
    '''
    full_pdb_fname = os.path.join(in_dir, pdb_fname)
    assert os.path.exists(full_pdb_fname), 'No such file: {}'.format(
        full_pdb_fname)
    mol = system.Molecule(full_pdb_fname)

    if dcd_fname:
        full_dcd_fname = os.path.join(in_dir, dcd_fname)
        assert os.path.exists(full_dcd_fname), 'No such file: {}'.format(
            full_dcd_fname)
        dcd_file = mol.open_dcd_read(full_dcd_fname)
        n_frames = dcd_file[2]
        out_prefix = dcd_fname[:-4]

    else:
        n_frames = 1
        out_prefix = pdb_fname[:-4]

    if out_dir:
        mkdir_p(out_dir)
        out_prefix = os.path.join(out_dir, out_prefix)

    tic = time.time()
    n_start = 0
    if restart:
        fnames = glob.glob(os.path.join(out_dir, '*.pr'))
        if fnames:
            fnames.sort()
            n_start = int(fnames[-1].replace(
                          '{}_'.format(out_prefix), '')[:-3])
            for i in xrange(n_start):
                mol.read_dcd_step(dcd_file, i)

    for i in xrange(n_start, n_frames):
        mol.read_dcd_step(dcd_file, i)
        pr = calc_pr_numba(mol.coor()[0])

        # output the result
        out_fname = '{}_{:05d}.pr'.format(out_prefix, i+1)
        np.savetxt(out_fname, pr, fmt='%d', delimiter=',')

    toc = time.time() - tic
    if n_start < n_frames:
        logging.info('calculated P(r) for {} structures in {} s'.format(
            n_frames-n_start, toc))
        logging.info('{} s for each structure'.format(toc/(n_frames-n_start)))
    else:
        logging.info('Output already exists. To recalculate, '
                     'run using `restart=False``')
    mol.close_dcd_read(dcd_file[0])


@numba.jit(['int64[:], int64[:]',
            'float32[:], int64[:]',
            'float64[:], int64[:]'],
           nopython=True)
def bincount(distances, pr):
    for dist in distances:
        pr[int(round(dist))] += 1


def calc_pr_numba(coor):
    '''
    calculate P(r) from an array of coordinates
    when written, this was twice as fast as python method
    '''
    # calculate the euclidean distances
    dist = pdist(coor)

    # bin the distances into P(r)
    r_max = dist.max()
    n_bins = np.round(r_max).astype(int) + 1
    pr = np.zeros([n_bins, 2], dtype=np.int)
    pr[:, 0] = np.arange(n_bins)
    bincount(dist, pr[:, 1])

    return pr


def mkdir_p(path):
    '''
    make directory recursively
    adapted from http://stackoverflow.com/questions/600268/
    '''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == '__main__':
    pdb_fname = 'new_nl1_nrx1b_00001.pdb'
    dcd_fname = 'new_nl1_nrx1b_1-5.dcd'
    in_dir = '/home/schowell/data/scratch/docking'
    out_dir = '/home/schowell/data/scratch/docking/pr_test'
    main(pdb_fname, dcd_fname=dcd_fname, in_dir=in_dir, out_dir=out_dir,
         restart=True)
