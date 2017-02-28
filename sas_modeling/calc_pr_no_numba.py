from __future__ import absolute_import, division, print_function

import errno
import logging
import os
import time

import numpy as np
from scipy.spatial.distance import pdist

from sasmol import sasmol  # https://github.com/madscatt/sasmol

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

def main(pdb_fname, dcd_fname='', in_dir='', out_dir='', restart=False):
    '''
    calculate the pair-distance distribution
    '''
    full_pdb_fname = os.path.join(in_dir, pdb_fname)
    assert os.path.exists(full_pdb_fname), 'No such file: {}'.format(
        full_pdb_fname)
    mol = sasmol.SasMol(0)
    mol.read_pdb(full_pdb_fname)

    if dcd_fname:
        full_dcd_fname = os.path.join(in_dir, dcd_fname)
        assert os.path.exists(full_dcd_fname), 'No such file: {}'.format(
            full_dcd_fname)
        dcd_file = mol.open_dcd_read(os.path.join(in_dir, full_dcd_fname))
        n_frames = dcd_file[2]
        out_prefix = dcd_fname[:-4]

    else:
        n_frame = 1
        out_prefix = pdb_fname[:-4]

    if out_dir:
        mkdir_p(out_dir)
        out_prefix = os.path.join(out_dir, out_prefix)

    tic = time.time()
    for i in xrange(n_frames):
        mol.read_dcd_step(dcd_file, i)
        pr = calc_pr_python(mol.coor()[0])

        # output the result
        out_fname = '{}_{:05d}.pr'.format(out_prefix, i+1)
        np.savetxt(out_fname, pr, fmt='%d', delimiter=',')

    toc = time.time() - tic
    logging.info('calculated P(r) for {} structures in {} s'.format(n_frames,
                                                                    toc))
    logging.info('{} s for each structure'.format(toc/n_frames))

    mol.close_dcd_read(dcd_file[0])


def calc_pr_python(coor):
    # calculate the euclidean distances
    dist = pdist(coor)

    # bin the distances into P(r)
    r_max = dist.max()
    n_bins = np.round(r_max).astype(int) + 1
    pr = np.empty([n_bins, 2], dtype=np.int)
    pr[:, 0] = np.arange(n_bins)
    int_dist = np.round(dist).astype(np.int)
    pr[:, 1] = np.bincount(int_dist)

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
    dcd_fname = 'new_nl1_nrx1b.dcd'
    # dcd_fname = 'new_nl1_nrx1b_1-5.dcd'
    in_dir = '/home/schowell/data/scratch/docking'
    out_dir = '/home/schowell/data/scratch/docking/pr_test'
    main(pdb_fname, dcd_fname=dcd_fname, in_dir=in_dir, out_dir=out_dir,
         restart=True)


