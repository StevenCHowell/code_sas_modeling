from __future__ import absolute_import, division, print_function

import os
import errno

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


def get_files(pr_dir, iq_dir):
    pr_files = glob.glob(os.path.join(pr_dir, '*.pr'))
    iq_files = glob.glob(os.path.join(iq_dir, '*.iq'))
    pr_files.sort()
    iq_files.sort()

    n_iq = len(iq_files)
    n_pr = len(pr_files)
    if n_iq != n_pr:
        logging.warning('mismatch number of files, n_iq: {}, n_pr: {}'.format(n_iq, n_pr))
    print('found {} P(r) and {} I(Q) files'.format(n_pr, n_iq))
    return pr_files, iq_files


def load_pr(pr_files):
    n_pr = len(pr_files)
    pr_data_l = []
    n_r = np.empty(n_pr, dtype=int)

    # load in all the data set
    for i, pr_file in enumerate(pr_files):
        n_data = np.loadtxt(pr_file, delimiter=',', dtype=int)
        pr_data_l.append(n_data[:, 1])
        n_r[i] = len(n_data)

    r_max = n_r.max()
    r = np.arange(r_max)
    pr_data = np.zeros([n_pr, r_max], dtype=int)
    for i, n_data in enumerate(pr_data_l):
        pr_data[i, :len(n_data)] = n_data

    return r, pr_data


def load_iq(iq_files):
    iq_data = []

    q_max = 0.18  # only use data up to 0.18 1/A

    # load in the first data set to setup the q-mask
    first_data = np.loadtxt(iq_files[0])
    q_mask = first_data[:, 0] <= q_max
    first_data = first_data[q_mask]
    iq_data.append(first_data[:, 1])

    # load in the rest of the data
    for iq_file in iq_files[1:]:
        x_data = np.loadtxt(iq_file)
        x_data = x_data[q_mask]
        assert np.allclose(x_data[0, 1], first_data[0, 1]), 'ERROR: data not normalized to I(0)'
        assert np.allclose(x_data[:, 0], first_data[:, 0]), 'ERROR: data not on same Q-grid'
        iq_data.append(x_data[:, 1])

    iq_data = np.array(iq_data)
    q = x_data[:, 0]

    return q, iq_data