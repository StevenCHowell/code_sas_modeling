#!/usr/bin/env python
#coding:utf-8
'''
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: the scattering of a geometric shapes
    Created: 02/03/2017

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
'''

import numpy as np
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def sphere(r, q, p_scat, p_sol, scale=1.0, bkg=0.0):
    if q[0] <= 0:
        logging.warning('skipping invalid Q-value: {}'.format(q[0]))
        q = q[1:]

    pq = np.empty([len(q), 2])
    pq[:, 0] = q

    v = 4 / 3 * np.pi * r**3
    dp = p_scat - p_sol
    qr = q * r

    pq[:, 1] = scale / v * (3 * v * dp * (np.sin(qr) - qr * np.cos(qr)) / qr**3
                            )**2 + bkg

    a_to_cm = 1e8
    pq[:, 1] *= a_to_cm  # convert from A^{-1} to cm^{-1}

    return pq


if __name__=="__main__":
    # to match https://www.ncnr.nist.gov/resources/sansmodels/Sphere.html

    # data from https://www.ncnr.nist.gov/resources/sansmodels/Sphere.html
    ref = np.loadtxt('ncnr_sphere.iq')

    scale = 1.0
    bkg = 0.0
    p_scat = 2e-6
    p_sol = 1e-6
    r = 60.0

    # q_min = ref[0, 0]
    # q_max = ref[-1, 0]
    # n_points = len(ref)
    # q = np.logspace(np.log10(q_min), np.log10(q_max), n_points)

    pq = sphere(r, ref[:, 0], p_scat, p_sol, scale=scale, bkg=bkg)

    # assert np.allclose(ref[:, 1], pq[:, 1], atol=1e-5)
    if not np.allclose(ref, pq, atol=1e-5):
        logging.error('results do not match')

    logging.debug('\m/ >.< \m/>')