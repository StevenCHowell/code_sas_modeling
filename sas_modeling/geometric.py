#!/usr/bin/env python
# coding:utf-8
'''
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: the scattering of a geometric shapes
    Created: 02/03/2017

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
'''
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import scipy.spatial.distance

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


class circle:
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center

    def perimiter(self):
        return 2 * np.pi * self.r

    def area(self):
        return np.pi * self.r ** 2

    def in_or_out(self, points):
        distances = scipy.spatial.distance.cdist(self.center, points)
        in_mask = distance < self.r
        return in_mask


class ellipse:
    def __init__(self, a, b, center=np.array([0, 0]),
                 orientation=np.array([1, 0])):
        self.a = a
        self.b = b
        self.f = np.sqrt(a**2 - b**2)
        self.center = center
        self.orientation = np.linalg.norm(orientation)
        self.f1 = self.center + self.orientation * f
        self.f2 = self.center - self.orientation * f

    def area(self):
        return np.pi * self.a * self.b

    def in_or_out(self, points):
        distance_f1 = scipy.spatial.distance.cdist(self.f1, points)
        distance_f2 = scipy.spatial.distance.cdist(self.f2, points)
        distance = distance_f1 + distance_d2
        in_mask = distance < 2 * a
        return in_mask


class rectangle:
    def __init__(self, s1, s2, center=np.array([0, 0]),
                 orientation=np.array([1, 0])):
        self.s1 = s1
        self.s2 = s2
        self.center = center
        self.orientation = np.linalg.norm(orientation)

    def area(self):
        return self.s1 * self.s2

    def in_or_out(self, points):
        axis = self.center + self.orientation
        d_parl = points.dot(axis.T)
        p_mag = np.linalg.norm(points)
        theta = np.arccos(d_parl/p_mag)
        d_perp = p_mag * np.sin(theta)

        in_mask_par = d_parl < self.s1
        in_mask_per = d_perp < self.s2
        in_mask = in_mask_par & in_mask_per

        return in_mask


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
