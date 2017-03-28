#!/usr/bin/env python
# coding:utf-8
'''
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: definiton of geometric shapes and their scattering
    Created: 02/03/2017

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
'''
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import scipy.spatial.distance

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


class Ellipse(object):
    r'''
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Parameters
    ----------
    a : float
        radius along the semi-major axis
    b : float
        radius along the semi-minor axis
    center : [float, float], optional
        point at the center of the ellipse
    orientation : [fload, floadt], optional
        unit vector along the semi-major axis

    Attributes
    ----------
    a : float
        radius along the semi-major axis
    b : float
        radius along the semi-minor axis
    center : [float, float], optional
        point at the center of the ellipse
    orientation : [fload, floadt], optional
        unit vector along the semi-major axis

    Methods
    -------
    area
        return the area of the ellipse
    radius(d=[1.0, 1.0])
        radius along an input direction
    in_or_out(points)
        create the `in_points`, `out_points`, and `n_in` points attributes
        cooresponding to which of the input `points` are inside/outside of
        the ellipse
    fill_with_points(density)
        fill the ellipse with points cooresponding to the input `density`

    Examples
    --------
    >>> e = Ellipse(2.0, 1.0, orientation=[1.0, 0.0], center=[0.0, 0.0])
    >>> e.radius([1.0, 0.0])
    2.0
    >>> e.radius([0.0, 1.0])
    1.0
    >>> e.radius([1.0, 1.0])
    1.2649110640673518
    >>> e.radius([1.0, np.sqrt(3)])
    1.1094003924504583
    >>> e.radius([np.sqrt(3), 1.0])
    1.5118578920369088

    Note that this satifies the condition that pf1 + pf2 = 2 * a
    >>> d = np.random.rand(2)
    >>> d /= np.linalg.norm(d)
    >>> r = e.radius(d)
    >>> p = r * d
    >>> pf1 = np.linalg.norm(e.f1 - p)
    >>> pf2 = np.linalg.norm(e.f2 - p)
    >>> np.allclose(pf1 + pf2, 2 * e.a)
    True

    This also works for rotated and offset ellipses
    >>> c = np.array([5.0, -2.0])
    >>> e = Ellipse(2.0, 1.0, orientation=[1.0, 1.0], center=c)
    >>> e.radius(c + [1.0, 1.0])
    2.0
    >>> e.radius(c + [-1.0, 1.0])
    1.0

    For an offset ellipse, demonstrating the radius value indeed satifies
    the condition that pf1 + pf2 = 2 * a requires taking into account the
    center point
    >>> d = np.random.rand(2)
    >>> d /= np.linalg.norm(d)
    >>> r = e.radius(d)
    >>> p = r * (d - c) / np.linalg.norm(d - c)
    >>> pf1 = np.linalg.norm(e.f1 - c - p)
    >>> pf2 = np.linalg.norm(e.f2 - c - p)
    >>> np.allclose(pf1 + pf2, 2 * e.a)
    True

    '''

    def __repr__(self):
        return ('ellipse with center: {}, a: {}, b: {}, and '
                'orientation: {}'.format(self.center, self.a, self.b,
                                         self.orientation))

    def __init__(self, a, b, center=[0, 0], orientation=[1, 0]):
        if a > b:
            self.a = a
            self.b = b
        else:
            logging.warning('semi-major and semi-minor axes input inverted')
            self.a = b
            self.b = a

        self.f = np.sqrt(self.a ** 2 - self.b ** 2)
        self.center = np.array(center)
        self._orientation = orientation / np.linalg.norm(orientation)
        self.f1 = self.center + self.orientation * self.f
        self.f2 = self.center - self.orientation * self.f

    @property
    def orientation(self):
        return self._orientation
    @orientation.setter
    def orientation(self, val):
        self._orientation = val / np.linalg.norm(val)

    @property
    def area(self):
        return np.pi * self.a * self.b

    def radius(self, d):
        theta = -np.arctan2(self.orientation[1], self.orientation[0])
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        d_p = d - self.center
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        d_pp = R.dot(d_p.reshape(2, 1)).reshape(2)

        if d_pp[0] == 0.0:
            return self.b

        else:
            m = d_pp[1] / d_pp[0]  # slope
            a2 = self.a ** 2
            b2 = self.b ** 2
            x2 = (b2 * a2 / (b2 + a2 * m ** 2))
            y2 = b2 * (1 - x2 / a2)
            return np.sqrt(x2 + y2)

    def _which_are_in(self, points):
        distance_f1 = scipy.spatial.distance.cdist(self.f1.reshape(1, 2),
                                                   points)
        distance_f2 = scipy.spatial.distance.cdist(self.f2.reshape(1, 2),
                                                   points)
        distance = distance_f1 + distance_f2
        in_mask = distance.reshape(-1) < 2 * self.a
        return in_mask

    def in_or_out(self, points):
        in_mask = self._which_are_in(points)
        self.in_points = points[in_mask]
        self.out_points = points[np.invert(in_mask)]
        self.n_in = len(self.in_points)

    def fill_with_points(self, density):
        n = np.round(density * self.area).astype(int)
        n_in = 0
        xy_range = 2 * (2.1 * self.a - self.b)
        xy_min = self.center - xy_range / 2
        in_points = []
        while n_in < n:
            n_remaining = n - n_in
            points = np.random.rand(int(1.7 * n_remaining), 2)
            points = points * xy_range + xy_min
            in_mask = self._which_are_in(points)
            n_in += in_mask.sum()
            in_points.append(points[in_mask])
        in_points = np.vstack(in_points)[:n]
        self.in_points = in_points
        self.n_in = n


class Circle(Ellipse):
    def __repr__(self):
        return 'circle centered at {} with r={}'.format(self.center,
                                                        self.radius)

    def __init__(self, radius, center=[0, 0]):
        self.radius = radius
        self.a = radius
        self.b = radius
        self.center = np.array(center)
        self.orientation = np.array([1, 0])
        self.f = 0.0  # np.sqrt(self.a ** 2 - self.b ** 2)
        self.f1 = self.center + self.orientation * self.f
        self.f2 = self.center - self.orientation * self.f

    @property
    def perimiter(self):
        return 2 * np.pi * self.radius


class Rectangle(object):
    def __repr__(self):
        return ('rectangle with center: {}, side 1: {}, side 2: {}, and '
                'orientation: {}'.format(self.center, self.side1,
                                         self.side2, self.orientation))

    def __init__(self, side1, side2, center=[0, 0], orientation=[1, 0]):
        self.side1 = side1
        self.side2 = side2
        self.center = np.array(center)
        self.orientation = orientation / np.linalg.norm(orientation)

    @property
    def area(self):
        return self.side1 * self.side2

    def in_or_out(self, points):
        v_points = points - self.center
        d_parl = v_points.dot(self.orientation.T)
        p_mag = np.linalg.norm(v_points, axis=1)
        if np.alltrue(v_points[p_mag.argmin()] == 0):
            p_mag[p_mag.argmin()] = 1.0

        arg = d_parl / p_mag
        gt1 = arg[arg > 1]
        lt1 = arg[arg < -1]
        for i, val in enumerate(gt1):
            assert np.allclose(val, 1), ('magnitude of cosine argument is '
                                         'not positive definite')
            gt1[i] = 1
        for i, val in enumerate(lt1):
            assert np.allclose(val, -1), ('magnitude of cosine argument is '
                                         'not positive definite')
            lt1[i] = 1
        arg[arg > 1] = gt1
        arg[arg < -1] = lt1

        assert np.alltrue(np.abs(arg) <= 1), 'invalid cosine arg'
        theta = np.arccos(arg)
        d_perp = p_mag * np.sin(theta)

        in_mask_par = np.abs(d_parl) < self.side1
        in_mask_per = d_perp < self.side2
        in_mask = in_mask_par & in_mask_per

        self.in_points = points[in_mask]
        self.out_points = points[np.invert(in_mask)]
        self.n_in = len(self.in_points)


class Square(Rectangle):
    def __repr__(self):
        return 'square with center: {}, side: {}, and orientation: {}'.format(
            self.center, self.side, self.orientation)

    def __init__(self, side, center=[0,0], orientation=[1,0]):
        self.side = side
        self.side1 = side
        self.side2 = side
        self.center = np.array(center)
        orientation = np.array(orientation)
        self.orientation = orientation / np.linalg.norm(orientation)


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


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # import bokeh.plotting
    # import bokeh.layouts # gridplot
    # from bokeh.palettes import Dark2_7 as palette


    # e = Ellipse(50, 50)
    # e.fill_with_points(1)

    # c = Circle(30)
    # c.fill_with_points(0.5)

    # x = np.arange(-110, 110)
    # y = np.arange(-110, 110)
    # grid_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # p = bokeh.plotting.figure()
    # for i, shape in enumerate(['square', 'circle', 'ellipse', 'rectangle']):

        # if shape == 'circle':

            # c = Circle(10, center=[-40, -8])

            # in_points = c.in_or_out(grid_points)
            # out_points = np.invert(in_points)

            # n_in = in_points.sum()
            # n_out = out_points.sum()
            # bokeh.plotting.output_file('circle_{}.html'.format(c.radius))

        # elif shape == 'ellipse':
            # e = Ellipse(35, 10, center=[30, -30], orientation=[-1, 2])
            # in_points = e.in_or_out(grid_points)
            # out_points = np.invert(in_points)

            # n_in = in_points.sum()
            # n_out = out_points.sum()
            # bokeh.plotting.output_file('ellipse_{}_{}'.format(e.a, e.b))

        # elif shape == 'square':
            # s = Square(30, center=[-40, 75], orientation=[-1, -5])
            # print(s.area)

            # in_points = s.in_or_out(grid_points)
            # out_points = np.invert(in_points)

            # n_in = in_points.sum()
            # n_out = out_points.sum()
            # bokeh.plotting.output_file('square_{}.html'.format(s.side))

        # elif shape == 'rectangle':
            # r = Rectangle(19, 14, center=[30, 60], orientation=[-1, 3])  #, center=[0.1, 0.1])
            # print(r.area)

            # in_points = r.in_or_out(grid_points)
            # out_points = np.invert(in_points)

            # n_in = in_points.sum()
            # n_out = out_points.sum()
            # bokeh.plotting.output_file('reccangle_{}_{}.html'.format(
                # r.side1, r.side2))

        # bokeh.plotting.output_file('shapes.html')

        # p.circle(grid_points[in_points, 0], grid_points[in_points, 1],
                 # color=palette[i])

    # # p.circle(grid_points[out_points, 0], grid_points[out_points, 1],
             # # color=palette[1])
    # bokeh.plotting.show(p)

    # '''
    # # data from https://www.ncnr.nist.gov/resources/sansmodels/Sphere.html
    # ref = np.loadtxt('ncnr_sphere.iq')

    # scale = 1.0
    # bkg = 0.0
    # p_scat = 2e-6
    # p_sol = 1e-6
    # r = 60.0

    # # q_min = ref[0, 0]
    # # q_max = ref[-1, 0]
    # # n_points = len(ref)
    # # q = np.logspace(np.log10(q_min), np.log10(q_max), n_points)

    # pq = sphere(r, ref[:, 0], p_scat, p_sol, scale=scale, bkg=bkg)

    # # assert np.allclose(ref[:, 1], pq[:, 1], atol=1e-5)
    # if not np.allclose(ref, pq, atol=1e-5):
        # logging.error('results do not match')
    # '''

    # logging.debug('\m/ >.< \m/>')
