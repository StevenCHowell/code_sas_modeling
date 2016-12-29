#!/usr/bin/env python
#coding:utf-8
"""
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: collection of helpful bokeh plotting functions
    Created: 12/22/2016

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
"""
import os

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Colorblind8 as palette


def errorbar(fig, x, y, xerr=None, yerr=None, color='red',
             point_kwargs={}, error_kwargs={}):

    fig.circle(x, y, color=color, **point_kwargs)

    if xerr is not None:
        x_err_x = []
        x_err_y = []
        for px, py, err in zip(x, y, xerr):
            x_err_x.append((px - err, px + err))
            x_err_y.append((py, py))
        fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

    if yerr is not None:
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(x, y, yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))
        fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)


def plot_fit(x, y, y_fit, xerr=None, yerr=None,
             save_fname='fit_comparison.html'):
    '''
    plot data and a fit line
    '''

    output_file(save_fname)


    p = figure(title=os.path.split(save_fname)[0], x_axis_label='q (1/A)',
               y_axis_label='I(q)')

    p.line(x, y_fit, color=palette[0], legend="fit", line_width=2)

    if xerr is None:
        errorbar(p, x[1:], y[1:], yerr=yerr[1:], color=palette[1],
             point_kwargs={'legend': 'raw'})
        errorbar(p, x[:1], y[:1], yerr=yerr[:1], color=palette[3],
             point_kwargs={'legend': 'extrapolated'})
    else:
        NotImplemented

    show(p)

