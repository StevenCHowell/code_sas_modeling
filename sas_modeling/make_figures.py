#!/usr/bin/env python
#coding:utf-8
"""
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: collection of helpful bokeh plotting functions
    Created: 12/22/2016

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
"""
import numpy as np

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Colorblind8 as palette
from bokeh.layouts import gridplot

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


def round_to_n(x, n=3):
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def plot_guinier_fit(q2, log_iq, fit_line, i0, i0_err, rg, rg_err, dlog_iq,
                     q_range, xerr=None, save_fname='fit_comparison.html'):
    '''
    plot data and a fit line
    '''

    output_file(save_fname)
    n_round = 6

    i0 = round_to_n(i0, n=n_round)
    rg = round_to_n(rg, n=n_round)
    if i0_err == 0:
        i0_err = 'NA'
    else:
        i0_err = round_to_n(i0_err, n=n_round)

    if rg_err == 0:
        rg_err = 'NA'
    else:
        rg_err = round_to_n(rg_err, n=n_round)


    title = '{}, I(0) = {}+/-{}, Rg = {}+/-{}, q-range: [{}, {}]'.format(
        save_fname.split('.')[0], i0, i0_err, rg, rg_err, q_range[0],
        q_range[1])
    p = figure(title=title, x_axis_label='q^2 (1/A^2)', y_axis_label='ln(I(q))')

    p.line(q2, fit_line, color=palette[0], legend="fit", line_width=2)

    if xerr is None:
        errorbar(p, q2[1:], log_iq[1:], yerr=dlog_iq[1:], color=palette[1],
                 point_kwargs={'legend': 'raw'})
        errorbar(p, q2[:1], log_iq[:1], yerr=dlog_iq[:1], color=palette[3],
                 point_kwargs={'legend': 'extrapolated'})
    else:
        NotImplemented

    show(p)


def plot_iq_and_guinier(q, iq, diq, save_fname='I(q)_and_guinier.html'):
    '''
    plot data using linear, log, and Guinier axes
    '''

    output_file(save_fname)

    p0 = figure(title='linear', x_axis_label='q (1/A)', y_axis_label='I(q)')
    errorbar(p0, q, iq, yerr=diq, color=palette[1])

    p1 = figure(title='log', x_axis_label='q (1/A)', y_axis_label='I(q)',
                x_axis_type='log', y_axis_type='log')  #,  x_range=p0.x_range, y_range=p0.y_range)
    errorbar(p1, q, iq, yerr=diq, color=palette[1])

    x = q ** 2
    y = np.log(iq)

    p2 = figure(title='Guinier', x_axis_label='q^2 (1/A^2)',
               y_axis_label='ln(I(q))')

    dy_skew = (np.log(iq + diq) - np.log(iq - diq)) / 2.0
    # p3 = figure(title='Guinier (skewed errorbars)', x_axis_label='q^2 (1/A^2)',
                # y_axis_label='ln(I(q))', x_range=p2.x_range,
                # y_range=p2.y_range)
    errorbar(p2, x, y, yerr=dy_skew, color=palette[0],
             point_kwargs={'legend': 'log(iq +/- averaged diq)'})

    dy = diq / iq
    errorbar(p2, x, y, yerr=dy, color=palette[1],
             point_kwargs={'legend': 'log(iq +/- diq)'})



    r0 = [p0, p1]
    r1 = [p2]  #, p3]

    layout = gridplot([r0, r1])

    show(layout)
